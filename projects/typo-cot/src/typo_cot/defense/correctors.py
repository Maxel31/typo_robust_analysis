"""校正器ラダー: 弱 (pyspellchecker) / 中 (seq2seq) / 強 (LLM) の統一インターフェース.

- PySpellCorrector: rebuttal の make_spellfix_dataset.py:correct_text と同一ロジック
  (単語単位・非文脈型。pyspellchecker==0.9.0 固定で byte 再現可能)
- Seq2SeqCorrector: ニューラル文脈型 (neuspell 系が導入不可の場合の T5 系 GEC 代替)。
  行単位で seq2seq 校正し、質問文の改行構造 (選択肢行など) を保存する。
- LLMCorrector: 保守的プロンプト ("typo のみ修正・他は一切変更禁止") + 温度0。
  出力は <corrected>...</corrected> タグから抽出し、失敗時1回リトライ。

GPU 依存はすべて遅延ロード。テストでは generate_fn を注入してモックする。
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

logger = logging.getLogger(__name__)

# rebuttal の make_spellfix_dataset.py:29 と同一
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

_CORRECTED_TAG_RE = re.compile(r"<corrected>(.*?)</corrected>", re.DOTALL)

# 保守的 LLM 校正プロンプト (温度0・修正後テキストのみを出力させる)
LLM_SYSTEM_PROMPT = "You are a careful and conservative proofreader."

LLM_USER_TEMPLATE = (
    "The following text may contain typographical errors (misspelled words). "
    "Fix ONLY the typos. Do not rephrase, do not add or remove words, and do not "
    "change punctuation, numbers, formatting, line breaks, or any correctly "
    "spelled words. If there are no typos, reproduce the text exactly as given. "
    "Output ONLY the corrected text, wrapped between <corrected> and "
    "</corrected> tags.\n\n<text>\n{text}\n</text>"
)

LLM_RETRY_REMINDER = (
    "\n\nReminder: your previous reply did not follow the required format. "
    "Reply with ONLY the corrected text wrapped between <corrected> and "
    "</corrected> tags, and nothing else."
)


def apply_case(template: str, corrected: str) -> str:
    """訂正結果に元の語のケースパターンを適用する (rebuttal 実装と同一)."""
    if template.isupper():
        return corrected.upper()
    if template[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


class Corrector(ABC):
    """校正器の共通インターフェース."""

    name: str = "base"

    @abstractmethod
    def correct(self, text: str) -> str:
        """テキストを校正して返す."""

    def correct_batch(self, texts: list[str]) -> list[str]:
        """複数テキストを校正する (デフォルトは逐次)."""
        return [self.correct(t) for t in texts]


class PySpellCorrector(Corrector):
    """pyspellchecker による非文脈・単語単位の校正 (rebuttal 1段目と同一ロジック)."""

    name = "pyspell"

    def __init__(self, language: str = "en") -> None:
        from spellchecker import SpellChecker

        self._spell = SpellChecker(language=language)

    def correct(self, text: str) -> str:
        corrected, _ = self.correct_with_changes(text)
        return corrected

    def correct_with_changes(self, text: str) -> tuple[str, list[dict]]:
        """テキストを単語単位でスペル訂正する.

        アルファベット語のみ対象 (数値・記号・選択肢ラベルはそのまま)。
        辞書に存在する語・1文字語は変更しない。訂正候補が無い語もそのまま。
        rebuttal の make_spellfix_dataset.py:correct_text と同一の動作。

        Returns:
            (訂正後テキスト, 変更ログ [{original, corrected, start}])
        """
        spell = self._spell
        changes: list[dict] = []
        out: list[str] = []
        last = 0
        for m in WORD_RE.finditer(text):
            word = m.group(0)
            out.append(text[last : m.start()])
            last = m.end()

            lower = word.lower()
            # 1文字語 (選択肢ラベル A-D, 冠詞 a 等) は訂正対象外
            if len(word) <= 1 or lower in spell:
                out.append(word)
                continue
            cand = spell.correction(lower)
            if cand is None or cand == lower:
                out.append(word)
                continue
            fixed = apply_case(word, cand)
            changes.append({"original": word, "corrected": fixed, "start": m.start()})
            out.append(fixed)
        out.append(text[last:])
        return "".join(out), changes


class Seq2SeqCorrector(Corrector):
    """seq2seq (T5 系 GEC / スペル訂正) モデルによるニューラル文脈型校正.

    質問文は複数行 (選択肢行を含む) になり得るため、行単位で校正して
    改行構造を保存する。空行はモデルを呼ばずそのまま通す。
    """

    name = "neural"

    def __init__(
        self,
        model_name: str = "ai-forever/T5-large-spell",
        prefix: str = "grammar: ",
        device: str | None = None,
        max_new_tokens: int = 256,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        """初期化.

        Args:
            model_name: HF seq2seq モデル名
            prefix: モデル固有のタスクプレフィックス (T5-large-spell は "grammar: ")
            device: 実行デバイス ("cuda"/"cpu"、None は自動)
            max_new_tokens: 1行あたりの最大生成トークン数
            generate_fn: テスト用注入点。行テキスト -> 校正済み行テキスト
        """
        self.model_name = model_name
        self.prefix = prefix
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._generate_fn = generate_fn

    def _build_generate_fn(self) -> Callable[[str], str]:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"seq2seq 校正モデルをロード: {self.model_name} ({device})")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        model.eval()

        def _fn(line: str) -> str:
            inputs = tokenizer(
                self.prefix + line, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        return _fn

    @property
    def generate_fn(self) -> Callable[[str], str]:
        if self._generate_fn is None:
            self._generate_fn = self._build_generate_fn()
        return self._generate_fn

    def correct(self, text: str) -> str:
        lines = text.split("\n")
        out_lines = []
        for line in lines:
            if line.strip() == "":
                out_lines.append(line)
            else:
                out_lines.append(self.generate_fn(line))
        return "\n".join(out_lines)


class LLMCorrector(Corrector):
    """LLM (既定: Qwen2.5-7B-Instruct) による保守的校正.

    評価モデルと別家族のモデルを使い「自分で自分の typo を直す」交絡を回避する。
    温度0 (greedy)。出力から <corrected>...</corrected> を抽出し、
    パース失敗時はフォーマット再指示を付けて1回だけリトライ。
    それでも失敗した場合は原文をそのまま返す (meta["parse_failed"]=True)。
    """

    name = "llm"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 1024,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        """初期化.

        Args:
            model_name: HF causal LM 名
            device: 実行デバイス (None は自動)
            max_new_tokens: 校正出力の最大トークン数
            generate_fn: テスト用注入点。ユーザープロンプト -> 応答テキスト
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._generate_fn = generate_fn

    def _build_generate_fn(self) -> Callable[[str], str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LLM 校正モデルをロード: {self.model_name} ({device})")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        model.eval()

        def _fn(user_prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(
                out_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            )

        return _fn

    @property
    def generate_fn(self) -> Callable[[str], str]:
        if self._generate_fn is None:
            self._generate_fn = self._build_generate_fn()
        return self._generate_fn

    @staticmethod
    def _parse(response: str) -> str | None:
        m = _CORRECTED_TAG_RE.search(response)
        if m is None:
            return None
        return m.group(1).strip("\n")

    def correct(self, text: str) -> str:
        corrected, _ = self.correct_with_meta(text)
        return corrected

    def correct_with_meta(self, text: str) -> tuple[str, dict]:
        """校正し、パース成否などのメタ情報も返す.

        Returns:
            (校正後テキスト, {"parse_failed": bool, "n_calls": int, "raw_response": str})
        """
        prompt = LLM_USER_TEMPLATE.format(text=text)
        response = self.generate_fn(prompt)
        parsed = self._parse(response)
        n_calls = 1
        if parsed is None:
            # greedy では同一プロンプト→同一出力のため、フォーマット再指示を付与
            retry_prompt = prompt + LLM_RETRY_REMINDER
            response = self.generate_fn(retry_prompt)
            parsed = self._parse(response)
            n_calls = 2
        if parsed is None:
            logger.warning("LLM 校正出力のパースに2回失敗。原文を返します。")
            return text, {
                "parse_failed": True,
                "n_calls": n_calls,
                "raw_response": response,
            }
        return parsed, {
            "parse_failed": False,
            "n_calls": n_calls,
            "raw_response": response,
        }


def create_corrector(kind: str, **kwargs) -> Corrector:
    """校正器のファクトリ関数.

    Args:
        kind: "pyspell" / "neural" / "llm"
        **kwargs: 各校正器のコンストラクタ引数

    Returns:
        Corrector インスタンス

    Raises:
        ValueError: 不明な kind の場合
    """
    if kind == "pyspell":
        return PySpellCorrector(**kwargs)
    if kind == "neural":
        return Seq2SeqCorrector(**kwargs)
    if kind == "llm":
        return LLMCorrector(**kwargs)
    raise ValueError(f"不明な校正器: {kind} (利用可能: pyspell, neural, llm)")
