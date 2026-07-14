"""摂動データセット作成モジュール."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from typo_cot.perturbation.generator import (
    CharacterPerturbationGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class PerturbedToken:
    """摂動されたトークンの情報.

    Attributes:
        token_index: トークンインデックス（プロンプト全体での位置）
        original_token: 元のトークン
        perturbed_token: 摂動後のトークン
        importance_score: 重要度スコア
        perturbation_type: 摂動の種類
        char_position: 摂動を適用した文字位置（トークン内）
    """

    token_index: int
    original_token: str
    perturbed_token: str
    importance_score: float
    perturbation_type: str
    char_position: int


@dataclass
class PerturbedSample:
    """摂動されたサンプル.

    Attributes:
        sample_id: 元のサンプルID
        original_question: 元の質問文
        perturbed_question: 摂動後の質問文
        perturbed_tokens: 摂動されたトークンのリスト
        choices: 元の選択肢リスト（MMLU/MMLU-Proの場合）
        correct_answer: 正解
        subset: サブセット名
        context: 摂動後のコンテキスト（SQuAD v2の場合）
        original_context: 元のコンテキスト（SQuAD v2の場合）
        perturbed_choices: 摂動後の選択肢リスト（MMLU/MMLU-Proの場合）
    """

    sample_id: str
    original_question: str
    perturbed_question: str
    perturbed_tokens: list[PerturbedToken]
    choices: list[str] | None
    correct_answer: str
    subset: str | None
    context: str | None = None
    original_context: str | None = None
    perturbed_choices: list[str] | None = None


@dataclass
class PerturbedDataset:
    """摂動データセット.

    Attributes:
        metadata: メタデータ
        samples: 摂動サンプルのリスト
    """

    metadata: dict
    samples: list[PerturbedSample]

    def save(self, output_path: Path) -> None:
        """データセットを保存.

        Args:
            output_path: 出力パス
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "samples": [asdict(s) for s in self.samples],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"摂動データセットを保存: {output_path}")

    @classmethod
    def load(cls, path: Path) -> "PerturbedDataset":
        """データセットを読み込み.

        Args:
            path: 入力パス

        Returns:
            摂動データセット
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for s in data["samples"]:
            perturbed_tokens = [PerturbedToken(**pt) for pt in s["perturbed_tokens"]]
            s["perturbed_tokens"] = perturbed_tokens
            samples.append(PerturbedSample(**s))

        return cls(metadata=data["metadata"], samples=samples)


class PerturbedDatasetCreator:
    """摂動データセット作成クラス."""

    def __init__(
        self,
        baseline_dir: Path,
        num_perturbations: int,
        seed: int = 42,
        random_perturbation: bool = False,
        include_choices: bool = True,
        bottom_k_perturbation: bool = False,
    ) -> None:
        """初期化.

        Args:
            baseline_dir: Phase 1の結果ディレクトリ
            num_perturbations: 摂動回数
            seed: ランダムシード
            random_perturbation: Trueの場合、重要度上位k個を除外してランダムに摂動
                （例: k=4なら上位4個以外からランダムに4個選択）
            include_choices: Trueの場合、選択肢も摂動対象に含める
            bottom_k_perturbation: Trueの場合、重要度下位k個のトークンに摂動（Anti-LRP）
        """
        self.baseline_dir = Path(baseline_dir)
        self.num_perturbations = num_perturbations
        self.seed = seed
        self.random_perturbation = random_perturbation
        self.include_choices = include_choices
        self.bottom_k_perturbation = bottom_k_perturbation
        self.generator = CharacterPerturbationGenerator(seed=seed)

        # 設定ファイルを読み込み
        config_path = self.baseline_dir / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # 結果ファイルを読み込み
        results_path = self.baseline_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"results.json が見つかりません: {results_path}")

        try:
            with open(results_path, encoding="utf-8") as f:
                self.results = json.load(f)
        except json.JSONDecodeError as e:
            # ファイルサイズと破損位置を報告
            file_size = results_path.stat().st_size
            raise ValueError(
                f"results.json が破損しています: {results_path}\n"
                f"  エラー位置: 行 {e.lineno}, 列 {e.colno} (文字位置 {e.pos})\n"
                f"  ファイルサイズ: {file_size:,} バイト\n"
                f"  原因: ファイルが不完全か、書き込み中に中断された可能性があります。\n"
                f"  対処法: Phase 1 の推論を再実行してください。"
            ) from e

        logger.info(f"Phase 1結果を読み込み: {len(self.results)} サンプル")
        logger.info(f"摂動対象: {'選択肢含む' if include_choices else '質問文のみ'}")

    def _load_importance_scores(self, sample_id: str) -> dict | None:
        """重要度スコアを読み込み.

        Args:
            sample_id: サンプルID

        Returns:
            重要度スコアデータ、または見つからない場合はNone
        """
        score_path = self.baseline_dir / "importance_scores" / f"{sample_id}.pt"
        if not score_path.exists():
            logger.warning(f"重要度スコアが見つかりません: {score_path}")
            return None

        return torch.load(score_path, map_location="cpu", weights_only=False)

    def _should_skip_token(self, token: str) -> bool:
        """摂動対象外のトークンかどうかを判定.

        以下のトークンは摂動対象外:
        - 数値のみのトークン（例: "123", "42"）
        - 選択肢記号（例: "(A)", "A.", "B)", "A"）
        - 括弧のみのトークン（例: "(", ")"）

        Args:
            token: 判定するトークン

        Returns:
            スキップすべき場合はTrue
        """
        import re

        # 空白を除去して判定
        stripped = token.strip()

        if not stripped:
            return True

        # 数値のみのトークン（カンマや小数点を含む数値も対象）
        if re.match(r"^[\d,\.]+$", stripped):
            return True

        # 括弧のみのトークン
        if re.match(r"^[\(\)]+$", stripped):
            return True

        # 選択肢記号のパターン
        # (A), (B), ..., (J) 形式
        if re.match(r"^\([A-Ja-j]\)$", stripped):
            return True

        # A., B., ..., J. 形式
        if re.match(r"^[A-Ja-j]\.$", stripped):
            return True

        # A), B), ..., J) 形式
        if re.match(r"^[A-Ja-j]\)$", stripped):
            return True

        # A:, B:, ..., J: 形式
        if re.match(r"^[A-Ja-j]:$", stripped):
            return True

        # 単独の選択肢文字（A, B, ..., J）- 常にスキップ
        return bool(re.match(r"^[A-Ja-j]$", stripped))

    def _get_question_tokens(
        self,
        importance_data: dict,
    ) -> list[tuple[int, str, float]]:
        """摂動対象トークンとスコアを取得（数値・選択肢記号は除外）.

        MMLU/MMLU-Pro: 質問文＋選択肢（include_choices=Trueの場合）または質問文のみ
        SQuAD v2: context＋質問文

        Args:
            importance_data: 重要度スコアデータ

        Returns:
            (トークンインデックス, トークン文字列, スコア) のリスト
        """
        tokens = importance_data.get("tokens", [])
        offset_mapping = importance_data.get("offset_mapping", [])

        # include_choicesに応じてtoken_scoresを選択
        if self.include_choices:
            # 選択肢を含むトークンスコアを使用
            token_scores = importance_data.get(
                "token_scores_with_choices",
                importance_data.get("token_scores", []),  # フォールバック
            )
        else:
            # 質問文のみのトークンスコアを使用
            token_scores = importance_data.get("token_scores", [])

        # SQuAD v2の場合はcontextも含める
        context_char_start = importance_data.get("context_char_start")
        if context_char_start is not None:
            # SQuAD v2: context_start から question_end まで
            perturbation_start = context_char_start
            perturbation_end = importance_data.get("question_char_end")
        else:
            # MMLU/MMLU-Pro
            if self.include_choices:
                # question_start から choices_end まで
                perturbation_start = importance_data.get("question_char_start")
                perturbation_end = importance_data.get(
                    "question_with_choices_end",
                    importance_data.get("question_char_end"),
                )
            else:
                # 質問文のみ
                perturbation_start = importance_data.get("question_char_start")
                perturbation_end = importance_data.get("question_char_end")

        if not tokens or not offset_mapping or not token_scores:
            logger.warning("必要なデータが不足しています")
            return []

        if perturbation_start is None or perturbation_end is None:
            logger.warning("摂動対象範囲の情報がありません")
            return []

        question_tokens = []

        for i, (token, score) in enumerate(token_scores):
            if i >= len(offset_mapping):
                break

            start, end = offset_mapping[i]

            # 摂動対象範囲内、スコアが0でない、摂動対象外でないトークンのみ
            is_in_range = start >= perturbation_start and end <= perturbation_end
            is_valid_score = abs(score) > 1e-9  # 正負問わず絶対値で判定
            is_perturbable = not self._should_skip_token(token)

            if is_in_range and is_valid_score and is_perturbable:
                question_tokens.append((i, token, abs(score)))  # 絶対値でスコアを保存

        return question_tokens

    def _split_question_and_choices(
        self,
        text: str,
        num_choices: int,
    ) -> tuple[str, list[str]]:
        """摂動後のテキストを質問文と選択肢に分離.

        形式: "{question}\\n(A) choice1\\n(B) choice2\\n..."

        Args:
            text: 摂動後の質問文＋選択肢テキスト
            num_choices: 選択肢の数

        Returns:
            (摂動後の質問文, 摂動後の選択肢リスト)
        """
        import re

        letters = "ABCDEFGHIJ"

        # 最初の選択肢マーカー \n(A) を探して質問文を分離
        first_marker_pattern = r"\n\(A\) "
        match = re.search(first_marker_pattern, text)

        if not match:
            # マーカーが見つからない場合は元のテキストをそのまま返す
            logger.warning("選択肢マーカー (A) が見つかりません")
            return text, []

        question = text[: match.start()]
        choices_text = text[match.end() :]  # (A) 以降のテキスト

        # 各選択肢を正規表現で分割
        # パターン: \n(B) , \n(C) , ... で分割
        choices = []
        remaining = choices_text

        for i in range(num_choices):
            if i == num_choices - 1:
                # 最後の選択肢は残り全部
                choices.append(remaining.strip())
            else:
                # 次のマーカーを探す
                next_letter = letters[i + 1]
                next_pattern = rf"\n\({next_letter}\) "
                next_match = re.search(next_pattern, remaining)

                if next_match:
                    choices.append(remaining[: next_match.start()].strip())
                    remaining = remaining[next_match.end() :]
                else:
                    # 次のマーカーが見つからない場合は残りを使用
                    choices.append(remaining.strip())
                    break

        return question, choices

    def _apply_perturbations(
        self,
        question: str,
        question_tokens: list[tuple[int, str, float]],
        question_char_start: int,
        offset_mapping: list[tuple[int, int]],
        sample_id: str,
    ) -> tuple[str, list[PerturbedToken]]:
        """質問文に摂動を適用.

        Args:
            question: 元の質問文
            question_tokens: 質問文内のトークンとスコア
            question_char_start: 質問文の開始位置（プロンプト全体での位置）
            offset_mapping: オフセットマッピング
            sample_id: サンプルID（再現性のため）

        Returns:
            (摂動後の質問文, 摂動されたトークンのリスト)
        """
        import random

        if not question_tokens:
            return question, []

        # トークン選択方法を決定
        if self.random_perturbation:
            # ランダム選択: スコアの低い（全トークン数-k）件からランダムに選択
            # 重要度順でソート（降順: 高い順）
            sorted_by_importance = sorted(
                question_tokens, key=lambda x: x[2], reverse=True
            )

            # 上位k個を除いた残りのトークン（スコアの低い方）を候補とする
            if len(sorted_by_importance) > self.num_perturbations:
                # トークン数がkより多い場合: 上位k個を除いた残りから選択
                low_importance_tokens = sorted_by_importance[self.num_perturbations :]
            else:
                # トークン数がk以下の場合: 全トークンから選択
                low_importance_tokens = sorted_by_importance
                logger.debug(
                    f"サンプル {sample_id}: トークン数({len(question_tokens)})が"
                    f"k({self.num_perturbations})以下のため、全トークンからランダムに選択"
                )

            # サンプルIDに基づいたシードでシャッフル
            rng = random.Random(hash((self.seed, sample_id, "selection")))
            shuffled_tokens = list(low_importance_tokens)
            rng.shuffle(shuffled_tokens)
            candidate_tokens = shuffled_tokens
        elif self.bottom_k_perturbation:
            # Anti-LRP: スコア昇順でソート（下位k個を選択）
            candidate_tokens = sorted(question_tokens, key=lambda x: x[2], reverse=False)
        else:
            # 重要度ベース: スコア降順でソート（上位k個を選択）
            candidate_tokens = sorted(question_tokens, key=lambda x: x[2], reverse=True)

        perturbed_question = question
        perturbed_tokens = []
        used_token_indices: set[int] = set()

        # 摂動による位置のずれを追跡
        offset_adjustment = 0

        # 指定した回数だけ摂動を適用（または候補トークンがなくなるまで）
        for token_index, token_str, score in candidate_tokens:
            if len(perturbed_tokens) >= self.num_perturbations:
                break

            # 既に使用したトークンはスキップ
            if token_index in used_token_indices:
                continue

            # トークンの位置を取得
            if token_index >= len(offset_mapping):
                continue

            char_start, char_end = offset_mapping[token_index]

            # 質問文内での相対位置を計算
            relative_start = char_start - question_char_start + offset_adjustment
            relative_end = char_end - question_char_start + offset_adjustment

            if relative_start < 0 or relative_end > len(perturbed_question):
                continue

            # 現在のトークン文字列を取得
            current_token = perturbed_question[relative_start:relative_end]

            # トークン文字列に基づいたシードで摂動を適用（再現性確保）
            token_seed = hash((self.seed, sample_id, token_str))
            token_generator = CharacterPerturbationGenerator(seed=token_seed)
            result = token_generator.perturb(current_token)

            if result is None:
                # 摂動が失敗した場合、次の候補トークンを試す
                continue

            # 摂動後の質問文を更新
            perturbed_question = (
                perturbed_question[:relative_start]
                + result.perturbed
                + perturbed_question[relative_end:]
            )

            # 位置のずれを更新
            length_diff = len(result.perturbed) - len(current_token)
            offset_adjustment += length_diff

            perturbed_tokens.append(
                PerturbedToken(
                    token_index=token_index,
                    original_token=token_str,
                    perturbed_token=result.perturbed,
                    importance_score=score,
                    perturbation_type=result.perturbation_type.value,
                    char_position=result.position,
                )
            )

            used_token_indices.add(token_index)

        # 位置順（先頭から）でソートして返す
        perturbed_tokens_sorted = sorted(perturbed_tokens, key=lambda x: x.token_index)

        return perturbed_question, perturbed_tokens_sorted

    def create(self) -> PerturbedDataset:
        """摂動データセットを作成.

        Returns:
            摂動データセット
        """
        samples = []
        skipped_count = 0

        for result in self.results:
            sample_id = result["sample_id"]

            # 重要度スコアを読み込み
            importance_data = self._load_importance_scores(sample_id)
            if importance_data is None:
                skipped_count += 1
                continue

            # 摂動対象のトークンを取得
            perturbation_tokens = self._get_question_tokens(importance_data)

            if not perturbation_tokens:
                logger.warning(f"摂動対象トークンが見つかりません: {sample_id}")
                skipped_count += 1
                continue

            # SQuAD v2の場合はcontextも摂動対象
            context_char_start = importance_data.get("context_char_start")
            original_context = result.get("context")
            original_question = result["question"]
            offset_mapping = importance_data.get("offset_mapping", [])

            if context_char_start is not None and original_context is not None:
                # SQuAD v2: context + questionを結合して摂動
                # 注: プロンプト内ではcontextとquestionの間に改行などがある可能性
                # offset_mappingはプロンプト全体に対するものなので、
                # context_char_startを基準に摂動を適用

                # 摂動を適用（context部分のみ）
                # context内のトークンとquestion内のトークンを分離
                context_char_end = importance_data.get("context_char_end", 0)
                question_char_start = importance_data.get("question_char_start", 0)

                # contextに対応するトークンを取得
                context_tokens = [
                    (i, t, s)
                    for i, t, s in perturbation_tokens
                    if i < len(offset_mapping)
                    and offset_mapping[i][0] >= context_char_start
                    and offset_mapping[i][1] <= context_char_end
                ]
                # questionに対応するトークンを取得
                question_tokens = [
                    (i, t, s)
                    for i, t, s in perturbation_tokens
                    if i < len(offset_mapping)
                    and offset_mapping[i][0] >= question_char_start
                ]

                # contextを摂動
                perturbed_context, context_perturbed_tokens = self._apply_perturbations(
                    question=original_context,
                    question_tokens=context_tokens,
                    question_char_start=context_char_start,
                    offset_mapping=offset_mapping,
                    sample_id=f"{sample_id}_context",
                )

                # questionを摂動
                perturbed_question, question_perturbed_tokens = self._apply_perturbations(
                    question=original_question,
                    question_tokens=question_tokens,
                    question_char_start=question_char_start,
                    offset_mapping=offset_mapping,
                    sample_id=f"{sample_id}_question",
                )

                # 摂動トークンを結合
                perturbed_tokens = context_perturbed_tokens + question_perturbed_tokens
            else:
                # MMLU/MMLU-Pro: question + choices を摂動
                question_char_start = importance_data.get("question_char_start", 0)
                question_char_end = importance_data.get("question_char_end", 0)
                choices = result.get("choices")

                if choices:
                    # 選択肢を含むテキストを構築（プロンプトと同じ形式：空白区切り）
                    letters = "ABCDEFGHIJ"
                    options_str = " ".join(
                        f"({letters[i]}) {choice}" for i, choice in enumerate(choices)
                    )
                    question_with_choices = f"{original_question}\n{options_str}"

                    # question_with_choices_endがない場合は計算（後方互換性）
                    if "question_with_choices_end" not in importance_data:
                        # 質問文の終了位置 + 改行 + 選択肢の長さ
                        calculated_end = question_char_end + 1 + len(options_str)
                        importance_data["question_with_choices_end"] = calculated_end
                        logger.debug(
                            f"question_with_choices_end を計算: {calculated_end} "
                            f"(question_char_end={question_char_end}, options_len={len(options_str)})"
                        )

                        # 摂動対象トークンを再取得（選択肢も含める）
                        perturbation_tokens = self._get_question_tokens(importance_data)

                    # デバッグログ
                    logger.debug(
                        f"question_with_choices length: {len(question_with_choices)}, "
                        f"expected: {importance_data.get('question_with_choices_end', 0) - question_char_start}"
                    )
                    logger.debug(f"perturbation_tokens count: {len(perturbation_tokens)}")

                    # 摂動を適用
                    perturbed_text, perturbed_tokens = self._apply_perturbations(
                        question=question_with_choices,
                        question_tokens=perturbation_tokens,
                        question_char_start=question_char_start,
                        offset_mapping=offset_mapping,
                        sample_id=sample_id,
                    )

                    # 摂動後のテキスト全体を保存（質問文＋選択肢）
                    perturbed_question = perturbed_text
                    perturbed_choices = None  # 分離しない
                else:
                    # 選択肢がない場合（GSM8Kなど）
                    perturbed_question, perturbed_tokens = self._apply_perturbations(
                        question=original_question,
                        question_tokens=perturbation_tokens,
                        question_char_start=question_char_start,
                        offset_mapping=offset_mapping,
                        sample_id=sample_id,
                    )
                    perturbed_choices = None

                perturbed_context = None

            # SQuAD v2の場合はperturbed_choicesはNone
            if context_char_start is not None:
                perturbed_choices = None

            # 摂動サンプルを作成
            sample = PerturbedSample(
                sample_id=sample_id,
                original_question=original_question,
                perturbed_question=perturbed_question,
                perturbed_tokens=perturbed_tokens,
                choices=result.get("choices"),
                correct_answer=result["correct_answer"],
                subset=result.get("subset"),
                context=perturbed_context,
                original_context=original_context if context_char_start is not None else None,
                perturbed_choices=perturbed_choices,
            )

            samples.append(sample)

        logger.info(
            f"摂動データセット作成完了: {len(samples)} サンプル（スキップ: {skipped_count}）"
        )

        # 摂動モードを決定
        if self.random_perturbation:
            perturbation_mode = "random"
        elif self.bottom_k_perturbation:
            perturbation_mode = "bottom_k"
        else:
            perturbation_mode = "importance"

        # メタデータを作成
        metadata = {
            "source_dir": str(self.baseline_dir),
            "source_model": self.config.get("model", "unknown"),
            "benchmark": self.config.get("benchmark", "unknown"),
            "num_perturbations": self.num_perturbations,
            "perturbation_mode": perturbation_mode,
            "include_choices": self.include_choices,
            "seed": self.seed,
            "total_samples": len(samples),
            "skipped_samples": skipped_count,
            "created_at": datetime.now().isoformat(),
            "perturbation_types": ["proximity", "double_typing", "omission"],
        }

        return PerturbedDataset(metadata=metadata, samples=samples)


def create_perturbed_dataset(
    baseline_dir: str | Path,
    num_perturbations: int,
    output_dir: str | Path,
    seed: int = 42,
    random_perturbation: bool = False,
    include_choices: bool = True,
    bottom_k_perturbation: bool = False,
) -> Path:
    """摂動データセットを作成して保存.

    Args:
        baseline_dir: Phase 1の結果ディレクトリ
        num_perturbations: 摂動回数
        output_dir: 出力ディレクトリ
        seed: ランダムシード
        random_perturbation: Trueの場合、重要度スコアを参照せずランダムに摂動
        include_choices: Trueの場合、選択肢も摂動対象に含める
        bottom_k_perturbation: Trueの場合、重要度下位k個のトークンに摂動（Anti-LRP）

    Returns:
        保存されたデータセットのパス
    """
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)

    # 設定を読み込んでディレクトリ名を決定
    config_path = baseline_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        model_name = config.get("model", "unknown").split("/")[-1]
        benchmark = config.get("benchmark", "unknown")
    else:
        # ディレクトリ名から推測
        dir_name = baseline_dir.name
        model_name = dir_name.split("_")[0] if "_" in dir_name else "unknown"
        benchmark = dir_name.split("_")[1] if "_" in dir_name else "unknown"

    # 出力ディレクトリを作成（モードに応じてサフィックスを付加）
    if random_perturbation:
        mode_suffix = "_random"
    elif bottom_k_perturbation:
        mode_suffix = "_bottom_k"
    else:
        mode_suffix = ""
    choices_suffix = "_with_choices" if include_choices else "_question_only"
    dataset_name = f"{model_name}_{benchmark}_k{num_perturbations}{mode_suffix}{choices_suffix}"
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # データセットを作成
    creator = PerturbedDatasetCreator(
        baseline_dir=baseline_dir,
        num_perturbations=num_perturbations,
        seed=seed,
        random_perturbation=random_perturbation,
        include_choices=include_choices,
        bottom_k_perturbation=bottom_k_perturbation,
    )
    dataset = creator.create()

    # 保存
    dataset_path = dataset_dir / "perturbed_dataset.json"
    dataset.save(dataset_path)

    # 設定も保存
    config_output_path = dataset_dir / "config.json"
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"摂動データセットを保存: {dataset_dir}")

    return dataset_path
