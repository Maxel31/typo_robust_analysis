"""AttnLRP分析モジュール.

lxtライブラリを使用して、入力トークンの重要度を計算する。
"""

import logging
import re
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class WordScore:
    """単語とスコアのペア.

    Attributes:
        word: 単語（トークン文字列）
        score: 重要度スコア
        token_indices: 単語を構成するトークンのインデックスリスト
        start_pos: 入力テキストにおける開始位置
        end_pos: 入力テキストにおける終了位置
    """

    word: str
    score: float
    token_indices: list[int]
    start_pos: int | None = None
    end_pos: int | None = None


@dataclass
class ImportanceResult:
    """重要度計算結果.

    Attributes:
        input_text: 入力テキスト
        token_scores: 各トークンのスコアリスト（トークン文字列、スコア）- 質問文のみ
        word_scores: 単語ごとの重要度スコアリスト
        top_k_words: 重要単語リスト（質問文のみ、スコア順）
        raw_relevance: 生のrelevanceテンソル
        tokens: トークン文字列のリスト（ヒートマップ用）
        offset_mapping: 各トークンの文字位置リスト（ヒートマップ用）
        token_scores_with_choices: 選択肢を含むトークンスコア（Phase 2用）
        top_k_with_choices: 選択肢を含む重要単語リスト（Phase 2用、スコア順）
    """

    input_text: str
    token_scores: list[tuple[str, float]]
    word_scores: list[WordScore]
    top_k_words: list[WordScore]
    raw_relevance: torch.Tensor
    tokens: list[str] | None = None
    offset_mapping: list[tuple[int, int]] | None = None
    token_scores_with_choices: list[tuple[str, float]] | None = None
    top_k_with_choices: list[WordScore] | None = None


@dataclass
class CombinedImportanceResult:
    """Phase 1.5: 質問文とCoT両方の重要度計算結果.

    Attributes:
        question_importance: 質問文トークンの重要度（CoT最初のトークンに対する）
        cot_importance: CoT推論過程の重要度（最終回答に対する）
        prompt_text: 入力プロンプト
        generated_text: 生成されたテキスト（CoT + 回答）
        full_text: プロンプト + 生成テキスト全体
        prompt_token_count: プロンプトのトークン数
        cot_token_start: CoT開始トークン位置（プロンプト終了位置）
        cot_token_end: CoT終了トークン位置（回答部分の直前）
        answer_token_start: 回答部分の開始トークン位置（ヒートマップ表示用）
        answer_token_end: 回答部分の終了トークン位置（ヒートマップ表示用）
        answer_target_position: 回答選択肢トークンの位置（重要度計算のターゲット）
    """

    question_importance: ImportanceResult
    cot_importance: ImportanceResult
    prompt_text: str
    generated_text: str
    full_text: str
    prompt_token_count: int
    cot_token_start: int
    cot_token_end: int
    answer_token_start: int | None = None
    answer_token_end: int | None = None
    answer_target_position: int | None = None


class AttnLRPAnalyzer:
    """AttnLRPを使用した重要度分析器.

    lxtライブラリを使用して、モデルの出力に対する入力トークンの重要度を計算する。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        top_k: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """初期化.

        Args:
            model: lxtでラップ済みのモデル
            tokenizer: トークナイザー
            top_k: 保存する上位単語数（Noneの場合は全トークン）
            device: 計算に使用するデバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_relevance(
        self,
        input_ids: torch.Tensor,
        target_position: int = -1,
    ) -> torch.Tensor:
        """LRPによるrelevanceスコアを計算.

        lxtのAttnLRPを使用して、入力トークンに対する重要度を計算する。
        https://lxt.readthedocs.io/en/latest/quickstart.html に準拠。

        Note:
            input_embedsは非リーフテンソル（embedding lookupの出力）のため、
            retain_grad()を呼び出さないと.gradが保持されない。
            また、モデルがeval()モードの場合でも勾配計算を行うために
            torch.enable_grad()コンテキストが必要。

        Args:
            input_ids: 入力トークンID (batch_size=1, seq_len)
            target_position: relevanceを計算する対象位置（-1で最後のトークン）

        Returns:
            relevanceテンソル (seq_len,)
        """
        input_ids = input_ids.to(self.device)

        # 既存の勾配をクリア（メモリリーク防止）
        self.model.zero_grad(set_to_none=True)

        # 勾配計算を有効化（eval()モードでも動作するように）
        with torch.enable_grad():
            # Step 1: 入力埋め込みを取得
            input_embeds = self.model.get_input_embeddings()(input_ids)

            # Step 2: 勾配追跡を有効化し、非リーフテンソルの勾配を保持
            input_embeds = input_embeds.requires_grad_(True)
            input_embeds.retain_grad()  # 非リーフテンソルの勾配を保持するために必要

            # Step 3: フォワードパス（use_cache=Falseが重要）
            outputs = self.model(
                inputs_embeds=input_embeds,
                use_cache=False,
            )
            output_logits = outputs.logits

            # Step 4: ターゲット位置の最大logitを取得
            max_logits, _ = torch.max(output_logits[0, target_position, :], dim=-1)

            # Step 5: バックワードパス（AttnLRP）
            max_logits.backward()

        # Step 6: relevanceスコアを計算（勾配と埋め込みの内積）
        if input_embeds.grad is None:
            logger.warning(
                "input_embeds.gradがNoneです。lxtのmonkey_patchが正しく適用されていない可能性があります。"
            )
            # メモリ解放
            del input_embeds, outputs, output_logits, max_logits
            self.model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.zeros(input_ids.shape[1], device=self.device)

        # 勾配計算グラフから切り離してCPUにコピー（メモリリーク防止）
        relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().clone()

        # NaNチェック（デバッグ用）
        if torch.isnan(relevance).any():
            logger.warning(
                f"relevanceにNaNが含まれています。"
                f"grad stats: min={input_embeds.grad.float().min():.4f}, max={input_embeds.grad.float().max():.4f}, "
                f"embed stats: min={input_embeds.float().min():.4f}, max={input_embeds.float().max():.4f}"
            )

        # バッチ次元を削除
        result = relevance.squeeze(0)

        # 中間テンソルを明示的に解放（メモリリーク防止）
        del input_embeds, outputs, output_logits, max_logits, relevance

        # 勾配を完全にクリア
        self.model.zero_grad(set_to_none=True)

        # CUDAキャッシュをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def tokens_to_words(
        self,
        tokens: list[str],
        relevance: torch.Tensor,
    ) -> list[WordScore]:
        """トークンを単語単位に集約.

        サブワードトークンを結合して単語単位のスコアを計算する。

        Args:
            tokens: トークン文字列のリスト
            relevance: 各トークンのrelevanceスコア

        Returns:
            単語ごとのスコアリスト
        """
        word_scores: list[WordScore] = []
        current_word = ""
        current_indices: list[int] = []
        current_score = 0.0

        for i, token in enumerate(tokens):
            # 特殊トークンをスキップ
            if token in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
                continue

            # サブワードの検出（先頭にスペースがない場合は前の単語の続き）
            is_continuation = not token.startswith(" ") and not token.startswith("▁")

            if is_continuation and current_word:
                # 前の単語に結合
                current_word += token.replace("▁", "")
                current_indices.append(i)
                current_score += relevance[i].item()
            else:
                # 前の単語を保存
                if current_word:
                    word_scores.append(
                        WordScore(
                            word=current_word.strip(),
                            score=current_score,
                            token_indices=current_indices,
                        )
                    )

                # 新しい単語を開始
                current_word = token.replace("▁", "").replace(" ", "")
                current_indices = [i]
                current_score = relevance[i].item()

        # 最後の単語を追加
        if current_word:
            word_scores.append(
                WordScore(
                    word=current_word.strip(),
                    score=current_score,
                    token_indices=current_indices,
                )
            )

        return word_scores

    def get_top_k_tokens(
        self,
        tokens: list[str],
        relevance: torch.Tensor,
        k: int | None = None,
    ) -> list[WordScore]:
        """サブワード（トークン）単位でスコアを取得.

        単語への集約を行わず、トークン単位でスコアをランキングする。
        k=Noneの場合は全トークンを返す。

        Args:
            tokens: トークン文字列のリスト
            relevance: 各トークンのrelevanceスコア
            k: 取得する上位件数（Noneの場合は全トークン）

        Returns:
            トークンスコアのリスト（スコア順）
        """
        token_scores: list[WordScore] = []

        for i, token in enumerate(tokens):
            # 特殊トークンをスキップ
            if token in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]:
                continue

            score = relevance[i].item()
            # スコアが0に近いトークンをスキップ
            if abs(score) < 1e-9:
                continue

            # サブワードプレフィックスを除去して表示用の文字列を作成
            display_token = token.replace("▁", " ").replace("Ġ", " ").strip()
            if not display_token:
                display_token = token  # 空になった場合は元のトークンを使用

            token_scores.append(
                WordScore(
                    word=display_token,
                    score=score,
                    token_indices=[i],
                )
            )

        # スコアの絶対値でソート
        sorted_scores = sorted(token_scores, key=lambda x: abs(x.score), reverse=True)

        # kがNoneの場合は全件返す、指定されている場合は上位k件
        if k is not None:
            return sorted_scores[:k]
        return sorted_scores

    def _compute_offset_mapping_fallback(
        self,
        input_text: str,
        input_ids: list[int],
    ) -> list[tuple[int, int]]:
        """offset_mappingを手動で計算するフォールバック.

        一部のトークナイザー（Gemmaなど）はreturn_offsets_mappingをサポートしていないため、
        手動でトークンの文字位置を計算する。

        Args:
            input_text: 入力テキスト
            input_ids: トークンIDのリスト

        Returns:
            各トークンの(開始位置, 終了位置)のリスト
        """
        offset_mapping: list[tuple[int, int]] = []
        current_pos = 0

        for token_id in input_ids:
            token_text = self.tokenizer.decode([token_id])

            # 特殊トークン（空文字列にデコードされる場合）
            if not token_text:
                offset_mapping.append((current_pos, current_pos))
                continue

            # トークンテキストの前後の空白を考慮して検索
            # トークナイザーによっては先頭に空白が含まれる場合がある
            search_text = token_text.lstrip()
            if not search_text:
                # 空白のみのトークン
                offset_mapping.append((current_pos, current_pos + len(token_text)))
                current_pos += len(token_text)
                continue

            # テキスト内でトークンを検索
            found_pos = input_text.find(search_text, current_pos)
            if found_pos != -1:
                # 先頭の空白分を調整
                start_pos = found_pos
                end_pos = found_pos + len(search_text)
                offset_mapping.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # 見つからない場合は現在位置を使用
                offset_mapping.append((current_pos, current_pos))

        return offset_mapping

    def analyze(
        self,
        input_text: str,
        target_position: int = -1,
        question_char_start: int | None = None,
        question_char_end: int | None = None,
    ) -> ImportanceResult:
        """入力テキストの重要度を分析.

        Args:
            input_text: 分析対象のテキスト（完全なプロンプト）
            target_position: relevanceを計算する対象位置
            question_char_start: 質問文の開始位置（文字単位、Noneの場合は全体を分析）
            question_char_end: 質問文の終了位置（文字単位、Noneの場合は全体を分析）

        Returns:
            重要度計算結果
        """
        # トークナイズ
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)

        # offset_mappingを計算（一部のトークナイザーはreturn_offsets_mappingをサポートしていないため、
        # 常にフォールバックメソッドを使用）
        offset_list = self._compute_offset_mapping_fallback(input_text, input_ids[0].tolist())

        # relevanceを計算
        relevance = self.compute_relevance(input_ids, target_position)

        # トークン文字列を取得
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # トークンごとのスコア
        token_scores = [(tokens[i], relevance[i].item()) for i in range(len(tokens))]

        # 質問文範囲が指定されている場合、範囲外のトークンのスコアを0にする
        if question_char_start is not None and question_char_end is not None:
            logger.info(
                f"質問文範囲: {question_char_start} - {question_char_end} (テキスト長: {len(input_text)})"
            )
            # 実際の質問文を表示
            question_text = input_text[question_char_start:question_char_end]
            logger.info(
                f"抽出された質問文: '{question_text[:100]}...' (長さ: {len(question_text)})"
            )

            if offset_list is not None:
                # offset_listを使って各トークンの文字範囲を取得
                offsets = offset_list
                filtered_relevance = relevance.clone()
                in_range_count = 0
                for i, (start, end) in enumerate(offsets):
                    # トークンが質問文範囲外の場合はスコアを0に
                    if end <= question_char_start or start >= question_char_end:
                        filtered_relevance[i] = 0.0
                    else:
                        in_range_count += 1
                # 質問文範囲内のトークンのスコアを確認
                in_range_scores = [
                    filtered_relevance[i].item()
                    for i, (start, end) in enumerate(offsets)
                    if not (end <= question_char_start or start >= question_char_end)
                ]
                if in_range_scores:
                    logger.info(
                        f"質問文範囲内のトークン数: {in_range_count}/{len(offsets)}, "
                        f"スコア範囲: [{min(in_range_scores):.4f}, {max(in_range_scores):.4f}], "
                        f"スコア合計: {sum(in_range_scores):.4f}"
                    )
                else:
                    logger.info(f"質問文範囲内のトークン数: {in_range_count}/{len(offsets)}")

                if in_range_count == 0:
                    # デバッグ: offsetの範囲を表示
                    max_offset = max(e for s, e in offsets)
                    logger.warning(
                        f"質問文範囲内のトークンが0個です。"
                        f"offset最大値: {max_offset}, 質問文開始: {question_char_start}"
                    )
                relevance_for_words = filtered_relevance
            else:
                # offset_mappingが取得できない場合は全体を使用
                relevance_for_words = relevance
                logger.warning("offset_mappingが取得できないため、全体を分析対象とします")
        else:
            relevance_for_words = relevance

        # 単語単位に集約（フィルタリング済みのrelevanceを使用）
        word_scores = self.tokens_to_words(tokens, relevance_for_words)

        # スコアが0でない単語のみを抽出してソート
        non_zero_words = [w for w in word_scores if abs(w.score) > 1e-9]
        sorted_words = sorted(non_zero_words, key=lambda x: abs(x.score), reverse=True)
        top_k_words = sorted_words[: self.top_k]

        # 分析対象の単語数をログに出力
        if question_char_start is not None:
            logger.info(
                f"分析完了: {len(tokens)}トークン → {len(word_scores)}単語 "
                f"(質問文範囲内: {len(non_zero_words)}単語)"
            )
        else:
            logger.info(f"分析完了: {len(tokens)}トークン → {len(word_scores)}単語")
        logger.info(f"重要単語（上位10件）: {[w.word for w in top_k_words[:10]]}")

        return ImportanceResult(
            input_text=input_text,
            token_scores=token_scores,
            word_scores=word_scores,
            top_k_words=top_k_words,
            raw_relevance=relevance,
            tokens=tokens,
            offset_mapping=offset_list,
        )

    def analyze_generation(
        self,
        prompt: str,
        generated_text: str,
    ) -> list[ImportanceResult]:
        """生成テキストの各ステップにおける重要度を分析.

        Args:
            prompt: 入力プロンプト
            generated_text: 生成されたテキスト

        Returns:
            各生成ステップでの重要度結果のリスト
        """
        results: list[ImportanceResult] = []

        # プロンプトのみの分析
        prompt_result = self.analyze(prompt)
        results.append(prompt_result)

        # 生成テキストを追加しながら分析
        full_text = prompt
        generated_tokens = self.tokenizer.encode(generated_text)

        for token_id in generated_tokens[:10]:  # 最初の10トークンのみ
            token_text = self.tokenizer.decode([token_id])
            full_text += token_text

            result = self.analyze(full_text, target_position=-1)
            results.append(result)

        return results

    def _find_answer_pattern(
        self,
        generated_text: str,
        tokens: list[str],
        prompt_token_count: int,
        offset_list: list[tuple[int, int]],
        prompt_length: int,
    ) -> tuple[int | None, int | None, int | None]:
        """生成テキスト内の回答パターン（"The answer is (X)"など）を検出.

        回答パターンを検出し、回答選択肢トークンの位置を返す。

        Args:
            generated_text: 生成されたテキスト
            tokens: 全体のトークンリスト
            prompt_token_count: プロンプトのトークン数
            offset_list: 各トークンの文字位置リスト
            prompt_length: プロンプトの文字数

        Returns:
            (answer_token_start, answer_token_end, answer_choice_position) のタプル
            answer_token_start: 回答パターン開始トークン位置
            answer_token_end: 回答パターン終了トークン位置
            answer_choice_position: 回答選択肢トークンの位置（重要度計算のターゲット）
        """
        # 回答パターンの正規表現（優先度順）
        # 選択肢パターン: "The answer is (A)", "The answer is A", "Answer: A" など
        answer_patterns = [
            # 標準的なパターン
            (r"[Tt]he\s+answer\s+is[:\s]*\(([A-Ja-j])\)", "choice"),
            (r"[Tt]he\s+answer\s+is[:\s]*([A-Ja-j])(?:\.|,|\s|$)", "choice"),
            (r"[Aa]nswer[:\s]+\(([A-Ja-j])\)", "choice"),
            (r"[Aa]nswer[:\s]+([A-Ja-j])(?:\.|,|\s|$)", "choice"),
            (r"\*\*\(([A-Ja-j])\)\*\*", "choice"),  # **(A)** 形式
            (r"\*\*([A-Ja-j])\*\*", "choice"),  # **A** 形式
            (r"(?:correct|right)\s+(?:answer|option)\s+is[:\s]*\(?([A-Ja-j])\)?", "choice"),
            # 数値パターン（GSM8Kなど）
            (r"[Tt]he\s+answer\s+is[:\s]*(-?[\d,]+(?:\.\d+)?)", "number"),
            (r"[Tt]he\s+answer\s+is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)", "number"),
            (r"####\s*(-?[\d,]+(?:\.\d+)?)", "number"),  # GSM8K形式
            (r"[Aa]nswer[:\s]+(-?[\d,]+(?:\.\d+)?)", "number"),
            # 最終フォールバック: 文末の選択肢
            (r"(?:^|\n)\s*\(?([A-Ja-j])\)?\s*\.?\s*$", "choice"),
        ]

        answer_match = None
        answer_choice = None
        match_start_in_gen = None
        pattern_type = None

        for pattern, ptype in answer_patterns:
            matches = list(re.finditer(pattern, generated_text))
            if matches:
                # 最後のマッチを使用（通常は最終回答が最後にある）
                answer_match = matches[-1]
                answer_choice = answer_match.group(1)
                if ptype == "choice":
                    answer_choice = answer_choice.upper()
                match_start_in_gen = answer_match.start()
                pattern_type = ptype
                logger.info(
                    f"回答パターン検出: '{answer_match.group()}' "
                    f"(タイプ: {ptype}, 値: {answer_choice})"
                )
                break

        if answer_match is None:
            logger.warning("回答パターンが検出できませんでした。最終トークンをターゲットとします。")
            return None, None, None

        # 回答パターンの文字位置（full_text内）
        answer_char_start = prompt_length + match_start_in_gen
        answer_char_end = prompt_length + answer_match.end()

        # 回答選択肢/数値の文字位置（full_text内）
        choice_start_in_gen = answer_match.start(1)
        choice_char_start = prompt_length + choice_start_in_gen
        choice_char_end = prompt_length + answer_match.end(1)

        # トークン位置を特定
        answer_token_start = None
        answer_token_end = None
        answer_choice_position = None

        for i, (start, end) in enumerate(offset_list):
            if i < prompt_token_count:
                continue  # プロンプト部分はスキップ

            # 回答パターン開始位置
            if answer_token_start is None and end > answer_char_start:
                answer_token_start = i

            # 回答選択肢/数値の位置
            # トークンが回答部分と重複する場合に特定
            if (
                answer_choice_position is None
                and start < choice_char_end
                and end > choice_char_start
            ):
                answer_choice_position = i
                logger.info(
                    f"回答トークン: position={i}, token='{tokens[i]}', type={pattern_type}"
                )

            # 回答パターン終了位置
            if start < answer_char_end:
                answer_token_end = i

        if answer_token_start is not None and answer_token_end is not None:
            logger.info(f"回答部分トークン範囲: [{answer_token_start}, {answer_token_end}]")

        return answer_token_start, answer_token_end, answer_choice_position

    def analyze_combined(
        self,
        prompt: str,
        generated_text: str,
        question_char_start: int | None = None,
        question_char_end: int | None = None,
        question_with_choices_end: int | None = None,
    ) -> CombinedImportanceResult:
        """Phase 1.5: 質問文とCoT推論過程の両方の重要度を計算.

        1回の生成結果に対して2回のRelevance計算を行う:
        1. 質問文（+選択肢）→CoT最初のトークン（question_importance）
        2. CoT推論部分→回答選択肢トークン（cot_importance）

        質問文の重要度計算とヒートマップには選択肢も含まれるが、
        上位k件のスコア保存は質問文のみ（選択肢を除外）を対象とする。
        これは摂動の対象が質問文のみであるため。

        CoT重要度計算では、回答部分（"The answer is (A)"など）は計算から除外されるが、
        ヒートマップには表示される（重要度0として）。

        Args:
            prompt: 入力プロンプト（Few-shot例示 + 質問文 + 選択肢）
            generated_text: 生成されたテキスト（CoT推論 + 回答）
            question_char_start: 質問文の開始位置（文字単位）
            question_char_end: 質問文のみの終了位置（文字単位、選択肢を含まない）
            question_with_choices_end: 質問文+選択肢の終了位置（文字単位、ヒートマップ用）

        Returns:
            質問文とCoT両方の重要度結果
        """
        # 完全なテキストを結合
        full_text = prompt + generated_text
        prompt_length = len(prompt)

        # トークナイズ
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_token_count = prompt_inputs["input_ids"].shape[1]

        full_inputs = self.tokenizer(full_text, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(self.device)
        total_token_count = full_input_ids.shape[1]

        # offset_mappingを計算
        offset_list = self._compute_offset_mapping_fallback(full_text, full_input_ids[0].tolist())

        # トークン文字列を取得
        tokens = [self.tokenizer.decode([tid]) for tid in full_input_ids[0].tolist()]

        # 回答パターンを検出
        answer_token_start, answer_token_end, answer_choice_position = self._find_answer_pattern(
            generated_text=generated_text,
            tokens=tokens,
            prompt_token_count=prompt_token_count,
            offset_list=offset_list,
            prompt_length=prompt_length,
        )

        # CoTのトークン範囲を特定
        cot_token_start = prompt_token_count

        # 回答パターンが検出された場合、CoT範囲は回答部分の直前まで
        if answer_token_start is not None:
            cot_token_end = answer_token_start - 1
        else:
            # 回答パターンが見つからない場合は最後のトークンの1つ前まで
            cot_token_end = total_token_count - 2

        # 重要度計算のターゲット位置
        if answer_choice_position is not None:
            target_position = answer_choice_position
        else:
            # 回答選択肢が見つからない場合は最後のトークン
            target_position = -1
            logger.warning(
                "回答選択肢トークンが特定できませんでした。最終トークンをターゲットとします。"
            )

        logger.info(
            f"Combined分析: プロンプト={prompt_token_count}トークン, "
            f"生成={total_token_count - prompt_token_count}トークン, "
            f"CoT範囲=[{cot_token_start}, {cot_token_end}], "
            f"ターゲット位置={target_position}"
        )

        # ============================================
        # 1. 質問文（+選択肢）→ CoT最初のトークン の重要度計算
        # ============================================
        logger.info("質問文→CoT重要度を計算中...")
        question_relevance = self.compute_relevance(
            full_input_ids,
            target_position=cot_token_start,  # CoT最初のトークン
        )

        # ヒートマップ用: 質問文+選択肢の範囲でフィルタリング
        heatmap_char_end = question_with_choices_end or question_char_end
        if question_char_start is not None and heatmap_char_end is not None:
            question_filtered_relevance = question_relevance.clone()
            for i, (start, end) in enumerate(offset_list):
                if end <= question_char_start or start >= heatmap_char_end:
                    question_filtered_relevance[i] = 0.0
        else:
            question_filtered_relevance = question_relevance

        # 上位k件用: 質問文のみ（選択肢を除外）の範囲でフィルタリング
        # 摂動の対象は質問文のみであるため
        if question_char_start is not None and question_char_end is not None:
            question_only_filtered_relevance = question_relevance.clone()
            for i, (start, end) in enumerate(offset_list):
                if end <= question_char_start or start >= question_char_end:
                    question_only_filtered_relevance[i] = 0.0
        else:
            question_only_filtered_relevance = question_filtered_relevance

        # 質問文のみのトークンスコアを計算（上位k件用、サブワード単位）
        question_top_k = self.get_top_k_tokens(tokens, question_only_filtered_relevance)

        # 質問文+選択肢のトークンスコアを計算（Phase 2用）
        question_with_choices_top_k = self.get_top_k_tokens(
            tokens, question_filtered_relevance
        )

        # 質問文+選択肢の単語スコア（互換性のため残す）
        question_word_scores = self.tokens_to_words(tokens, question_filtered_relevance)

        # 質問文のみのトークンスコア（フィルタリング済み）
        question_only_token_scores = [
            (tokens[i], question_only_filtered_relevance[i].item())
            for i in range(len(tokens))
        ]

        # 質問文+選択肢のトークンスコア（フィルタリング済み）
        question_with_choices_token_scores = [
            (tokens[i], question_filtered_relevance[i].item())
            for i in range(len(tokens))
        ]

        question_importance = ImportanceResult(
            input_text=full_text,
            token_scores=question_only_token_scores,  # 質問文のみ
            word_scores=question_word_scores,
            top_k_words=question_top_k,  # 質問文のみから抽出した上位k件
            raw_relevance=question_filtered_relevance,  # ヒートマップ用（選択肢も含む）
            tokens=tokens,
            offset_mapping=offset_list,
            token_scores_with_choices=question_with_choices_token_scores,  # 選択肢含む
            top_k_with_choices=question_with_choices_top_k,  # 選択肢含む上位k件
        )

        logger.info(
            f"質問文重要度（選択肢除外）: {len(question_top_k)}トークン, 上位10件 = {[w.word for w in question_top_k[:10]]}"
        )
        logger.info(
            f"質問文重要度（選択肢含む）: {len(question_with_choices_top_k)}トークン, 上位10件 = {[w.word for w in question_with_choices_top_k[:10]]}"
        )

        # ============================================
        # 2. CoT推論部分 → 回答選択肢 の重要度計算
        # ============================================
        logger.info("CoT→回答重要度を計算中...")
        cot_relevance = self.compute_relevance(
            full_input_ids,
            target_position=target_position,  # 回答選択肢トークン
        )

        # CoT範囲でフィルタリング
        # - プロンプト部分を0に
        # - 回答部分（"The answer is..."）も0に
        cot_filtered_relevance = cot_relevance.clone()

        # プロンプト部分を除外
        for i in range(prompt_token_count):
            cot_filtered_relevance[i] = 0.0

        # 回答部分を除外（ヒートマップには表示するが重要度計算からは除外）
        if answer_token_start is not None:
            for i in range(answer_token_start, total_token_count):
                cot_filtered_relevance[i] = 0.0

        # CoTのトークンスコアを計算（サブワード単位）
        cot_top_k = self.get_top_k_tokens(tokens, cot_filtered_relevance)

        # 互換性のため単語スコアも計算
        cot_word_scores = self.tokens_to_words(tokens, cot_filtered_relevance)

        cot_token_scores = [(tokens[i], cot_relevance[i].item()) for i in range(len(tokens))]

        cot_importance = ImportanceResult(
            input_text=full_text,
            token_scores=cot_token_scores,
            word_scores=cot_word_scores,
            top_k_words=cot_top_k,
            raw_relevance=cot_filtered_relevance,  # 回答部分が0のフィルタリング済みrelevance
            tokens=tokens,
            offset_mapping=offset_list,
        )

        logger.info(f"CoT重要度: {len(cot_top_k)}トークン, 上位10件 = {[w.word for w in cot_top_k[:10]]}")

        return CombinedImportanceResult(
            question_importance=question_importance,
            cot_importance=cot_importance,
            prompt_text=prompt,
            generated_text=generated_text,
            full_text=full_text,
            prompt_token_count=prompt_token_count,
            cot_token_start=cot_token_start,
            cot_token_end=cot_token_end,
            answer_token_start=answer_token_start,
            answer_token_end=answer_token_end if answer_token_end else total_token_count - 1,
            answer_target_position=answer_choice_position,
        )

    def analyze_squad(
        self,
        prompt: str,
        generated_text: str,
        context_char_start: int | None = None,
        context_char_end: int | None = None,
        question_char_start: int | None = None,
        question_char_end: int | None = None,
    ) -> CombinedImportanceResult:
        """SQuAD用: 全生成トークンをターゲットとした重要度計算.

        SQuADでは回答パターンが存在しないため、生成されたすべてのトークンを
        ターゲットとして重要度を計算し、それらを合算する。

        Args:
            prompt: 入力プロンプト（コンテキスト + 質問文）
            generated_text: 生成されたテキスト（回答）
            context_char_start: コンテキストの開始位置（文字単位）
            context_char_end: コンテキストの終了位置（文字単位）
            question_char_start: 質問文の開始位置（文字単位）
            question_char_end: 質問文の終了位置（文字単位）

        Returns:
            コンテキスト+質問文の重要度結果
        """
        # 完全なテキストを結合
        full_text = prompt + generated_text

        # トークナイズ
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_token_count = prompt_inputs["input_ids"].shape[1]

        full_inputs = self.tokenizer(full_text, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(self.device)
        total_token_count = full_input_ids.shape[1]

        # offset_mappingを計算
        offset_list = self._compute_offset_mapping_fallback(full_text, full_input_ids[0].tolist())

        # トークン文字列を取得
        tokens = [self.tokenizer.decode([tid]) for tid in full_input_ids[0].tolist()]

        # 生成されたトークン数
        generated_token_count = total_token_count - prompt_token_count

        logger.info(
            f"SQuAD分析: プロンプト={prompt_token_count}トークン, "
            f"生成={generated_token_count}トークン"
        )

        if generated_token_count == 0:
            logger.warning("生成されたトークンがありません")
            # 空の結果を返す
            empty_relevance = torch.zeros(total_token_count, device=self.device)
            token_scores = [(tokens[i], 0.0) for i in range(len(tokens))]
            empty_result = ImportanceResult(
                input_text=full_text,
                token_scores=token_scores,
                word_scores=[],
                top_k_words=[],
                raw_relevance=empty_relevance,
                tokens=tokens,
                offset_mapping=offset_list,
            )
            return CombinedImportanceResult(
                question_importance=empty_result,
                cot_importance=empty_result,
                prompt_text=prompt,
                generated_text=generated_text,
                full_text=full_text,
                prompt_token_count=prompt_token_count,
                cot_token_start=prompt_token_count,
                cot_token_end=total_token_count - 1,
            )

        # ============================================
        # 全生成トークンに対する重要度を合算
        # ============================================
        logger.info(f"全生成トークン({generated_token_count}個)に対する重要度を計算中...")

        # 各生成トークンをターゲットとしてrelevanceを計算し、合算
        aggregated_relevance = torch.zeros(total_token_count, device=self.device)

        for target_idx in range(prompt_token_count, total_token_count):
            relevance = self.compute_relevance(full_input_ids, target_position=target_idx)
            aggregated_relevance += relevance

            # メモリ解放（ループ内でのメモリリーク防止）
            del relevance

            # 進捗ログ（10トークンごと）
            progress = target_idx - prompt_token_count + 1
            if progress % 10 == 0:
                logger.debug(f"重要度計算進捗: {progress}/{generated_token_count}")
                # 定期的にメモリをクリア
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 平均化（トークン数で割る）
        aggregated_relevance = aggregated_relevance / generated_token_count

        logger.info(
            f"重要度合算完了: min={aggregated_relevance.min():.4f}, "
            f"max={aggregated_relevance.max():.4f}, "
            f"sum={aggregated_relevance.sum():.4f}"
        )

        # ============================================
        # コンテキスト+質問文範囲でフィルタリング
        # ============================================
        # 分析対象範囲を決定（context開始からquestion終了まで）
        if context_char_start is not None and question_char_end is not None:
            analysis_start = context_char_start
            analysis_end = question_char_end
        elif question_char_start is not None and question_char_end is not None:
            # contextが指定されていない場合は質問文のみ
            analysis_start = question_char_start
            analysis_end = question_char_end
        else:
            # 範囲が指定されていない場合はプロンプト全体を使用
            analysis_start = 0
            analysis_end = len(prompt)

        # コンテキスト+質問文の範囲でフィルタリング
        input_filtered_relevance = aggregated_relevance.clone()
        for i, (start, end) in enumerate(offset_list):
            if end <= analysis_start or start >= analysis_end:
                input_filtered_relevance[i] = 0.0

        # 上位k件用のトークンスコアを計算（サブワード単位）
        input_top_k = self.get_top_k_tokens(tokens, input_filtered_relevance)

        # 単語スコア（互換性のため）
        input_word_scores = self.tokens_to_words(tokens, input_filtered_relevance)

        input_token_scores = [
            (tokens[i], aggregated_relevance[i].item()) for i in range(len(tokens))
        ]

        question_importance = ImportanceResult(
            input_text=full_text,
            token_scores=input_token_scores,
            word_scores=input_word_scores,
            top_k_words=input_top_k,
            raw_relevance=input_filtered_relevance,
            tokens=tokens,
            offset_mapping=offset_list,
        )

        logger.info(
            f"SQuAD入力重要度（context+question）: {len(input_top_k)}トークン, 上位10件 = {[w.word for w in input_top_k[:10]]}"
        )

        # ============================================
        # 回答（生成テキスト）部分のヒートマップ用
        # ============================================
        # SQuADではCoT推論がないため、生成テキスト全体を「回答」として扱う
        answer_filtered_relevance = aggregated_relevance.clone()
        # プロンプト部分は除外
        for i in range(prompt_token_count):
            answer_filtered_relevance[i] = 0.0

        answer_top_k = self.get_top_k_tokens(tokens, answer_filtered_relevance)
        answer_word_scores = self.tokens_to_words(tokens, answer_filtered_relevance)
        answer_token_scores = [
            (tokens[i], aggregated_relevance[i].item()) for i in range(len(tokens))
        ]

        answer_importance = ImportanceResult(
            input_text=full_text,
            token_scores=answer_token_scores,
            word_scores=answer_word_scores,
            top_k_words=answer_top_k,
            raw_relevance=answer_filtered_relevance,
            tokens=tokens,
            offset_mapping=offset_list,
        )

        logger.info(
            f"SQuAD回答重要度: {len(answer_top_k)}トークン, 上位10件 = {[w.word for w in answer_top_k[:10]]}"
        )

        return CombinedImportanceResult(
            question_importance=question_importance,
            cot_importance=answer_importance,  # SQuADでは回答部分をcot_importanceに格納
            prompt_text=prompt,
            generated_text=generated_text,
            full_text=full_text,
            prompt_token_count=prompt_token_count,
            cot_token_start=prompt_token_count,  # 回答開始位置
            cot_token_end=total_token_count - 1,  # 回答終了位置
            answer_token_start=prompt_token_count,
            answer_token_end=total_token_count - 1,
            answer_target_position=None,  # SQuADでは単一ターゲットがない
        )


def create_analyzer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> AttnLRPAnalyzer:
    """AttnLRP分析器を作成するファクトリ関数.

    Args:
        model: lxtでラップ済みのモデル
        tokenizer: トークナイザー
        top_k: 保存する上位単語数（Noneの場合は全トークン）
        device: 計算に使用するデバイス

    Returns:
        AttnLRPAnalyzerインスタンス
    """
    return AttnLRPAnalyzer(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        device=device,
    )
