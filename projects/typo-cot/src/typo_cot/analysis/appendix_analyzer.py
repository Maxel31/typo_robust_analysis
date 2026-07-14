"""Appendix分析: トークン特性可視化モジュール.

摂動前のimportance_scoresデータを使い、質問トークン・CoTトークンの
特性を可視化する。各分析はC→C / C→I パターン別に実行。

分析A（質問トークン）:
  A1: WordCloud（IDF重み付き）
  A2: 品詞分布（POS比率）
  A3: ポジション散布図（位置 vs スコア）

分析B（CoTトークン）:
  B1: WordCloud（IDF重み付き）
  B2: 品詞分布（POS比率）
  B3: ポジション散布図（位置 vs スコア）
"""

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
from wordcloud import WordCloud

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# 品詞マッピング（spaCy POS → 簡略カテゴリ）
POS_MAPPING: dict[str, str] = {
    "NOUN": "Noun",
    "PROPN": "Noun",
    "VERB": "Verb",
    "AUX": "Verb",
    "ADJ": "Adjective",
    "ADP": "Preposition",
    "ADV": "Adverb",
    "DET": "Determiner",
    "PRON": "Pronoun",
    "NUM": "Numeral",
    "CCONJ": "Conjunction",
    "SCONJ": "Conjunction",
}

# 品詞カテゴリの表示順序
POS_ORDER: list[str] = [
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Preposition",
    "Determiner",
    "Pronoun",
    "Numeral",
    "Conjunction",
    "Other",
]


def merge_subwords(tokens: list[str], scores: list[float]) -> list[tuple[str, float]]:
    """サブワードトークンをマージして単語単位に集約する.

    SentencePiece (▁) と BPE (Ġ) の両方に対応。
    スコアは合計で集約する。

    Args:
        tokens: トークン文字列のリスト
        scores: 各トークンの重要度スコアのリスト

    Returns:
        (マージ済み単語, 合計スコア) のリスト
    """
    if not tokens:
        return []

    merged: list[tuple[str, float]] = []
    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, scores, strict=True):
        # 新しい単語の開始判定
        is_word_start = (
            token.startswith("\u2581")  # SentencePiece: ▁
            or token.startswith("\u0120")  # BPE: Ġ
            or token.startswith(" ")  # スペース付き
        )

        if is_word_start and current_word:
            # 前の単語を保存
            merged.append((current_word, current_score))
            current_word = ""
            current_score = 0.0

        # サブワードマーカーを除去して結合（改行・制御文字も除去）
        clean = token.lstrip("\u2581\u0120 ").replace("\n", "").replace("\r", "")
        current_word += clean
        current_score += score

    # 最後の単語
    if current_word:
        merged.append((current_word, current_score))

    return merged


def assign_pos_tags(words: list[str], nlp: spacy.language.Language) -> list[str]:
    """単語リストに品詞タグを割り当てる.

    spaCyのパイプラインで解析し、文字位置ベースで
    元の単語リストに品詞を対応付ける。

    Args:
        words: 単語文字列のリスト
        nlp: spaCyの言語モデル

    Returns:
        各単語に対応する簡略品詞カテゴリのリスト
    """
    if not words:
        return []

    # 単語を空白結合してspaCyで解析
    text = " ".join(words)
    doc = nlp(text)

    # 文字位置ベースで元の単語に品詞を割り当て
    pos_tags: list[str] = []
    char_offset = 0

    for word in words:
        # この単語の開始位置
        word_start = char_offset
        word_end = word_start + len(word)

        # この位置範囲に重なるspaCyトークンの品詞を取得
        best_pos = "X"
        best_overlap = 0
        for spacy_token in doc:
            # 重なりを計算
            overlap_start = max(word_start, spacy_token.idx)
            overlap_end = min(word_end, spacy_token.idx + len(spacy_token.text))
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = spacy_token.pos_

        pos_tags.append(POS_MAPPING.get(best_pos, "Other"))
        # 次の単語の位置（空白1文字分を加算）
        char_offset = word_end + 1

    return pos_tags


def compute_idf_weights(
    all_word_lists: list[list[str]],
) -> dict[str, float]:
    """全サンプルの単語リストからIDF重みを計算する.

    IDF(w) = log(N / df(w))
    ここでNはサンプル数、df(w)は単語wを含むサンプル数。

    Args:
        all_word_lists: 各サンプルの単語リスト

    Returns:
        単語→IDF重みの辞書
    """
    n_docs = len(all_word_lists)
    if n_docs == 0:
        return {}

    # 各単語のドキュメント頻度を計算
    df: Counter[str] = Counter()
    for word_list in all_word_lists:
        unique_words = {w.lower() for w in word_list}
        df.update(unique_words)

    # IDF計算
    idf: dict[str, float] = {}
    for word, freq in df.items():
        idf[word] = math.log(n_docs / freq)

    return idf


class AppendixAnalyzer:
    """Appendix分析を実行するクラス.

    摂動前のimportance_scoresから質問/CoTトークンを抽出し、
    C→C / C→Iパターン別にWordCloud・品詞分布・ポジション散布図を生成する。
    """

    # 分析対象パターン
    TARGET_PATTERNS: list[str] = ["correct→correct", "correct→incorrect"]
    PATTERN_SHORT: dict[str, str] = {
        "correct→correct": "cc",
        "correct→incorrect": "ci",
    }

    def __init__(
        self,
        analysis_dir: Path,
        output_dir: Path,
        top_k: int = 10,
    ) -> None:
        """初期化.

        Args:
            analysis_dir: 分析結果ディレクトリ
                         (outputs/analysis/{dataset}/{model}/k{N}_{type})
            output_dir: 出力ディレクトリ
            top_k: 上位何トークンを使用するか
        """
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = Path(output_dir)
        self.top_k = top_k

        # spaCyモデルをロード
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        # full_results.jsonからサンプル情報を読み込み
        self.full_results = self._load_full_results()
        # before_dir/after_dirを特定
        self.before_dir, self.after_dir = self._resolve_data_dirs()

    def _load_full_results(self) -> dict:
        """full_results.jsonを読み込む."""
        results_path = self.analysis_dir / "full_results.json"
        if not results_path.exists():
            msg = f"full_results.jsonが見つかりません: {results_path}"
            raise FileNotFoundError(msg)

        with open(results_path, encoding="utf-8") as f:
            return json.load(f)

    def _resolve_data_dirs(self) -> tuple[Path, Path]:
        """full_results.jsonからbefore_dir/after_dirを解決する."""
        metadata = self.full_results.get("metadata", {})
        before_dir = Path(metadata.get("before_dir", ""))
        after_dir = Path(metadata.get("after_dir", ""))

        if not before_dir.exists():
            msg = f"before_dirが見つかりません: {before_dir}"
            raise FileNotFoundError(msg)
        if not after_dir.exists():
            msg = f"after_dirが見つかりません: {after_dir}"
            raise FileNotFoundError(msg)

        return before_dir, after_dir

    def _get_samples_by_pattern(self) -> dict[str, list[dict]]:
        """サンプルをパターン別に分類する.

        Returns:
            パターン名→サンプルリストの辞書
        """
        samples_by_pattern: dict[str, list[dict]] = defaultdict(list)
        for sample in self.full_results.get("sample_results", []):
            pattern = sample.get("pattern", "")
            if pattern in self.TARGET_PATTERNS:
                samples_by_pattern[pattern].append(sample)
        return dict(samples_by_pattern)

    def _load_importance_data(
        self, result_dir: Path, sample_id: str, score_type: str = "question"
    ) -> dict | None:
        """重要度スコアデータを読み込む.

        Args:
            result_dir: 結果ディレクトリ
            sample_id: サンプルID
            score_type: "question" または "cot"

        Returns:
            tokens, scores, 関連メタデータを含む辞書。
            Question用: offset_mapping, question_char_start/end を含む。
            CoT用: cot_token_start/end を含む。
        """
        filename = f"{sample_id}_cot.pt" if score_type == "cot" else f"{sample_id}.pt"
        importance_path = result_dir / "importance_scores" / filename
        if not importance_path.exists():
            return None

        try:
            data = torch.load(importance_path, map_location="cpu", weights_only=False)
            token_scores = data.get("token_scores")
            if token_scores is None:
                return None
            result = {
                "scores": [score for _, score in token_scores],
                "tokens": [token for token, _ in token_scores],
            }
            if score_type == "question":
                # Prompt除外用の境界情報を読み込む
                result["offset_mapping"] = data.get("offset_mapping")
                result["question_char_start"] = data.get("question_char_start")
                result["question_char_end"] = data.get("question_char_end")
            else:
                result["cot_token_start"] = data.get("cot_token_start")
                result["cot_token_end"] = data.get("cot_token_end")
            return result
        except Exception as e:
            logger.warning(f"重要度データ読み込みエラー ({importance_path}): {e}")
            return None

    def _extract_top_k_tokens(
        self, data: dict, score_type: str
    ) -> list[tuple[str, float, int]] | None:
        """重要度データからtop-kトークンを抽出する（Prompt/few-shot部分を除外）.

        Question: offset_mappingとquestion_char_start/endで
                 実際の質問文範囲内のトークンのみを対象とする。
        CoT: cot_token_start/endの範囲内のトークンのみを対象とする。
             境界情報がない場合はスキップ（Noneを返す）。

        Args:
            data: _load_importance_dataの結果
            score_type: "question" または "cot"

        Returns:
            (トークン文字列, スコア, 位置インデックス) のリスト、またはNone
        """
        tokens = data["tokens"]
        scores = data["scores"]

        if score_type == "question":
            # offset_mappingとquestion_char_start/endで質問文範囲を厳密にフィルタ
            offset_mapping = data.get("offset_mapping")
            q_char_start = data.get("question_char_start")
            q_char_end = data.get("question_char_end")

            if offset_mapping is not None and q_char_start is not None and q_char_end is not None:
                # 文字位置ベースで質問文範囲内のトークンのみ選択
                valid_indices = [
                    i
                    for i in range(len(tokens))
                    if offset_mapping[i][0] >= q_char_start
                    and offset_mapping[i][1] <= q_char_end + 5  # 末尾の改行等に余裕
                    and abs(scores[i]) > 1e-8
                ]
            else:
                # フォールバック: 非ゼロスコアのみ（従来方式）
                logger.debug("offset_mapping/question_char境界なし: 非ゼロフィルタに退行")
                valid_indices = [i for i, s in enumerate(scores) if abs(s) > 1e-8]
        else:
            # CoT: cot_token_start/endが必須
            cot_start = data.get("cot_token_start")
            cot_end = data.get("cot_token_end")
            if cot_start is None or cot_end is None:
                logger.warning("cot_token_start/endが未設定: スキップ")
                return None
            valid_indices = list(range(cot_start, min(cot_end, len(tokens))))

        if not valid_indices:
            return None

        # 有効範囲内のトークンとスコアを取得
        valid_pairs = [(tokens[i], scores[i], i) for i in valid_indices]

        # スコア降順でソートしてtop-k
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        return valid_pairs[: self.top_k]

    def _collect_token_data(
        self,
        samples: list[dict],
        score_type: str,
    ) -> tuple[
        list[list[str]],
        list[list[float]],
        list[list[int]],
        list[dict],
    ]:
        """サンプル群からトークンデータを収集する.

        各サンプルのtop-kトークンをサブワードマージし、
        単語・スコア・位置を収集する。
        同時に、サンプルごとのtop-kトークン情報も記録する。

        Args:
            samples: サンプルリスト
            score_type: "question" または "cot"

        Returns:
            (単語リストのリスト, スコアリストのリスト, 位置リストのリスト,
             サンプル別トークン情報のリスト)
        """
        all_words: list[list[str]] = []
        all_scores: list[list[float]] = []
        all_positions: list[list[int]] = []
        sample_token_records: list[dict] = []

        for sample in samples:
            sample_id = sample["sample_id"]
            # 摂動前データを使用
            data = self._load_importance_data(self.before_dir, sample_id, score_type)
            if data is None:
                continue

            top_k_items = self._extract_top_k_tokens(data, score_type)
            if top_k_items is None:
                continue

            # サンプルごとのtop-kトークン情報を記録
            token_details = [
                {
                    "token": token,
                    "score": round(score, 6),
                    "position": pos,
                    "source": score_type,
                }
                for token, score, pos in top_k_items
            ]
            sample_token_records.append(
                {
                    "sample_id": sample_id,
                    "pattern": sample.get("pattern", ""),
                    "tokens": token_details,
                }
            )

            # top-kトークンのみでサブワードマージ
            tk_tokens = [t for t, _, _ in top_k_items]
            tk_scores = [s for _, s, _ in top_k_items]
            tk_positions = [p for _, _, p in top_k_items]

            # サブワードマージ
            merged = merge_subwords(tk_tokens, tk_scores)
            words = [w for w, _ in merged if w.strip()]
            scores = [s for w, s in merged if w.strip()]

            # 位置はマージ前のトークン位置の先頭を使用
            positions: list[int] = []
            for idx, token in enumerate(tk_tokens):
                clean = token.lstrip("\u2581\u0120 ")
                if clean.strip():
                    is_start = (
                        token.startswith("\u2581")
                        or token.startswith("\u0120")
                        or token.startswith(" ")
                    )
                    if is_start or idx == 0:
                        positions.append(tk_positions[idx])
            # 位置リストの長さを単語リストに合わせる
            positions = positions[: len(words)]
            while len(positions) < len(words):
                positions.append(positions[-1] if positions else 0)

            if words:
                all_words.append(words)
                all_scores.append(scores)
                all_positions.append(positions)

        return all_words, all_scores, all_positions, sample_token_records

    def _generate_wordcloud(
        self,
        all_words: list[list[str]],
        all_scores: list[list[float]],
        idf_weights: dict[str, float],
        output_path: Path,
        title: str,
    ) -> dict:
        """IDF重み付きWordCloudを生成する.

        Args:
            all_words: 各サンプルの単語リスト
            all_scores: 各サンプルのスコアリスト
            idf_weights: IDF重み辞書
            output_path: 画像出力パス
            title: 図のタイトル

        Returns:
            上位単語のスコア辞書
        """
        # 単語ごとの重み付きスコアを集計
        word_weighted_scores: Counter[str] = Counter()
        word_counts: Counter[str] = Counter()

        for words, scores in zip(all_words, all_scores, strict=True):
            for word, score in zip(words, scores, strict=True):
                w_lower = word.lower()
                idf = idf_weights.get(w_lower, 1.0)
                word_weighted_scores[w_lower] += score * idf
                word_counts[w_lower] += 1

        if not word_weighted_scores:
            logger.warning(f"WordCloud生成: データなし ({title})")
            return {}

        # 負の値を0にクリップ、改行・制御文字を含む単語を除外
        freq_dict = {
            w: max(0.0, s)
            for w, s in word_weighted_scores.items()
            if s > 0 and w.strip() and "\n" not in w and "\r" not in w
        }

        if not freq_dict:
            logger.warning(f"WordCloud生成: 正のスコアなし ({title})")
            return {}

        # WordCloud生成
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            colormap="viridis",
            prefer_horizontal=0.7,
        )
        wc.generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 上位30単語を返す
        top_words = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:30])
        return top_words

    def _generate_pos_distribution(
        self,
        all_words: list[list[str]],
        all_scores: list[list[float]],
        output_path: Path,
        title: str,
    ) -> dict:
        """品詞分布を計算・可視化する.

        Args:
            all_words: 各サンプルの単語リスト
            all_scores: 各サンプルのスコアリスト
            output_path: 画像出力パス
            title: 図のタイトル

        Returns:
            品詞カテゴリ別の集計結果辞書
        """
        pos_counts: Counter[str] = Counter()
        pos_scores: defaultdict[str, list[float]] = defaultdict(list)

        for words, scores in zip(all_words, all_scores, strict=True):
            pos_tags = assign_pos_tags(words, self.nlp)
            for _word, score, pos in zip(words, scores, pos_tags, strict=True):
                pos_counts[pos] += 1
                pos_scores[pos].append(score)

        if not pos_counts:
            logger.warning(f"品詞分布生成: データなし ({title})")
            return {}

        total = sum(pos_counts.values())

        # 表示用データ準備（順序固定）
        categories = [p for p in POS_ORDER if pos_counts.get(p, 0) > 0]
        counts = [pos_counts[p] for p in categories]
        ratios = [c / total * 100 for c in counts]

        # 棒グラフ生成
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, ratios, color="steelblue", edgecolor="black")
        ax.set_xlabel("POS Category", fontsize=12)
        ax.set_ylabel("Ratio (%)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, max(ratios) * 1.2 if ratios else 100)

        # 各棒の上にパーセンテージを表示
        for bar, ratio in zip(bars, ratios, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{ratio:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 結果辞書
        result = {}
        for pos in categories:
            scores_list = pos_scores[pos]
            result[pos] = {
                "count": pos_counts[pos],
                "ratio": pos_counts[pos] / total,
                "mean_score": float(np.mean(scores_list)) if scores_list else 0.0,
            }
        return result

    def _generate_position_scatter(
        self,
        all_words: list[list[str]],
        all_scores: list[list[float]],
        all_positions: list[list[int]],
        output_path: Path,
        title: str,
    ) -> dict:
        """ポジション散布図（位置 vs スコア）を生成する.

        Args:
            all_words: 各サンプルの単語リスト
            all_scores: 各サンプルのスコアリスト
            all_positions: 各サンプルの位置リスト
            output_path: 画像出力パス
            title: 図のタイトル

        Returns:
            ポジション統計の辞書
        """
        # 全サンプルの位置・スコアをフラット化
        flat_positions: list[int] = []
        flat_scores: list[float] = []
        flat_words: list[str] = []

        for words, scores, positions in zip(all_words, all_scores, all_positions, strict=True):
            for w, s, p in zip(words, scores, positions, strict=True):
                flat_positions.append(p)
                flat_scores.append(s)
                flat_words.append(w)

        if not flat_positions:
            logger.warning(f"ポジション散布図生成: データなし ({title})")
            return {}

        # 散布図生成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            flat_positions,
            flat_scores,
            alpha=0.3,
            s=15,
            c="steelblue",
            edgecolors="none",
        )
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14)

        # トレンドライン（移動平均）
        if len(flat_positions) > 10:
            sorted_pairs = sorted(zip(flat_positions, flat_scores, strict=True))
            sorted_pos = [p for p, _ in sorted_pairs]
            sorted_sc = [s for _, s in sorted_pairs]

            # ビン分割して移動平均
            n_bins = min(20, len(set(sorted_pos)))
            if n_bins > 1:
                bin_edges = np.linspace(min(sorted_pos), max(sorted_pos), n_bins + 1)
                bin_centers = []
                bin_means = []
                for i in range(n_bins):
                    mask = [bin_edges[i] <= p < bin_edges[i + 1] for p in sorted_pos]
                    bin_scores = [s for s, m in zip(sorted_sc, mask, strict=True) if m]
                    if bin_scores:
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_means.append(np.mean(bin_scores))
                if bin_centers:
                    ax.plot(
                        bin_centers,
                        bin_means,
                        color="red",
                        linewidth=2,
                        label="Bin Average",
                    )
                    ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 統計情報
        positions_arr = np.array(flat_positions, dtype=float)
        scores_arr = np.array(flat_scores, dtype=float)

        # 相関係数
        if len(positions_arr) > 2 and np.std(positions_arr) > 0:
            correlation = float(np.corrcoef(positions_arr, scores_arr)[0, 1])
        else:
            correlation = 0.0

        return {
            "n_points": len(flat_positions),
            "position_mean": float(np.mean(positions_arr)),
            "position_std": float(np.std(positions_arr)),
            "score_mean": float(np.mean(scores_arr)),
            "score_std": float(np.std(scores_arr)),
            "position_score_correlation": correlation,
        }

    def analyze(self) -> dict:
        """全分析を実行する.

        Returns:
            分析結果の辞書
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        samples_by_pattern = self._get_samples_by_pattern()
        results: dict = {
            "analysis_dir": str(self.analysis_dir),
            "top_k": self.top_k,
            "patterns": {},
        }

        for pattern in self.TARGET_PATTERNS:
            short = self.PATTERN_SHORT[pattern]
            samples = samples_by_pattern.get(pattern, [])

            if not samples:
                logger.warning(f"パターン '{pattern}' のサンプルがありません")
                results["patterns"][pattern] = {"n_samples": 0}
                continue

            logger.info(f"パターン '{pattern}' ({short}): {len(samples)}サンプル")
            pattern_results: dict = {"n_samples": len(samples)}

            # === 分析A: 質問トークン ===
            logger.info(f"  分析A: 質問トークン解析 ({short})")
            q_words, q_scores, q_positions, q_token_records = self._collect_token_data(
                samples, "question"
            )
            q_idf = compute_idf_weights(q_words)

            if q_words:
                # A1: WordCloud
                a1_path = self.output_dir / f"wordcloud_A_{short}.png"
                a1_result = self._generate_wordcloud(
                    q_words,
                    q_scores,
                    q_idf,
                    a1_path,
                    f"Question Top-{self.top_k} Token WordCloud ({pattern})",
                )
                pattern_results["A1_wordcloud"] = {
                    "top_words": a1_result,
                    "image": str(a1_path.name),
                }

                # A2: 品詞分布
                a2_path = self.output_dir / f"pos_A_{short}.png"
                a2_result = self._generate_pos_distribution(
                    q_words,
                    q_scores,
                    a2_path,
                    f"Question Top-{self.top_k} POS Distribution ({pattern})",
                )
                pattern_results["A2_pos_distribution"] = {
                    "distribution": a2_result,
                    "image": str(a2_path.name),
                }

                # A3: ポジション散布図
                a3_path = self.output_dir / f"position_A_{short}.png"
                a3_result = self._generate_position_scatter(
                    q_words,
                    q_scores,
                    q_positions,
                    a3_path,
                    f"Question Top-{self.top_k} Position vs Score ({pattern})",
                )
                pattern_results["A3_position_scatter"] = {
                    "statistics": a3_result,
                    "image": str(a3_path.name),
                }
            else:
                logger.warning(f"  質問トークンデータなし ({short})")

            # === 分析B: CoTトークン ===
            logger.info(f"  分析B: CoTトークン解析 ({short})")
            c_words, c_scores, c_positions, c_token_records = self._collect_token_data(
                samples, "cot"
            )
            c_idf = compute_idf_weights(c_words)

            if c_words:
                # B1: WordCloud
                b1_path = self.output_dir / f"wordcloud_B_{short}.png"
                b1_result = self._generate_wordcloud(
                    c_words,
                    c_scores,
                    c_idf,
                    b1_path,
                    f"CoT Top-{self.top_k} Token WordCloud ({pattern})",
                )
                pattern_results["B1_wordcloud"] = {
                    "top_words": b1_result,
                    "image": str(b1_path.name),
                }

                # B2: 品詞分布
                b2_path = self.output_dir / f"pos_B_{short}.png"
                b2_result = self._generate_pos_distribution(
                    c_words,
                    c_scores,
                    b2_path,
                    f"CoT Top-{self.top_k} POS Distribution ({pattern})",
                )
                pattern_results["B2_pos_distribution"] = {
                    "distribution": b2_result,
                    "image": str(b2_path.name),
                }

                # B3: ポジション散布図
                b3_path = self.output_dir / f"position_B_{short}.png"
                b3_result = self._generate_position_scatter(
                    c_words,
                    c_scores,
                    c_positions,
                    b3_path,
                    f"CoT Top-{self.top_k} Position vs Score ({pattern})",
                )
                pattern_results["B3_position_scatter"] = {
                    "statistics": b3_result,
                    "image": str(b3_path.name),
                }
            else:
                logger.warning(f"  CoTトークンデータなし ({short})")

            # === サンプルごとのトークン情報を保存 ===
            all_token_records = q_token_records + c_token_records
            if all_token_records:
                # sample_idごとにグループ化
                merged: dict[str, dict] = {}
                for rec in all_token_records:
                    sid = rec["sample_id"]
                    if sid not in merged:
                        merged[sid] = {
                            "sample_id": sid,
                            "pattern": rec["pattern"],
                            "tokens": [],
                        }
                    merged[sid]["tokens"].extend(rec["tokens"])
                # スコア降順でソート
                for entry in merged.values():
                    entry["tokens"].sort(key=lambda t: t["score"], reverse=True)

                sample_tokens_path = self.output_dir / f"sample_tokens_{short}.json"
                with open(sample_tokens_path, "w", encoding="utf-8") as f:
                    json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)
                logger.info(
                    f"  サンプルトークン情報を保存: {sample_tokens_path} ({len(merged)}サンプル)"
                )
                pattern_results["sample_tokens"] = str(sample_tokens_path.name)

            results["patterns"][pattern] = pattern_results

        # 結果をJSONに保存
        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Appendix分析結果を保存: {results_path}")
        return results


def run_appendix_analysis(
    outputs_dir: Path,
    output_base_dir: Path | None = None,
    top_k: int = 10,
) -> None:
    """全設定のAppendix分析を一括実行する.

    outputs/analysis/ 配下の全ディレクトリを走査し、
    各設定（dataset/model/k{N}_{type}）に対して分析を実行する。

    Args:
        outputs_dir: outputsディレクトリのパス
        output_base_dir: 出力ベースディレクトリ（Noneの場合はoutputs/appendix_analysis）
        top_k: 上位何トークンを使用するか
    """
    analysis_base = outputs_dir / "analysis"
    if not analysis_base.exists():
        logger.error(f"分析ディレクトリが見つかりません: {analysis_base}")
        return

    if output_base_dir is None:
        output_base_dir = outputs_dir / "appendix_analysis"

    # analysis配下の全設定ディレクトリを探索
    settings_dirs: list[Path] = []
    for dataset_dir in sorted(analysis_base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for setting_dir in sorted(model_dir.iterdir()):
                if not setting_dir.is_dir():
                    continue
                if (setting_dir / "full_results.json").exists():
                    settings_dirs.append(setting_dir)

    logger.info(f"分析対象設定数: {len(settings_dirs)}")

    for setting_dir in settings_dirs:
        # パスからdataset/model/settingを抽出
        relative = setting_dir.relative_to(analysis_base)
        output_dir = output_base_dir / relative

        logger.info(f"=== Appendix分析: {relative} ===")
        try:
            analyzer = AppendixAnalyzer(
                analysis_dir=setting_dir,
                output_dir=output_dir,
                top_k=top_k,
            )
            analyzer.analyze()
        except Exception as e:
            logger.error(f"分析エラー ({relative}): {e}")
            continue

    logger.info("全設定のAppendix分析が完了しました")
