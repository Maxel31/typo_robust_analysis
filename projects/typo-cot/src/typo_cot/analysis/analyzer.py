"""Phase 4: 摂動前後の分析を行うモジュール.

摂動前と摂動後の結果を比較し、
モデルの注意分布やCoT推論過程の変化を分析する。
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..perturbation.dataset import PerturbedDataset, PerturbedToken
from .metrics import (
    cohens_d,
    js_divergence,
    mann_whitney_u_test,
    pearson_correlation,
    rouge_l_score,
    shannon_entropy,
    spearman_correlation,
    top_k_concentration,
    top_k_jaccard,
    top_k_jaccard_by_token,
)

logger = logging.getLogger(__name__)

# 固定トークン数ベースのk値
Q_TOP_COUNTS = [3, 5, 10]  # 質問文用
COT_TOP_COUNTS = [3, 5, 10, 15, 20]  # CoT用（より長いシーケンス対応）
# 後方互換性のため残す
K_TOP_COUNTS = Q_TOP_COUNTS


@dataclass
class MetricStats:
    """メトリクスの統計情報（平均と標準偏差）."""

    mean: float
    std: float
    n: int

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {"mean": self.mean, "std": self.std, "n": self.n}


@dataclass
class SamplePairResult:
    """サンプルペア（摂動前後）の分析結果."""

    sample_id: str

    # 正解情報
    before_correct: bool
    after_correct: bool
    pattern: str  # "correct→correct", "correct→incorrect", etc.
    answer_changed: bool

    # 回答テキスト
    before_answer: str
    after_answer: str

    # Question→CoT のメトリクス
    question_entropy_before: float
    question_entropy_after: float
    question_delta_entropy: float
    question_js_divergence: float
    # 質問文全トークン対象のSpearman相関係数（アライメント使用）
    question_spearman_r: float
    # Top-k Concentration (固定トークン数ベース)
    question_concentration_before: dict[str, float]  # {"top3": 0.x, "top5": 0.x, "top10": 0.x}
    question_concentration_after: dict[str, float]
    question_delta_concentration: dict[str, float]
    # Top-k Jaccard (固定トークン数ベース)
    question_jaccard: dict[str, float]  # {"top3": 0.x, "top5": 0.x, "top10": 0.x}

    # CoT→Answer のメトリクス
    cot_entropy_before: float
    cot_entropy_after: float
    cot_delta_entropy: float
    # Top-k Concentration (固定トークン数ベース)
    cot_concentration_before: dict[str, float]
    cot_concentration_after: dict[str, float]
    cot_delta_concentration: dict[str, float]
    # Top-k Jaccard (固定トークン数ベース)
    cot_jaccard: dict[
        str, float
    ]  # {"top3": ..., "top5": ..., "top10": ..., "top15": ..., "top20": ...}
    # ROUGE-L
    cot_rouge_l: dict[str, float]

    # トークン数変化（質問文）- デフォルト値あり
    question_token_count_before: int = 0
    question_token_count_after: int = 0
    question_token_count_diff: int = 0  # after - before（負の値も取りうる）

    # 生成テキスト - デフォルト値あり
    before_generated_text: str = ""
    after_generated_text: str = ""


@dataclass
class StatisticalTestResult:
    """統計的検定の結果."""

    metric_name: str
    group1_name: str
    group2_name: str
    group1_n: int
    group2_n: int
    group1_mean: float
    group1_std: float
    group2_mean: float
    group2_std: float
    delta: float
    mann_whitney_u: float
    mann_whitney_p: float
    cohens_d: float
    significance: str  # "", "*", "**", "***"


@dataclass
class CorrelationResult:
    """相関分析の結果."""

    variable1: str
    variable2: str
    group_name: str  # "all", "correct→incorrect", etc.
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    interpretation: str  # 相関の強さの解釈


@dataclass
class PartialCorrelationResult:
    """偏相関分析の結果.

    ROUGE-L（文章類似度）の影響を統制した上で、
    Jaccard係数（注目トークン集合の類似度）と回答の正誤の関係を分析する。
    """

    variable: str  # "cot_jaccard_top10" など
    control_variable: str  # "cot_rouge_l_f1"（統制変数）
    target_variable: str  # "answer_correctness"（0=C→I, 1=C→C）
    n: int
    partial_r: float  # 偏相関係数
    partial_p: float  # p値
    zero_order_r: float  # 統制前の単純相関係数（参考値）
    zero_order_p: float  # 統制前のp値
    interpretation: str  # 結果の解釈


@dataclass
class AnalysisResult:
    """分析全体の結果."""

    # メタデータ
    before_dir: str
    after_dir: str
    total_samples: int

    # 実験メタデータ（config.jsonから抽出）
    dataset: str = ""
    model: str = ""
    num_perturbations: int = 0
    perturbation_type: str = ""  # "importance", "random", or "bottom_k"

    # パターン別件数
    pattern_counts: dict[str, int] = field(default_factory=dict)
    answer_changed_count: int = 0
    answer_unchanged_count: int = 0

    # typo前・後ともに回答スパンが検出できなかったため集計対象から除外したサンプル数
    excluded_no_answer_count: int = 0

    # パターン別メトリクス（平均値と標準偏差）
    pattern_metrics: dict[str, dict[str, dict]] = field(default_factory=dict)

    # answer_changed/unchanged別メトリクス（平均値と標準偏差）
    answer_change_metrics: dict[str, dict[str, dict]] = field(default_factory=dict)

    # 全体メトリクス（平均値と標準偏差）
    overall_metrics: dict[str, dict] = field(default_factory=dict)

    # 統計的検定結果
    statistical_tests: list[StatisticalTestResult] = field(default_factory=list)

    # 相関分析結果
    correlation_results: list[CorrelationResult] = field(default_factory=list)

    # 偏相関分析結果
    partial_correlation_results: list[PartialCorrelationResult] = field(default_factory=list)

    # 具体例サンプル（定性分析用）
    # 8パターン: (C→C / C→I) × (ROUGE-L 高/低) × (CoT-Jaccard@10 高/低)
    # 各パターン5事例、合計40事例（重複なし）
    # 各サンプルは {sample_id, pattern, cot_jaccard_10, rouge_l, q_jaccard_10} の辞書
    example_samples: dict[str, list[dict]] = field(
        default_factory=lambda: {
            "cc_rouge_high_jaccard_high": [],
            "cc_rouge_high_jaccard_low": [],
            "cc_rouge_low_jaccard_high": [],
            "cc_rouge_low_jaccard_low": [],
            "ci_rouge_high_jaccard_high": [],
            "ci_rouge_high_jaccard_low": [],
            "ci_rouge_low_jaccard_high": [],
            "ci_rouge_low_jaccard_low": [],
        }
    )

    # CoT寄与度分析結果（分析1: CoT vs 質問文の寄与スコア比較）
    cot_relevance_analysis: dict = field(default_factory=dict)

    # CoTトークン分類結果（分析2: 共通/CoT固有トークンの割合）
    cot_token_classification: dict = field(default_factory=dict)

    # 層別Jaccard相関分析結果（分析3: 共通/CoT固有のJaccardと回答変化の相関）
    cot_stratified_jaccard_analysis: dict = field(default_factory=dict)

    # 個別サンプル結果
    sample_results: list[SamplePairResult] = field(default_factory=list)


def normalize_token(token: str) -> str:
    """トークンを正規化する（サブワードマーカー除去 + 小文字化）.

    SentencePiece (▁)、BPE (Ġ) などのサブワードマーカーを除去し、
    小文字化して比較可能な形式にする。

    Args:
        token: 正規化するトークン文字列

    Returns:
        正規化されたトークン文字列
    """
    # サブワードマーカーを除去: ▁ (U+2581, SentencePiece), Ġ (GPT-2 BPE)
    normalized = token.lstrip("\u2581\u0120 ")
    return normalized.lower()


def compute_stats(values: list[float]) -> MetricStats:
    """値のリストから統計情報を計算する."""
    if not values:
        return MetricStats(mean=0.0, std=0.0, n=0)

    arr = np.array(values, dtype=np.float64)
    return MetricStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        n=len(arr),
    )


def get_significance(p_value: float) -> str:
    """p値から有意性マーカーを取得する."""
    if math.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def interpret_correlation(r: float) -> str:
    """相関係数の強さを解釈する."""
    if math.isnan(r):
        return "N/A"
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


class PerturbationAnalyzer:
    """摂動前後の分析を行うクラス."""

    PATTERNS = [
        "correct→correct",
        "correct→incorrect",
        "incorrect→correct",
        "incorrect→incorrect",
    ]

    def __init__(
        self,
        before_dir: Path,
        after_dir: Path,
    ) -> None:
        """初期化.

        Args:
            before_dir: 摂動前の結果ディレクトリ
            after_dir: 摂動後の結果ディレクトリ
        """
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)

        # 結果ファイルを読み込み
        self.before_results = self._load_results(self.before_dir)
        self.after_results = self._load_results(self.after_dir)

        # メタデータを抽出
        self.metadata = self._extract_metadata()

        # tokenizerを遅延ロード用にNoneで初期化
        self._tokenizer = None
        self._model_name = self._get_model_name()

        logger.info(f"摂動前結果: {len(self.before_results)} サンプル")
        logger.info(f"摂動後結果: {len(self.after_results)} サンプル")

    def _load_results(self, result_dir: Path) -> dict[str, dict]:
        """結果ファイルを読み込む.

        Args:
            result_dir: 結果ディレクトリ

        Returns:
            sample_id -> 結果データの辞書
        """
        results_path = result_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"results.json が見つかりません: {results_path}")

        with open(results_path, encoding="utf-8") as f:
            results_list = json.load(f)

        # sample_idをキーにした辞書に変換
        return {r["sample_id"]: r for r in results_list}

    def _extract_metadata(self) -> dict[str, str | int]:
        """config.jsonからメタデータを抽出する.

        Returns:
            dataset, model, num_perturbations, perturbation_type を含む辞書
        """
        metadata = {
            "dataset": "",
            "model": "",
            "num_perturbations": 0,
            "perturbation_type": "",
        }

        # Phase 3（摂動後）のconfig.jsonを読み込み
        after_config_path = self.after_dir / "config.json"
        if after_config_path.exists():
            with open(after_config_path, encoding="utf-8") as f:
                config = json.load(f)

            # トップレベルの benchmark と model を優先（実際に実行されたベンチマーク）
            metadata["dataset"] = config.get("benchmark", "")
            metadata["model"] = config.get("model", "").split("/")[-1]

            # perturbed_metadata から摂動パラメータを取得
            if "perturbed_metadata" in config:
                pm = config["perturbed_metadata"]
                metadata["num_perturbations"] = pm.get("num_perturbations", 0)
                metadata["perturbation_type"] = pm.get("perturbation_mode", "")

        # Phase 1（摂動前）のconfig.jsonを補完
        before_config_path = self.before_dir / "config.json"
        if before_config_path.exists() and not metadata["model"]:
            with open(before_config_path, encoding="utf-8") as f:
                config = json.load(f)
            metadata["dataset"] = config.get("benchmark", "")
            metadata["model"] = config.get("model", "").split("/")[-1]

        logger.info(
            f"メタデータ: dataset={metadata['dataset']}, model={metadata['model']}, "
            f"k={metadata['num_perturbations']}, type={metadata['perturbation_type']}"
        )
        return metadata

    def _get_model_name(self) -> str:
        """config.jsonからフルモデル名を取得する.

        Returns:
            モデル名（例: "google/gemma-3-4b-it"）
        """
        # Phase 1（摂動前）のconfig.jsonを優先（Phase 3はperturbed_metadataから取得）
        for config_dir in [self.before_dir, self.after_dir]:
            config_path = config_dir / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                model_name = config.get("model", "")
                if model_name:
                    return model_name
        return ""

    @property
    def tokenizer(self):
        """tokenizerを遅延ロードして返す."""
        if self._tokenizer is None and self._model_name:
            try:
                from transformers import AutoTokenizer

                logger.info(f"Tokenizerをロード中: {self._model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            except Exception as e:
                logger.warning(f"Tokenizerのロードに失敗: {e}")
                self._tokenizer = None
        return self._tokenizer

    def _count_question_tokens_from_results(self, sample_id: str, is_before: bool = True) -> int:
        """results.jsonの質問文をtokenizerでトークン化してカウント.

        Args:
            sample_id: サンプルID
            is_before: Trueなら摂動前、Falseなら摂動後

        Returns:
            質問文（+選択肢）のトークン数
        """
        results = self.before_results if is_before else self.after_results
        result = results.get(sample_id)
        if result is None:
            return 0

        # 質問文を取得
        question = result.get("question", "")
        choices = result.get("choices")

        # 選択肢がリストで存在する場合は結合（ベースラインの場合）
        if choices and isinstance(choices, list):
            choices_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            full_question = f"{question}\n{choices_text}"
        else:
            # 摂動後は既にquestionに選択肢が含まれている
            full_question = question

        # tokenizerでトークン化
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(full_question, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"トークン化エラー ({sample_id}): {e}")
                return 0

        # tokenizerがない場合は文字数で概算（1トークン≒4文字）
        return len(full_question) // 4

    def _load_importance_data(
        self, result_dir: Path, sample_id: str, score_type: str = "question"
    ) -> dict | None:
        """重要度スコアのフルデータを読み込む.

        Args:
            result_dir: 結果ディレクトリ
            sample_id: サンプルID
            score_type: "question" または "cot"

        Returns:
            token_scores, tokens, offset_mapping, question_char_start/end を含む辞書、またはNone
            score_type="cot"の場合はcot_token_start/endも含む
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
                "offset_mapping": data.get("offset_mapping"),
                "question_char_start": data.get("question_char_start"),
                "question_char_end": data.get("question_char_end"),
                "question_with_choices_end": data.get("question_with_choices_end"),
            }
            # CoTデータの場合はCoT境界情報も読み込む
            if score_type == "cot":
                result["cot_token_start"] = data.get("cot_token_start")
                result["cot_token_end"] = data.get("cot_token_end")
            return result
        except Exception as e:
            logger.warning(f"重要度データ読み込みエラー ({importance_path}): {e}")
            return None

    def _get_question_token_range(self, result_dir: Path, sample_id: str) -> tuple[int, int] | None:
        """質問文のトークンインデックス範囲を取得する.

        Question .ptのoffset_mappingとquestion_char_start/endから
        実際の質問文のトークン範囲を特定する。few-shot例を除外するために使用。

        Args:
            result_dir: 結果ディレクトリ
            sample_id: サンプルID

        Returns:
            (開始インデックス, 終了インデックス) のタプル、または取得できない場合はNone
        """
        q_data = self._load_importance_data(result_dir, sample_id, "question")
        if q_data is None:
            return None

        offset_mapping = q_data.get("offset_mapping")
        q_char_start = q_data.get("question_char_start")
        q_char_end = q_data.get("question_char_end")

        if offset_mapping is None or q_char_start is None or q_char_end is None:
            return None

        # 質問文範囲内のトークンインデックスを収集
        q_indices = [
            i
            for i in range(len(offset_mapping))
            if offset_mapping[i][0] >= q_char_start and offset_mapping[i][1] <= q_char_end + 5
        ]
        if not q_indices:
            return None

        return (min(q_indices), max(q_indices) + 1)

    def _load_perturbed_dataset(self) -> PerturbedDataset | None:
        """摂動データセットを読み込む.

        Returns:
            摂動データセット、またはNone
        """
        if hasattr(self, "_perturbed_dataset"):
            return self._perturbed_dataset

        # config.jsonからperturbed_dataset_pathを取得
        after_config_path = self.after_dir / "config.json"
        if not after_config_path.exists():
            logger.warning("config.jsonが見つかりません")
            self._perturbed_dataset = None
            return None

        with open(after_config_path, encoding="utf-8") as f:
            config = json.load(f)

        dataset_path = config.get("perturbed_dataset_path")
        if not dataset_path:
            logger.warning("perturbed_dataset_pathが設定されていません")
            self._perturbed_dataset = None
            return None

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.warning(f"摂動データセットが見つかりません: {dataset_path}")
            self._perturbed_dataset = None
            return None

        try:
            self._perturbed_dataset = PerturbedDataset.load(dataset_path)
            logger.info(
                f"摂動データセットを読み込み: {len(self._perturbed_dataset.samples)} サンプル"
            )
            return self._perturbed_dataset
        except Exception as e:
            logger.warning(f"摂動データセット読み込みエラー: {e}")
            self._perturbed_dataset = None
            return None

    def _create_token_alignment(
        self,
        original_offset_mapping: list[tuple[int, int]],
        perturbed_offset_mapping: list[tuple[int, int]],
        perturbed_tokens: list[PerturbedToken],
    ) -> dict[int, list[int]]:
        """摂動後トークンから元トークンへの堅牢なアライメントを作成.

        PerturbedToken.token_indexの情報を使用し、文字位置の累積オフセットを
        正確に追跡してマッピングを行う。

        Args:
            original_offset_mapping: 元のトークンのoffset_mapping
            perturbed_offset_mapping: 摂動後のトークンのoffset_mapping
            perturbed_tokens: 摂動されたトークンのリスト

        Returns:
            元トークンインデックス -> 摂動後トークンインデックスのリスト
        """
        # 摂動情報をtoken_indexでマップ
        perturbed_info: dict[int, PerturbedToken] = {pt.token_index: pt for pt in perturbed_tokens}
        perturbed_indices = set(perturbed_info.keys())

        # 摂動による文字位置変化を記録（元の開始位置でソート）
        # (original_start, original_end, perturbed_length, original_length)
        perturbation_events: list[tuple[int, int, int, int]] = []
        for pt in perturbed_tokens:
            if pt.token_index < len(original_offset_mapping):
                orig_start, orig_end = original_offset_mapping[pt.token_index]
                perturbation_events.append(
                    (
                        orig_start,
                        orig_end,
                        len(pt.perturbed_token),
                        len(pt.original_token),
                    )
                )
        perturbation_events.sort(key=lambda x: x[0])

        def map_char_position(orig_pos: int) -> int:
            """元のテキストの文字位置を摂動後テキストの位置にマッピング."""
            offset = 0
            for orig_start, orig_end, pert_len, orig_len in perturbation_events:
                if orig_pos >= orig_end:
                    # この摂動は完全に前にある → オフセットを加算
                    offset += pert_len - orig_len
                elif orig_pos > orig_start:
                    # 摂動範囲内にいる → 比例マッピング
                    # 摂動範囲の開始位置 + オフセット
                    ratio = (orig_pos - orig_start) / orig_len if orig_len > 0 else 0
                    return orig_start + offset + int(ratio * pert_len)
            return orig_pos + offset

        alignment: dict[int, list[int]] = {}

        for orig_idx, (orig_start, orig_end) in enumerate(original_offset_mapping):
            if orig_start == orig_end:  # 空トークンをスキップ
                continue

            # 期待される摂動後テキストでの文字範囲を計算
            if orig_idx in perturbed_indices:
                # 摂動されたトークン: 摂動後の文字長を使用
                pt = perturbed_info[orig_idx]
                expected_start = map_char_position(orig_start)
                expected_end = expected_start + len(pt.perturbed_token)
            else:
                # 摂動されていないトークン: 元の長さを維持
                expected_start = map_char_position(orig_start)
                expected_end = map_char_position(orig_end)

            # 期待範囲と重複する摂動後トークンを全て取得
            matching: list[int] = []
            for pert_idx, (pert_start, pert_end) in enumerate(perturbed_offset_mapping):
                if pert_start == pert_end:
                    continue
                # 重複判定: 範囲が重なっているか
                if pert_start < expected_end and pert_end > expected_start:
                    matching.append(pert_idx)

            if matching:
                alignment[orig_idx] = matching
            else:
                # マッチが見つからない場合のフォールバック
                # 同じインデックスを使用（但し範囲内の場合のみ）
                if orig_idx < len(perturbed_offset_mapping):
                    alignment[orig_idx] = [orig_idx]

        return alignment

    def _align_perturbed_scores(
        self,
        original_scores: list[float],
        perturbed_scores: list[float],
        alignment: dict[int, list[int]],
    ) -> list[float]:
        """摂動後スコアを元のトークン空間にアライメント.

        Args:
            original_scores: 元のスコアリスト
            perturbed_scores: 摂動後のスコアリスト
            alignment: 元トークンインデックス -> 摂動後トークンインデックスのリスト

        Returns:
            元のトークン空間にアライメントされたスコアリスト
        """
        aligned_scores = [0.0] * len(original_scores)

        for orig_idx in range(len(original_scores)):
            if orig_idx in alignment:
                pert_indices = alignment[orig_idx]
                # 複数の摂動後トークンがある場合は合計（sum）
                total_score = sum(
                    perturbed_scores[i] for i in pert_indices if i < len(perturbed_scores)
                )
                aligned_scores[orig_idx] = total_score
            else:
                # アライメントがない場合は元のスコアを使用（通常は発生しない）
                if orig_idx < len(perturbed_scores):
                    aligned_scores[orig_idx] = perturbed_scores[orig_idx]

        return aligned_scores

    def _compute_jaccard_with_alignment(
        self,
        original_scores: list[float],
        perturbed_scores: list[float],
        alignment: dict[int, list[int]] | None,
    ) -> dict[str, float]:
        """アライメントを考慮したTop-k Jaccardを計算（固定トークン数ベース）.

        Args:
            original_scores: 元のスコアリスト
            perturbed_scores: 摂動後のスコアリスト
            alignment: トークンアライメント（Noneの場合は従来の計算）

        Returns:
            各固定トークン数でのJaccard係数 {"top3": 0.x, "top5": 0.x, "top10": 0.x}
        """
        if not original_scores or not perturbed_scores:
            return {f"top{k}": 0.0 for k in K_TOP_COUNTS}

        # アライメントがある場合は摂動後スコアをアライメント
        if alignment:
            aligned_perturbed = self._align_perturbed_scores(
                original_scores, perturbed_scores, alignment
            )
        else:
            aligned_perturbed = perturbed_scores

        result = {}
        for k in K_TOP_COUNTS:
            key = f"top{k}"
            result[key] = top_k_jaccard(original_scores, aligned_perturbed, k=k)
        return result

    def _compute_spearman_with_alignment(
        self,
        original_scores: list[float],
        perturbed_scores: list[float],
        alignment: dict[int, list[int]] | None,
    ) -> float:
        """アライメントを考慮した質問文全トークン対象のSpearman相関係数を計算.

        アライメントにより摂動前後のトークン位置集合が一致することが保証される。

        Args:
            original_scores: 元のスコアリスト
            perturbed_scores: 摂動後のスコアリスト
            alignment: トークンアライメント（Noneの場合は従来の計算）

        Returns:
            Spearman相関係数（-1〜1）
        """
        from scipy import stats

        if not original_scores or not perturbed_scores:
            return 0.0

        # アライメントがある場合は摂動後スコアをアライメント
        if alignment:
            aligned_perturbed = self._align_perturbed_scores(
                original_scores, perturbed_scores, alignment
            )
        else:
            aligned_perturbed = perturbed_scores

        # 長さを揃える（短い方に合わせる）
        min_len = min(len(original_scores), len(aligned_perturbed))
        if min_len < 3:  # Spearman相関には最低3点必要
            return 0.0

        orig = original_scores[:min_len]
        pert = aligned_perturbed[:min_len]

        # 分散がない場合は計算不可
        if np.std(orig) < 1e-9 or np.std(pert) < 1e-9:
            return 0.0

        rho, _ = stats.spearmanr(orig, pert)
        return float(rho) if not np.isnan(rho) else 0.0

    def _compute_js_divergence_with_alignment(
        self,
        original_scores: list[float],
        perturbed_scores: list[float],
        alignment: dict[int, list[int]] | None,
    ) -> float:
        """アライメントを考慮したJS-Divergenceを計算.

        Args:
            original_scores: 元のスコアリスト
            perturbed_scores: 摂動後のスコアリスト
            alignment: トークンアライメント（Noneの場合は従来の計算）

        Returns:
            JS-Divergence値
        """
        if not original_scores or not perturbed_scores:
            return 0.0

        # アライメントがある場合は摂動後スコアをアライメント
        if alignment:
            aligned_perturbed = self._align_perturbed_scores(
                original_scores, perturbed_scores, alignment
            )
        else:
            aligned_perturbed = perturbed_scores

        return js_divergence(original_scores, aligned_perturbed)

    def _get_pattern(self, before_correct: bool, after_correct: bool) -> str:
        """正解パターンを取得する."""
        before_str = "correct" if before_correct else "incorrect"
        after_str = "correct" if after_correct else "incorrect"
        return f"{before_str}→{after_str}"

    def _compute_concentration_metrics(self, scores: list[float] | None) -> dict[str, float]:
        """固定トークン数ベースのTop-k Concentrationを計算する."""
        if not scores:
            return {f"top{k}": 0.0 for k in K_TOP_COUNTS}

        result = {}
        for k in K_TOP_COUNTS:
            key = f"top{k}"
            result[key] = top_k_concentration(scores, k=k)
        return result

    def _compute_jaccard_metrics_by_token(
        self,
        tokens1: list[str] | None,
        scores1: list[float] | None,
        tokens2: list[str] | None,
        scores2: list[float] | None,
    ) -> dict[str, float]:
        """トークンベースのTop-k Jaccardを計算する（CoT用）.

        同じトークンが複数回出現する場合は、最も高いスコアのトークンのみを残して
        重複を排除してから計算を行う。

        Args:
            tokens1: 分布1のトークンリスト
            scores1: 分布1のスコア
            tokens2: 分布2のトークンリスト
            scores2: 分布2のスコア

        Returns:
            {"top3": ..., "top5": ..., "top10": ..., "top15": ..., "top20": ...} の形式の辞書
        """
        if not tokens1 or not scores1 or not tokens2 or not scores2:
            return {f"top{k}": 0.0 for k in COT_TOP_COUNTS}

        result = {}
        for k in COT_TOP_COUNTS:
            key = f"top{k}"
            result[key] = top_k_jaccard_by_token(tokens1, scores1, tokens2, scores2, k=k)
        return result

    def _compute_delta_metrics(
        self,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Δメトリクス（after - before）を計算する."""
        return {
            key: after_metrics.get(key, 0.0) - before_metrics.get(key, 0.0)
            for key in before_metrics
        }

    def _analyze_sample_pair(
        self,
        sample_id: str,
        before_data: dict,
        after_data: dict,
    ) -> SamplePairResult | None:
        """サンプルペアを分析する.

        Args:
            sample_id: サンプルID
            before_data: 摂動前の結果
            after_data: 摂動後の結果

        Returns:
            分析結果、またはNone（データ不足の場合）
        """
        # 正解情報
        before_correct = before_data.get("is_correct", False)
        after_correct = after_data.get("is_correct", False)
        pattern = self._get_pattern(before_correct, after_correct)

        # 回答テキスト
        before_answer = before_data.get("extracted_answer", "")
        after_answer = after_data.get("extracted_answer", "")
        answer_changed = before_answer != after_answer

        # 生成テキスト
        before_generated = before_data.get("generated_text", "")
        after_generated = after_data.get("generated_text", "")

        # 質問文Relevanceスコアを読み込み（フルデータ）
        q_data_before = self._load_importance_data(self.before_dir, sample_id, "question")
        q_data_after = self._load_importance_data(self.after_dir, sample_id, "question")

        q_scores_before = q_data_before["scores"] if q_data_before else None
        q_scores_after = q_data_after["scores"] if q_data_after else None

        # 質問文（+選択肢）のトークン数を計算（results.jsonからtokenizerで計算）
        q_token_count_before = self._count_question_tokens_from_results(sample_id, is_before=True)
        q_token_count_after = self._count_question_tokens_from_results(sample_id, is_before=False)
        q_token_count_diff = q_token_count_after - q_token_count_before

        # CoT Relevanceスコアを読み込み（トークンベースJaccard用にフルデータを取得）
        cot_data_before = self._load_importance_data(self.before_dir, sample_id, "cot")
        cot_data_after = self._load_importance_data(self.after_dir, sample_id, "cot")
        cot_scores_before = cot_data_before["scores"] if cot_data_before else None
        cot_scores_after = cot_data_after["scores"] if cot_data_after else None
        cot_tokens_before = cot_data_before["tokens"] if cot_data_before else None
        cot_tokens_after = cot_data_after["tokens"] if cot_data_after else None

        # トークンアライメントを作成（Q-JaccardとQ-JS-Divergence用）
        q_alignment = None
        perturbed_dataset = self._load_perturbed_dataset()
        if (
            perturbed_dataset
            and q_data_before
            and q_data_after
            and q_data_before.get("offset_mapping")
            and q_data_after.get("offset_mapping")
        ):
            # 該当サンプルの摂動トークン情報を取得
            perturbed_sample = next(
                (s for s in perturbed_dataset.samples if s.sample_id == sample_id), None
            )
            if perturbed_sample and perturbed_sample.perturbed_tokens:
                q_alignment = self._create_token_alignment(
                    q_data_before["offset_mapping"],
                    q_data_after["offset_mapping"],
                    perturbed_sample.perturbed_tokens,
                )

        # ========== Question→CoT メトリクス ==========
        # Entropy（自然対数で正規化）
        if q_scores_before and q_scores_after:
            q_entropy_before = shannon_entropy(q_scores_before, normalize=True)
            q_entropy_after = shannon_entropy(q_scores_after, normalize=True)
            q_delta_entropy = q_entropy_after - q_entropy_before
            # JS-Divergence（アライメント考慮）
            q_js_div = self._compute_js_divergence_with_alignment(
                q_scores_before, q_scores_after, q_alignment
            )
        else:
            logger.warning(f"質問文スコアが不足: {sample_id}")
            q_entropy_before = 0.0
            q_entropy_after = 0.0
            q_delta_entropy = 0.0
            q_js_div = 0.0

        # Top-k Concentration
        q_conc_before = self._compute_concentration_metrics(q_scores_before)
        q_conc_after = self._compute_concentration_metrics(q_scores_after)
        q_delta_conc = self._compute_delta_metrics(q_conc_before, q_conc_after)

        # Top-k Jaccard（アライメント考慮）
        q_jaccard = self._compute_jaccard_with_alignment(
            q_scores_before, q_scores_after, q_alignment
        )

        # Spearman相関係数（全トークン対象、アライメント考慮）
        q_spearman_r = self._compute_spearman_with_alignment(
            q_scores_before, q_scores_after, q_alignment
        )

        # ========== CoT→Answer メトリクス ==========
        # Entropy（参考値）
        if cot_scores_before and cot_scores_after:
            cot_entropy_before = shannon_entropy(cot_scores_before, normalize=True)
            cot_entropy_after = shannon_entropy(cot_scores_after, normalize=True)
            cot_delta_entropy = cot_entropy_after - cot_entropy_before
        else:
            cot_entropy_before = 0.0
            cot_entropy_after = 0.0
            cot_delta_entropy = 0.0

        # Top-k Concentration
        cot_conc_before = self._compute_concentration_metrics(cot_scores_before)
        cot_conc_after = self._compute_concentration_metrics(cot_scores_after)
        cot_delta_conc = self._compute_delta_metrics(cot_conc_before, cot_conc_after)

        # Top-k Jaccard（トークンベース: 同一トークン集合の比較、重複排除済み）
        cot_jaccard = self._compute_jaccard_metrics_by_token(
            cot_tokens_before, cot_scores_before, cot_tokens_after, cot_scores_after
        )

        # ROUGE-L
        cot_rouge = rouge_l_score(before_generated, after_generated)

        return SamplePairResult(
            sample_id=sample_id,
            before_correct=before_correct,
            after_correct=after_correct,
            pattern=pattern,
            answer_changed=answer_changed,
            before_answer=before_answer,
            after_answer=after_answer,
            # トークン数変化
            question_token_count_before=q_token_count_before,
            question_token_count_after=q_token_count_after,
            question_token_count_diff=q_token_count_diff,
            # Question metrics
            question_entropy_before=q_entropy_before,
            question_entropy_after=q_entropy_after,
            question_delta_entropy=q_delta_entropy,
            question_js_divergence=q_js_div,
            question_spearman_r=q_spearman_r,
            question_concentration_before=q_conc_before,
            question_concentration_after=q_conc_after,
            question_delta_concentration=q_delta_conc,
            question_jaccard=q_jaccard,
            # CoT metrics
            cot_entropy_before=cot_entropy_before,
            cot_entropy_after=cot_entropy_after,
            cot_delta_entropy=cot_delta_entropy,
            cot_concentration_before=cot_conc_before,
            cot_concentration_after=cot_conc_after,
            cot_delta_concentration=cot_delta_conc,
            cot_jaccard=cot_jaccard,
            cot_rouge_l=cot_rouge,
            # Generated text
            before_generated_text=before_generated,
            after_generated_text=after_generated,
        )

    def _compute_group_metrics(self, samples: list[SamplePairResult]) -> dict[str, dict]:
        """グループのメトリクス統計（平均と標準偏差）を計算する."""
        if not samples:
            return {}

        # メトリクス収集用辞書
        metrics_values: dict[str, list[float]] = {}

        # スカラーメトリクス
        scalar_keys = [
            "question_entropy_before",
            "question_entropy_after",
            "question_delta_entropy",
            "question_js_divergence",
            "question_spearman_r",
            "question_token_count_diff",
            "cot_entropy_before",
            "cot_entropy_after",
            "cot_delta_entropy",
            "cot_rouge_l_f1",
        ]
        for key in scalar_keys:
            metrics_values[key] = []

        # 固定トークン数ベースメトリクス（質問文用）
        q_top_k_keys = [f"top{k}" for k in Q_TOP_COUNTS]
        for prefix in [
            "question_concentration_before",
            "question_concentration_after",
            "question_delta_concentration",
            "question_jaccard",
        ]:
            for k in q_top_k_keys:
                metrics_values[f"{prefix}_{k}"] = []

        # 固定トークン数ベースメトリクス（CoT用）
        cot_top_k_keys = [f"top{k}" for k in COT_TOP_COUNTS]
        for prefix in [
            "cot_concentration_before",
            "cot_concentration_after",
            "cot_delta_concentration",
            "cot_jaccard",
        ]:
            for k in cot_top_k_keys:
                metrics_values[f"{prefix}_{k}"] = []

        # 値を収集
        for sr in samples:
            metrics_values["question_entropy_before"].append(sr.question_entropy_before)
            metrics_values["question_entropy_after"].append(sr.question_entropy_after)
            metrics_values["question_delta_entropy"].append(sr.question_delta_entropy)
            metrics_values["question_js_divergence"].append(sr.question_js_divergence)
            metrics_values["question_spearman_r"].append(sr.question_spearman_r)
            metrics_values["question_token_count_diff"].append(float(sr.question_token_count_diff))
            metrics_values["cot_entropy_before"].append(sr.cot_entropy_before)
            metrics_values["cot_entropy_after"].append(sr.cot_entropy_after)
            metrics_values["cot_delta_entropy"].append(sr.cot_delta_entropy)
            metrics_values["cot_rouge_l_f1"].append(sr.cot_rouge_l["f1"])

            # 固定トークン数ベースメトリクス（質問文用）
            for k in q_top_k_keys:
                metrics_values[f"question_concentration_before_{k}"].append(
                    sr.question_concentration_before.get(k, 0.0)
                )
                metrics_values[f"question_concentration_after_{k}"].append(
                    sr.question_concentration_after.get(k, 0.0)
                )
                metrics_values[f"question_delta_concentration_{k}"].append(
                    sr.question_delta_concentration.get(k, 0.0)
                )
                metrics_values[f"question_jaccard_{k}"].append(sr.question_jaccard.get(k, 0.0))

            # 固定トークン数ベースメトリクス（CoT用）
            for k in cot_top_k_keys:
                metrics_values[f"cot_concentration_before_{k}"].append(
                    sr.cot_concentration_before.get(k, 0.0)
                )
                metrics_values[f"cot_concentration_after_{k}"].append(
                    sr.cot_concentration_after.get(k, 0.0)
                )
                metrics_values[f"cot_delta_concentration_{k}"].append(
                    sr.cot_delta_concentration.get(k, 0.0)
                )
                metrics_values[f"cot_jaccard_{k}"].append(sr.cot_jaccard.get(k, 0.0))

        # 統計を計算
        return {
            metric: compute_stats(values).to_dict() for metric, values in metrics_values.items()
        }

    def _run_statistical_tests(self, result: AnalysisResult) -> list[StatisticalTestResult]:
        """統計的検定を実行する."""
        tests = []

        # グループ分け
        correct_to_correct = [sr for sr in result.sample_results if sr.pattern == "correct→correct"]
        correct_to_incorrect = [
            sr for sr in result.sample_results if sr.pattern == "correct→incorrect"
        ]
        answer_changed = [sr for sr in result.sample_results if sr.answer_changed]
        answer_unchanged = [sr for sr in result.sample_results if not sr.answer_changed]

        # 検定対象メトリクス
        metrics_to_test = [
            ("question_entropy_before", lambda sr: sr.question_entropy_before),
            ("question_delta_entropy", lambda sr: sr.question_delta_entropy),
            ("question_js_divergence", lambda sr: sr.question_js_divergence),
            ("cot_delta_entropy", lambda sr: sr.cot_delta_entropy),
            ("cot_rouge_l_f1", lambda sr: sr.cot_rouge_l["f1"]),
        ]

        # 固定トークン数ベースのメトリクスを追加
        for k in K_TOP_COUNTS:
            key = f"top{k}"
            metrics_to_test.extend(
                [
                    (
                        f"question_concentration_before_{key}",
                        lambda sr, k=key: sr.question_concentration_before.get(k, 0.0),
                    ),
                    (
                        f"question_delta_concentration_{key}",
                        lambda sr, k=key: sr.question_delta_concentration.get(k, 0.0),
                    ),
                    (
                        f"question_jaccard_{key}",
                        lambda sr, k=key: sr.question_jaccard.get(k, 0.0),
                    ),
                    (
                        f"cot_concentration_before_{key}",
                        lambda sr, k=key: sr.cot_concentration_before.get(k, 0.0),
                    ),
                    (
                        f"cot_delta_concentration_{key}",
                        lambda sr, k=key: sr.cot_delta_concentration.get(k, 0.0),
                    ),
                    (
                        f"cot_jaccard_{key}",
                        lambda sr, k=key: sr.cot_jaccard.get(k, 0.0),
                    ),
                ]
            )

        # (A) correct→incorrect vs correct→correct
        if len(correct_to_incorrect) >= 2 and len(correct_to_correct) >= 2:
            for metric_name, getter in metrics_to_test:
                group1 = [getter(sr) for sr in correct_to_incorrect]
                group2 = [getter(sr) for sr in correct_to_correct]

                stats1 = compute_stats(group1)
                stats2 = compute_stats(group2)
                mw = mann_whitney_u_test(group1, group2)
                cd = cohens_d(group1, group2)

                tests.append(
                    StatisticalTestResult(
                        metric_name=metric_name,
                        group1_name="correct→incorrect",
                        group2_name="correct→correct",
                        group1_n=stats1.n,
                        group2_n=stats2.n,
                        group1_mean=stats1.mean,
                        group1_std=stats1.std,
                        group2_mean=stats2.mean,
                        group2_std=stats2.std,
                        delta=stats1.mean - stats2.mean,
                        mann_whitney_u=mw["statistic"],
                        mann_whitney_p=mw["p_value"],
                        cohens_d=cd,
                        significance=get_significance(mw["p_value"]),
                    )
                )

        # (B) answer_changed vs answer_unchanged
        if len(answer_changed) >= 2 and len(answer_unchanged) >= 2:
            for metric_name, getter in metrics_to_test:
                group1 = [getter(sr) for sr in answer_changed]
                group2 = [getter(sr) for sr in answer_unchanged]

                stats1 = compute_stats(group1)
                stats2 = compute_stats(group2)
                mw = mann_whitney_u_test(group1, group2)
                cd = cohens_d(group1, group2)

                tests.append(
                    StatisticalTestResult(
                        metric_name=metric_name,
                        group1_name="answer_changed",
                        group2_name="answer_unchanged",
                        group1_n=stats1.n,
                        group2_n=stats2.n,
                        group1_mean=stats1.mean,
                        group1_std=stats1.std,
                        group2_mean=stats2.mean,
                        group2_std=stats2.std,
                        delta=stats1.mean - stats2.mean,
                        mann_whitney_u=mw["statistic"],
                        mann_whitney_p=mw["p_value"],
                        cohens_d=cd,
                        significance=get_significance(mw["p_value"]),
                    )
                )

        return tests

    def _run_correlation_analysis(self, result: AnalysisResult) -> list[CorrelationResult]:
        """相関分析を実行する.

        Jaccard係数とROUGE-Lの相関を分析し、
        「どこに注目するか」の変化と「生成されるCoT」の変化の関係を検証する。

        分析対象:
        1. Question→CoT: question_jaccard vs cot_rouge_l
        2. CoT→Ans: cot_jaccard vs cot_rouge_l（メイン分析項目）
        """
        logger.info("相関分析を開始...")
        correlations = []

        # 分析対象のJaccard係数（固定トークン数ベース）
        q_top_k_keys = [f"top{k}" for k in Q_TOP_COUNTS]
        cot_top_k_keys = [f"top{k}" for k in COT_TOP_COUNTS]
        logger.info(f"Question Top-k keys: {q_top_k_keys}")
        logger.info(f"CoT Top-k keys: {cot_top_k_keys}")

        # グループ定義
        groups = {
            "all": result.sample_results,
            "correct→correct": [
                sr for sr in result.sample_results if sr.pattern == "correct→correct"
            ],
            "correct→incorrect": [
                sr for sr in result.sample_results if sr.pattern == "correct→incorrect"
            ],
            "answer_changed": [sr for sr in result.sample_results if sr.answer_changed],
            "answer_unchanged": [sr for sr in result.sample_results if not sr.answer_changed],
        }

        for group_name, samples in groups.items():
            if len(samples) < 10:  # 最低10サンプル必要
                continue

            # ROUGE-Lを取得
            rouge_l_values = [sr.cot_rouge_l["f1"] for sr in samples]

            # === Question→CoT: question_spearman_r vs ROUGE-L ===
            spearman_r_values = [sr.question_spearman_r for sr in samples]
            pearson = pearson_correlation(spearman_r_values, rouge_l_values)
            spearman = spearman_correlation(spearman_r_values, rouge_l_values)
            correlations.append(
                CorrelationResult(
                    variable1="question_spearman_r",
                    variable2="cot_rouge_l_f1",
                    group_name=group_name,
                    n=len(samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # === Question→CoT: question_spearman_r vs cot_jaccard（固定トークン数ベース）===
            for k in cot_top_k_keys:
                cot_jaccard_values = [sr.cot_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(spearman_r_values, cot_jaccard_values)
                spearman = spearman_correlation(spearman_r_values, cot_jaccard_values)

                correlations.append(
                    CorrelationResult(
                        variable1="question_spearman_r",
                        variable2=f"cot_jaccard_{k}",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # === Question→CoT: question_jaccard vs ROUGE-L（固定トークン数ベース）===
            for k in q_top_k_keys:
                jaccard_values = [sr.question_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(jaccard_values, rouge_l_values)
                spearman = spearman_correlation(jaccard_values, rouge_l_values)

                correlations.append(
                    CorrelationResult(
                        variable1=f"question_jaccard_{k}",
                        variable2="cot_rouge_l_f1",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # === CoT→Ans: cot_jaccard vs ROUGE-L（固定トークン数ベース、メイン分析項目）===
            for k in cot_top_k_keys:
                cot_jaccard_values = [sr.cot_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(cot_jaccard_values, rouge_l_values)
                spearman = spearman_correlation(cot_jaccard_values, rouge_l_values)

                correlations.append(
                    CorrelationResult(
                        variable1=f"cot_jaccard_{k}",
                        variable2="cot_rouge_l_f1",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # ΔEntropyとROUGE-Lの相関も追加
            delta_entropy_values = [sr.question_delta_entropy for sr in samples]
            pearson = pearson_correlation(delta_entropy_values, rouge_l_values)
            spearman = spearman_correlation(delta_entropy_values, rouge_l_values)

            correlations.append(
                CorrelationResult(
                    variable1="question_delta_entropy",
                    variable2="cot_rouge_l_f1",
                    group_name=group_name,
                    n=len(samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # JS-DivergenceとROUGE-Lの相関
            js_div_values = [sr.question_js_divergence for sr in samples]
            pearson = pearson_correlation(js_div_values, rouge_l_values)
            spearman = spearman_correlation(js_div_values, rouge_l_values)

            correlations.append(
                CorrelationResult(
                    variable1="question_js_divergence",
                    variable2="cot_rouge_l_f1",
                    group_name=group_name,
                    n=len(samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # === CoT→Ans: cot_delta_entropy vs ROUGE-L ===
            cot_delta_entropy_values = [sr.cot_delta_entropy for sr in samples]
            pearson = pearson_correlation(cot_delta_entropy_values, rouge_l_values)
            spearman = spearman_correlation(cot_delta_entropy_values, rouge_l_values)

            correlations.append(
                CorrelationResult(
                    variable1="cot_delta_entropy",
                    variable2="cot_rouge_l_f1",
                    group_name=group_name,
                    n=len(samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # === Question Jaccard vs 二値変数（正誤・回答変化）の相関分析 ===
        # 全サンプルを対象とした分析

        # 1. Question Jaccard vs answer_correctness (correct→correct=1, correct→incorrect=0)
        # 対象: correct→* サンプルのみ（摂動前に正解だったサンプル）
        correct_before_samples = [
            sr
            for sr in result.sample_results
            if sr.pattern in ("correct→correct", "correct→incorrect")
        ]

        if len(correct_before_samples) >= 10:
            # 正誤を二値変数に変換
            answer_correctness = [
                1.0 if sr.pattern == "correct→correct" else 0.0 for sr in correct_before_samples
            ]

            for k in q_top_k_keys:
                q_jaccard_values = [
                    sr.question_jaccard.get(k, 0.0) for sr in correct_before_samples
                ]

                pearson = pearson_correlation(q_jaccard_values, answer_correctness)
                spearman = spearman_correlation(q_jaccard_values, answer_correctness)

                correlations.append(
                    CorrelationResult(
                        variable1=f"question_jaccard_{k}",
                        variable2="answer_correctness",
                        group_name="correct_before",
                        n=len(correct_before_samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

        # 2. Question Jaccard vs answer_changed (changed=1, unchanged=0)
        # 対象: 全サンプル
        all_samples = result.sample_results
        if len(all_samples) >= 10:
            answer_changed = [1.0 if sr.answer_changed else 0.0 for sr in all_samples]

            for k in q_top_k_keys:
                q_jaccard_values = [sr.question_jaccard.get(k, 0.0) for sr in all_samples]

                pearson = pearson_correlation(q_jaccard_values, answer_changed)
                spearman = spearman_correlation(q_jaccard_values, answer_changed)

                correlations.append(
                    CorrelationResult(
                        variable1=f"question_jaccard_{k}",
                        variable2="answer_changed",
                        group_name="all",
                        n=len(all_samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

        # === CoT Jaccard vs 二値変数（正誤・回答変化）の相関分析（件数ベース）===

        # 3. CoT Jaccard vs answer_correctness (correct→correct=1, correct→incorrect=0)
        # 対象: correct→* サンプルのみ
        if len(correct_before_samples) >= 10:
            answer_correctness = [
                1.0 if sr.pattern == "correct→correct" else 0.0 for sr in correct_before_samples
            ]

            for k in cot_top_k_keys:
                cot_jaccard_values = [sr.cot_jaccard.get(k, 0.0) for sr in correct_before_samples]

                pearson = pearson_correlation(cot_jaccard_values, answer_correctness)
                spearman = spearman_correlation(cot_jaccard_values, answer_correctness)

                correlations.append(
                    CorrelationResult(
                        variable1=f"cot_jaccard_{k}",
                        variable2="answer_correctness",
                        group_name="correct_before",
                        n=len(correct_before_samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

        # 4. CoT Jaccard vs answer_changed (changed=1, unchanged=0)
        # 対象: 全サンプル
        if len(all_samples) >= 10:
            answer_changed = [1.0 if sr.answer_changed else 0.0 for sr in all_samples]

            for k in cot_top_k_keys:
                cot_jaccard_values = [sr.cot_jaccard.get(k, 0.0) for sr in all_samples]

                pearson = pearson_correlation(cot_jaccard_values, answer_changed)
                spearman = spearman_correlation(cot_jaccard_values, answer_changed)

                correlations.append(
                    CorrelationResult(
                        variable1=f"cot_jaccard_{k}",
                        variable2="answer_changed",
                        group_name="all",
                        n=len(all_samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

        # ========================================
        # 追加相関分析: Q Jaccard vs CoT Jaccard, ΔEntropy版パターン
        # ========================================

        # 5. Question Jaccard vs CoT Jaccard（同じキーで比較）
        for group_name, samples in groups.items():
            if len(samples) < 10:
                continue

            for k in q_top_k_keys:
                q_jaccard = [sr.question_jaccard.get(k, 0.0) for sr in samples]
                cot_jaccard = [sr.cot_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(q_jaccard, cot_jaccard)
                spearman = spearman_correlation(q_jaccard, cot_jaccard)

                correlations.append(
                    CorrelationResult(
                        variable1=f"question_jaccard_{k}",
                        variable2=f"cot_jaccard_{k}",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # 6. Question ΔEntropy vs CoT Jaccard（件数ベース）
            for k in cot_top_k_keys:
                q_delta_entropy = [sr.question_delta_entropy for sr in samples]
                cot_jaccard = [sr.cot_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(q_delta_entropy, cot_jaccard)
                spearman = spearman_correlation(q_delta_entropy, cot_jaccard)

                correlations.append(
                    CorrelationResult(
                        variable1="question_delta_entropy",
                        variable2=f"cot_jaccard_{k}",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # 6b. Question JS-Divergence vs CoT Jaccard（件数ベース）
            for k in cot_top_k_keys:
                q_js_div = [sr.question_js_divergence for sr in samples]
                cot_jaccard = [sr.cot_jaccard.get(k, 0.0) for sr in samples]

                pearson = pearson_correlation(q_js_div, cot_jaccard)
                spearman = spearman_correlation(q_js_div, cot_jaccard)

                correlations.append(
                    CorrelationResult(
                        variable1="question_js_divergence",
                        variable2=f"cot_jaccard_{k}",
                        group_name=group_name,
                        n=len(samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

            # 7. Question ΔEntropy vs CoT ΔEntropy
            q_delta_entropy = [sr.question_delta_entropy for sr in samples]
            cot_delta_entropy = [sr.cot_delta_entropy for sr in samples]

            pearson = pearson_correlation(q_delta_entropy, cot_delta_entropy)
            spearman = spearman_correlation(q_delta_entropy, cot_delta_entropy)

            correlations.append(
                CorrelationResult(
                    variable1="question_delta_entropy",
                    variable2="cot_delta_entropy",
                    group_name=group_name,
                    n=len(samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 7. Question ΔEntropy vs answer_correctness
        if len(correct_before_samples) >= 10:
            answer_correctness = [
                1.0 if sr.pattern == "correct→correct" else 0.0 for sr in correct_before_samples
            ]
            q_delta_entropy = [sr.question_delta_entropy for sr in correct_before_samples]

            pearson = pearson_correlation(q_delta_entropy, answer_correctness)
            spearman = spearman_correlation(q_delta_entropy, answer_correctness)

            correlations.append(
                CorrelationResult(
                    variable1="question_delta_entropy",
                    variable2="answer_correctness",
                    group_name="correct_before",
                    n=len(correct_before_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 8. Question ΔEntropy vs answer_changed
        if len(all_samples) >= 10:
            answer_changed = [1.0 if sr.answer_changed else 0.0 for sr in all_samples]
            q_delta_entropy = [sr.question_delta_entropy for sr in all_samples]

            pearson = pearson_correlation(q_delta_entropy, answer_changed)
            spearman = spearman_correlation(q_delta_entropy, answer_changed)

            correlations.append(
                CorrelationResult(
                    variable1="question_delta_entropy",
                    variable2="answer_changed",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # 8b. Q-Spearman-r vs answer_changed
            q_spearman_r = [sr.question_spearman_r for sr in all_samples]

            pearson = pearson_correlation(q_spearman_r, answer_changed)
            spearman = spearman_correlation(q_spearman_r, answer_changed)

            correlations.append(
                CorrelationResult(
                    variable1="question_spearman_r",
                    variable2="answer_changed",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # 8c. Q-Token数変化 vs answer_changed
            q_token_diff = [float(sr.question_token_count_diff) for sr in all_samples]

            pearson = pearson_correlation(q_token_diff, answer_changed)
            spearman = spearman_correlation(q_token_diff, answer_changed)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="answer_changed",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

            # 8d. Q-Jaccard@Top-K vs answer_changed
            for k in q_top_k_keys:
                q_jaccard = [sr.question_jaccard.get(k, 0.0) for sr in all_samples]

                pearson = pearson_correlation(q_jaccard, answer_changed)
                spearman = spearman_correlation(q_jaccard, answer_changed)

                correlations.append(
                    CorrelationResult(
                        variable1=f"question_jaccard_{k}",
                        variable2="answer_changed",
                        group_name="all",
                        n=len(all_samples),
                        pearson_r=pearson["correlation"],
                        pearson_p=pearson["p_value"],
                        spearman_rho=spearman["correlation"],
                        spearman_p=spearman["p_value"],
                        interpretation=interpret_correlation(spearman["correlation"]),
                    )
                )

        # 9. CoT ΔEntropy vs answer_correctness
        if len(correct_before_samples) >= 10:
            answer_correctness = [
                1.0 if sr.pattern == "correct→correct" else 0.0 for sr in correct_before_samples
            ]
            cot_delta_entropy = [sr.cot_delta_entropy for sr in correct_before_samples]

            pearson = pearson_correlation(cot_delta_entropy, answer_correctness)
            spearman = spearman_correlation(cot_delta_entropy, answer_correctness)

            correlations.append(
                CorrelationResult(
                    variable1="cot_delta_entropy",
                    variable2="answer_correctness",
                    group_name="correct_before",
                    n=len(correct_before_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 10. CoT ΔEntropy vs answer_changed
        if len(all_samples) >= 10:
            answer_changed = [1.0 if sr.answer_changed else 0.0 for sr in all_samples]
            cot_delta_entropy = [sr.cot_delta_entropy for sr in all_samples]

            pearson = pearson_correlation(cot_delta_entropy, answer_changed)
            spearman = spearman_correlation(cot_delta_entropy, answer_changed)

            correlations.append(
                CorrelationResult(
                    variable1="cot_delta_entropy",
                    variable2="answer_changed",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # ========================================
        # 追加相関分析: Q-トークン数変化 vs 各指標
        # ========================================

        # 11. Q-トークン数変化(abs) vs Q-Spearman-r
        if len(all_samples) >= 10:
            token_diff = [float(sr.question_token_count_diff) for sr in all_samples]
            q_spearman_r = [sr.question_spearman_r for sr in all_samples]

            pearson = pearson_correlation(token_diff, q_spearman_r)
            spearman = spearman_correlation(token_diff, q_spearman_r)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="question_spearman_r",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 12. Q-トークン数変化(abs) vs CoT-ROUGE-L
        if len(all_samples) >= 10:
            token_diff = [float(sr.question_token_count_diff) for sr in all_samples]
            rouge_l = [sr.cot_rouge_l["f1"] for sr in all_samples]

            pearson = pearson_correlation(token_diff, rouge_l)
            spearman = spearman_correlation(token_diff, rouge_l)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="cot_rouge_l_f1",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 13. Q-トークン数変化(abs) vs CoT-Jaccard@Top10
        if len(all_samples) >= 10:
            token_diff = [float(sr.question_token_count_diff) for sr in all_samples]
            cot_jaccard_10 = [sr.cot_jaccard.get("top10", 0.0) for sr in all_samples]

            pearson = pearson_correlation(token_diff, cot_jaccard_10)
            spearman = spearman_correlation(token_diff, cot_jaccard_10)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="cot_jaccard_top10",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 14. Q-トークン数変化(abs) vs 回答正誤 (correct_before)
        if len(correct_before_samples) >= 10:
            token_diff = [float(sr.question_token_count_diff) for sr in correct_before_samples]
            answer_correctness = [
                1.0 if sr.pattern == "correct→correct" else 0.0 for sr in correct_before_samples
            ]

            pearson = pearson_correlation(token_diff, answer_correctness)
            spearman = spearman_correlation(token_diff, answer_correctness)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="answer_correctness",
                    group_name="correct_before",
                    n=len(correct_before_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        # 15. Q-トークン数変化(abs) vs 回答変化
        if len(all_samples) >= 10:
            token_diff = [float(sr.question_token_count_diff) for sr in all_samples]
            answer_changed = [1.0 if sr.answer_changed else 0.0 for sr in all_samples]

            pearson = pearson_correlation(token_diff, answer_changed)
            spearman = spearman_correlation(token_diff, answer_changed)

            correlations.append(
                CorrelationResult(
                    variable1="question_token_count_diff",
                    variable2="answer_changed",
                    group_name="all",
                    n=len(all_samples),
                    pearson_r=pearson["correlation"],
                    pearson_p=pearson["p_value"],
                    spearman_rho=spearman["correlation"],
                    spearman_p=spearman["p_value"],
                    interpretation=interpret_correlation(spearman["correlation"]),
                )
            )

        logger.info(f"相関分析完了: {len(correlations)}件の相関を計算")
        return correlations

    def _run_partial_correlation_analysis(
        self, result: AnalysisResult
    ) -> list[PartialCorrelationResult]:
        """偏相関分析を実行する.

        ROUGE-L（文章類似度）の影響を統制した上で、
        Jaccard係数（注目トークン集合の類似度）と回答の正誤/回答変化の関係を分析する。

        目的:
        「摂動前後での注目トークン集合の類似度が最終的な回答の正誤/変化に影響する」
        ことを、文章類似度の影響を除いた上で示す。

        分析パターン:
        1. answer_correctness（正誤）: correct→correct/incorrect のサンプルのみ
        2. answer_changed（回答変化）: 全サンプル
        """
        logger.info("偏相関分析を開始...")
        partial_correlations = []

        # 分析対象のJaccard係数（固定トークン数ベース）
        q_top_k_keys = [f"top{k}" for k in Q_TOP_COUNTS]
        cot_top_k_keys = [f"top{k}" for k in COT_TOP_COUNTS]

        # ========================================
        # 1. answer_correctness（正誤）をターゲットとした偏相関分析
        # ========================================
        # 対象サンプル: correct→correct と correct→incorrect のみ
        correct_before_samples = [
            sr
            for sr in result.sample_results
            if sr.pattern in ("correct→correct", "correct→incorrect")
        ]

        if len(correct_before_samples) >= 20:
            # 回答の正誤を二値変数に変換（1=correct→correct, 0=correct→incorrect）
            answer_correctness = np.array(
                [1 if sr.pattern == "correct→correct" else 0 for sr in correct_before_samples]
            )

            # ROUGE-L（統制変数）
            rouge_l_values = np.array([sr.cot_rouge_l["f1"] for sr in correct_before_samples])

            # CoT Jaccard係数について偏相関を計算（件数ベース）
            for k in cot_top_k_keys:
                cot_jaccard_values = np.array(
                    [sr.cot_jaccard.get(k, 0.0) for sr in correct_before_samples]
                )
                partial_result = self._compute_partial_correlation(
                    x=cot_jaccard_values,
                    y=answer_correctness,
                    z=rouge_l_values,
                    variable_name=f"cot_jaccard_{k}",
                    control_variable_name="cot_rouge_l_f1",
                    target_name="answer_correctness",
                )
                if partial_result is not None:
                    partial_correlations.append(partial_result)

            # Q指標の準備
            token_diff_values = np.array(
                [float(sr.question_token_count_diff) for sr in correct_before_samples]
            )
            q_spearman_r_values = np.array(
                [sr.question_spearman_r for sr in correct_before_samples]
            )
            q_jaccard_10_values = np.array(
                [sr.question_jaccard.get("top10", 0.0) for sr in correct_before_samples]
            )

            # Q-ΔToken-Num vs 回答正誤 (Q-Spearman-rを統制)
            partial_result = self._compute_partial_correlation(
                x=token_diff_values,
                y=answer_correctness,
                z=q_spearman_r_values,
                variable_name="question_token_count_diff",
                control_variable_name="question_spearman_r",
                target_name="answer_correctness",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Q-ΔToken-Num vs 回答正誤 (Q-Jaccard@10を統制)
            partial_result = self._compute_partial_correlation(
                x=token_diff_values,
                y=answer_correctness,
                z=q_jaccard_10_values,
                variable_name="question_token_count_diff",
                control_variable_name="question_jaccard_top10",
                target_name="answer_correctness",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Question Jaccard係数について偏相関を計算（Q-ΔToken-NumとQ-Spearman-rを統制）
            for k in q_top_k_keys:
                q_jaccard_values = np.array(
                    [sr.question_jaccard.get(k, 0.0) for sr in correct_before_samples]
                )
                # Q-Token Diffを統制
                partial_result_q = self._compute_partial_correlation(
                    x=q_jaccard_values,
                    y=answer_correctness,
                    z=token_diff_values,
                    variable_name=f"question_jaccard_{k}",
                    control_variable_name="question_token_count_diff",
                    target_name="answer_correctness",
                )
                if partial_result_q is not None:
                    partial_correlations.append(partial_result_q)

                # Q-Spearman-rを統制
                partial_result_q = self._compute_partial_correlation(
                    x=q_jaccard_values,
                    y=answer_correctness,
                    z=q_spearman_r_values,
                    variable_name=f"question_jaccard_{k}",
                    control_variable_name="question_spearman_r",
                    target_name="answer_correctness",
                )
                if partial_result_q is not None:
                    partial_correlations.append(partial_result_q)

            # Q-Spearman-r vs 回答正誤 (Q-Token Diffを統制)
            partial_result = self._compute_partial_correlation(
                x=q_spearman_r_values,
                y=answer_correctness,
                z=token_diff_values,
                variable_name="question_spearman_r",
                control_variable_name="question_token_count_diff",
                target_name="answer_correctness",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Q-Spearman-r vs 回答正誤 (Q-Jaccard@10を統制)
            partial_result = self._compute_partial_correlation(
                x=q_spearman_r_values,
                y=answer_correctness,
                z=q_jaccard_10_values,
                variable_name="question_spearman_r",
                control_variable_name="question_jaccard_top10",
                target_name="answer_correctness",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # ΔEntropy についても偏相関を計算
            for var_name, values in [
                ("cot_delta_entropy", [sr.cot_delta_entropy for sr in correct_before_samples]),
                (
                    "question_delta_entropy",
                    [sr.question_delta_entropy for sr in correct_before_samples],
                ),
                (
                    "question_js_divergence",
                    [sr.question_js_divergence for sr in correct_before_samples],
                ),
            ]:
                var_values = np.array(values)
                partial_result = self._compute_partial_correlation(
                    x=var_values,
                    y=answer_correctness,
                    z=rouge_l_values,
                    variable_name=var_name,
                    control_variable_name="cot_rouge_l_f1",
                    target_name="answer_correctness",
                )
                if partial_result is not None:
                    partial_correlations.append(partial_result)
        else:
            logger.warning(
                f"answer_correctness偏相関分析をスキップ: サンプル数不足 ({len(correct_before_samples)} < 20)"
            )

        # ========================================
        # 2. answer_changed（回答変化）をターゲットとした偏相関分析
        # ========================================
        # 対象サンプル: 全サンプル
        all_samples = result.sample_results

        if len(all_samples) >= 20:
            # 回答変化を二値変数に変換（1=changed, 0=unchanged）
            answer_changed = np.array([1 if sr.answer_changed else 0 for sr in all_samples])

            # ROUGE-L（統制変数）
            rouge_l_all = np.array([sr.cot_rouge_l["f1"] for sr in all_samples])

            # CoT Jaccard係数について偏相関を計算（件数ベース）
            for k in cot_top_k_keys:
                cot_jaccard_all = np.array([sr.cot_jaccard.get(k, 0.0) for sr in all_samples])
                partial_result = self._compute_partial_correlation(
                    x=cot_jaccard_all,
                    y=answer_changed,
                    z=rouge_l_all,
                    variable_name=f"cot_jaccard_{k}",
                    control_variable_name="cot_rouge_l_f1",
                    target_name="answer_changed",
                )
                if partial_result is not None:
                    partial_correlations.append(partial_result)

            # Q指標の準備
            token_diff_all = np.array([float(sr.question_token_count_diff) for sr in all_samples])
            q_spearman_r_all = np.array([sr.question_spearman_r for sr in all_samples])
            q_jaccard_10_all = np.array(
                [sr.question_jaccard.get("top10", 0.0) for sr in all_samples]
            )

            # Q-ΔToken-Num vs 回答変化 (Q-Spearman-rを統制)
            partial_result = self._compute_partial_correlation(
                x=token_diff_all,
                y=answer_changed,
                z=q_spearman_r_all,
                variable_name="question_token_count_diff",
                control_variable_name="question_spearman_r",
                target_name="answer_changed",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Q-ΔToken-Num vs 回答変化 (Q-Jaccard@10を統制)
            partial_result = self._compute_partial_correlation(
                x=token_diff_all,
                y=answer_changed,
                z=q_jaccard_10_all,
                variable_name="question_token_count_diff",
                control_variable_name="question_jaccard_top10",
                target_name="answer_changed",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Question Jaccard係数について偏相関を計算（Q-ΔToken-NumとQ-Spearman-rを統制）
            for k in q_top_k_keys:
                q_jaccard_all = np.array([sr.question_jaccard.get(k, 0.0) for sr in all_samples])
                # Q-Token Diffを統制
                partial_result_q = self._compute_partial_correlation(
                    x=q_jaccard_all,
                    y=answer_changed,
                    z=token_diff_all,
                    variable_name=f"question_jaccard_{k}",
                    control_variable_name="question_token_count_diff",
                    target_name="answer_changed",
                )
                if partial_result_q is not None:
                    partial_correlations.append(partial_result_q)

                # Q-Spearman-rを統制
                partial_result_q = self._compute_partial_correlation(
                    x=q_jaccard_all,
                    y=answer_changed,
                    z=q_spearman_r_all,
                    variable_name=f"question_jaccard_{k}",
                    control_variable_name="question_spearman_r",
                    target_name="answer_changed",
                )
                if partial_result_q is not None:
                    partial_correlations.append(partial_result_q)

            # Q-Spearman-r vs 回答変化 (Q-Token Diffを統制)
            partial_result = self._compute_partial_correlation(
                x=q_spearman_r_all,
                y=answer_changed,
                z=token_diff_all,
                variable_name="question_spearman_r",
                control_variable_name="question_token_count_diff",
                target_name="answer_changed",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Q-Spearman-r vs 回答変化 (Q-Jaccard@10を統制)
            partial_result = self._compute_partial_correlation(
                x=q_spearman_r_all,
                y=answer_changed,
                z=q_jaccard_10_all,
                variable_name="question_spearman_r",
                control_variable_name="question_jaccard_top10",
                target_name="answer_changed",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # Q-Spearman-r vs CoT-ROUGE-L (Q-Token Diffを統制) - 実験3-d用
            partial_result = self._compute_partial_correlation(
                x=q_spearman_r_all,
                y=rouge_l_all,
                z=token_diff_all,
                variable_name="question_spearman_r",
                control_variable_name="question_token_count_diff",
                target_name="cot_rouge_l_f1",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

            # ΔEntropy についても偏相関を計算
            for var_name, values in [
                ("cot_delta_entropy", [sr.cot_delta_entropy for sr in all_samples]),
                ("question_delta_entropy", [sr.question_delta_entropy for sr in all_samples]),
                ("question_js_divergence", [sr.question_js_divergence for sr in all_samples]),
            ]:
                var_values = np.array(values)
                partial_result = self._compute_partial_correlation(
                    x=var_values,
                    y=answer_changed,
                    z=rouge_l_all,
                    variable_name=var_name,
                    control_variable_name="cot_rouge_l_f1",
                    target_name="answer_changed",
                )
                if partial_result is not None:
                    partial_correlations.append(partial_result)
        else:
            logger.warning(
                f"answer_changed偏相関分析をスキップ: サンプル数不足 ({len(all_samples)} < 20)"
            )

        # ========================================
        # 3. ROUGE-L の偏相関分析（Jaccard@Top10を統制変数）
        # ========================================
        # answer_correctness
        if len(correct_before_samples) >= 20:
            rouge_l_values = np.array([sr.cot_rouge_l["f1"] for sr in correct_before_samples])
            answer_correctness = np.array(
                [1 if sr.pattern == "correct→correct" else 0 for sr in correct_before_samples]
            )
            cot_jaccard_10 = np.array(
                [sr.cot_jaccard.get("top10", 0.0) for sr in correct_before_samples]
            )

            partial_result = self._compute_partial_correlation(
                x=rouge_l_values,
                y=answer_correctness,
                z=cot_jaccard_10,
                variable_name="cot_rouge_l_f1",
                control_variable_name="cot_jaccard_top10",
                target_name="answer_correctness",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

        # answer_changed
        if len(all_samples) >= 20:
            rouge_l_all = np.array([sr.cot_rouge_l["f1"] for sr in all_samples])
            answer_changed = np.array([1 if sr.answer_changed else 0 for sr in all_samples])
            cot_jaccard_10_all = np.array([sr.cot_jaccard.get("top10", 0.0) for sr in all_samples])

            partial_result = self._compute_partial_correlation(
                x=rouge_l_all,
                y=answer_changed,
                z=cot_jaccard_10_all,
                variable_name="cot_rouge_l_f1",
                control_variable_name="cot_jaccard_top10",
                target_name="answer_changed",
            )
            if partial_result is not None:
                partial_correlations.append(partial_result)

        logger.info(f"偏相関分析完了: {len(partial_correlations)}件の偏相関を計算")
        return partial_correlations

    def _compute_partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        variable_name: str,
        control_variable_name: str,
        target_name: str = "answer_correctness",
    ) -> PartialCorrelationResult | None:
        """偏相関係数を計算する.

        Args:
            x: 独立変数（Jaccard係数など）
            y: 従属変数（回答の正誤/変化: 0/1）
            z: 統制変数（ROUGE-Lなど）
            variable_name: 変数名
            control_variable_name: 統制変数名
            target_name: ターゲット変数名（answer_correctness or answer_changed）

        Returns:
            偏相関分析の結果
        """
        from scipy import stats

        # NaN/Infを除去
        valid_mask = ~(
            np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isinf(x) | np.isinf(y) | np.isinf(z)
        )
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]

        n = len(x)
        if n < 20:
            return None

        # 統制前の単純相関（参考値）
        zero_order_r, zero_order_p = stats.pointbiserialr(y, x)

        # 偏相関の計算
        # 1. xからzの影響を除去
        slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
        residual_x = x - (slope_xz * z + intercept_xz)

        # 2. yからzの影響を除去
        slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
        residual_y = y - (slope_yz * z + intercept_yz)

        # 3. 残差同士の相関（点双列相関）
        # Note: 残差は連続変数なので、ピアソン相関を使用
        partial_r, partial_p = stats.pearsonr(residual_x, residual_y)

        # 結果の解釈
        interpretation = self._interpret_partial_correlation(
            partial_r, partial_p, zero_order_r, zero_order_p
        )

        return PartialCorrelationResult(
            variable=variable_name,
            control_variable=control_variable_name,
            target_variable=target_name,
            n=n,
            partial_r=float(partial_r),
            partial_p=float(partial_p),
            zero_order_r=float(zero_order_r),
            zero_order_p=float(zero_order_p),
            interpretation=interpretation,
        )

    def _interpret_partial_correlation(
        self,
        partial_r: float,
        partial_p: float,
        zero_order_r: float,
        zero_order_p: float,
    ) -> str:
        """偏相関の結果を解釈する."""
        # 偏相関の有意性
        if partial_p < 0.001:
            sig = "***"
        elif partial_p < 0.01:
            sig = "**"
        elif partial_p < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        # 偏相関の強さ
        abs_r = abs(partial_r)
        if abs_r >= 0.5:
            strength = "強い"
        elif abs_r >= 0.3:
            strength = "中程度の"
        elif abs_r >= 0.1:
            strength = "弱い"
        else:
            strength = "ごく弱い"

        # 統制前後の比較
        r_change = partial_r - zero_order_r
        if abs(r_change) < 0.05:
            control_effect = "ROUGE-Lの統制による影響は小さい"
        elif r_change < 0:
            control_effect = "ROUGE-Lの統制により相関が減少"
        else:
            control_effect = "ROUGE-Lの統制により相関が増加"

        direction = "正" if partial_r > 0 else "負"

        if sig == "n.s.":
            return f"{strength}{direction}の相関（有意でない）。{control_effect}"
        else:
            return f"{strength}{direction}の相関（{sig}）。{control_effect}"

    def _analyze_cot_relevance_distribution(
        self, result: AnalysisResult, k_values: list[int] | None = None
    ) -> None:
        """分析1: CoT内トークン vs 質問文内トークンの寄与スコア比較.

        CoT推論過程が回答決定において重要な役割を果たしているかを検証する。
        _cot.ptのtoken_scores（フィルタ前の全トークンスコア）を使用し、
        cot_token_startで質問文（プロンプト）部分とCoT部分に分割して比較する。

        Args:
            result: 分析結果（cot_relevance_analysisに結果を格納）
            k_values: 上位k件のCoT割合を計算するkのリスト
        """
        if k_values is None:
            k_values = [5, 10, 20]

        logger.info("=== 分析1: CoT vs 質問文の寄与スコア比較 ===")

        common_ids = {sr.sample_id for sr in result.sample_results}
        per_sample_results: list[dict] = []
        skipped = 0

        for sample_id in sorted(common_ids):
            # 摂動前のCoT→Answer寄与度データを読み込み
            cot_data = self._load_importance_data(self.before_dir, sample_id, "cot")
            if cot_data is None:
                skipped += 1
                continue

            cot_token_start = cot_data.get("cot_token_start")
            cot_token_end = cot_data.get("cot_token_end")
            if cot_token_start is None or cot_token_end is None:
                skipped += 1
                continue

            scores = cot_data["scores"]

            # 質問文トークン範囲を取得（few-shot例を除外）
            q_range = self._get_question_token_range(self.before_dir, sample_id)
            if q_range is not None:
                q_start, q_end = q_range
                prompt_scores = scores[q_start:q_end]
            else:
                # フォールバック: cot_token_start前の全トークン（従来方式）
                logger.debug(f"質問文範囲取得不可（{sample_id}）: 従来方式にフォールバック")
                prompt_scores = scores[:cot_token_start]

            cot_scores = scores[cot_token_start:cot_token_end]

            if not prompt_scores or not cot_scores:
                skipped += 1
                continue

            prompt_scores_arr = np.array(prompt_scores, dtype=np.float64)
            cot_scores_arr = np.array(cot_scores, dtype=np.float64)

            # 各領域の統計量
            prompt_mean = float(np.mean(prompt_scores_arr))
            prompt_max = float(np.max(prompt_scores_arr))
            prompt_sum = float(np.sum(prompt_scores_arr))
            cot_mean = float(np.mean(cot_scores_arr))
            cot_max = float(np.max(cot_scores_arr))
            cot_sum = float(np.sum(cot_scores_arr))

            # CoT/Prompt比率（promptが0の場合はNone）
            cot_prompt_mean_ratio = cot_mean / prompt_mean if prompt_mean > 0 else None

            # 上位k件中のCoTトークン割合を計算（質問文+CoT範囲のみ対象）
            top_k_cot_ratios: dict[str, float] = {}
            if q_range is not None:
                q_start, q_end = q_range
                # 質問文とCoTのトークンインデックスのみを対象にする
                relevant_indices = list(range(q_start, q_end)) + list(
                    range(cot_token_start, cot_token_end)
                )
            else:
                # フォールバック: cot_token_start前の全トークン + CoT範囲
                relevant_indices = list(range(cot_token_end))
            relevant_scores = [(idx, scores[idx]) for idx in relevant_indices if idx < len(scores)]
            relevant_sorted = sorted(relevant_scores, key=lambda x: x[1], reverse=True)

            for k in k_values:
                top_k_indices = [idx for idx, _ in relevant_sorted[:k]]
                cot_count = sum(
                    1 for idx in top_k_indices if cot_token_start <= idx < cot_token_end
                )
                top_k_cot_ratios[f"top_{k}"] = cot_count / k

            sample_result = {
                "sample_id": sample_id,
                "prompt_token_count": len(prompt_scores),
                "cot_token_count": len(cot_scores),
                "prompt_mean": prompt_mean,
                "prompt_max": prompt_max,
                "prompt_sum": prompt_sum,
                "cot_mean": cot_mean,
                "cot_max": cot_max,
                "cot_sum": cot_sum,
                "cot_prompt_mean_ratio": cot_prompt_mean_ratio,
                "top_k_cot_ratios": top_k_cot_ratios,
            }
            per_sample_results.append(sample_result)

        logger.info(f"分析1: {len(per_sample_results)}サンプル処理完了 (スキップ: {skipped})")

        if not per_sample_results:
            logger.warning("分析1: 有効なサンプルがありません")
            result.cot_relevance_analysis = {"error": "有効なサンプルなし"}
            return

        # 全サンプルの集計統計
        def aggregate_metric(key: str) -> dict:
            values = [s[key] for s in per_sample_results if s[key] is not None]
            if not values:
                return {"mean": None, "std": None, "n": 0}
            arr = np.array(values, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }

        summary = {
            "n_samples": len(per_sample_results),
            "n_skipped": skipped,
            "prompt_mean_score": aggregate_metric("prompt_mean"),
            "prompt_max_score": aggregate_metric("prompt_max"),
            "prompt_sum_score": aggregate_metric("prompt_sum"),
            "cot_mean_score": aggregate_metric("cot_mean"),
            "cot_max_score": aggregate_metric("cot_max"),
            "cot_sum_score": aggregate_metric("cot_sum"),
            "cot_prompt_mean_ratio": aggregate_metric("cot_prompt_mean_ratio"),
        }

        # 上位k件中のCoTトークン割合の集計
        for k in k_values:
            key = f"top_{k}"
            values = [s["top_k_cot_ratios"][key] for s in per_sample_results]
            arr = np.array(values, dtype=np.float64)
            summary[f"top_{k}_cot_ratio"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }

        result.cot_relevance_analysis = {
            "summary": summary,
            "per_sample": per_sample_results,
        }

        # ログ出力
        logger.info(f"  プロンプト平均寄与度: {summary['prompt_mean_score']['mean']:.6f}")
        logger.info(f"  CoT平均寄与度: {summary['cot_mean_score']['mean']:.6f}")
        ratio_info = summary["cot_prompt_mean_ratio"]
        if ratio_info["mean"] is not None:
            logger.info(f"  CoT/プロンプト比率: {ratio_info['mean']:.2f}")
        for k in k_values:
            ratio = summary[f"top_{k}_cot_ratio"]["mean"]
            logger.info(f"  上位{k}件中のCoTトークン割合: {ratio:.1%}")

    def _analyze_cot_token_classification(
        self, result: AnalysisResult, k_values: list[int] | None = None, n_examples: int = 5
    ) -> None:
        """分析2: 回答寄与トークン内の共通/CoT固有トークンの割合.

        CoT内の回答寄与トークン（上位k件）を質問文トークン集合と比較し、
        質問文にも出現する「共通トークン」とCoTにのみ出現する「CoT固有トークン」に分類する。
        摂動前のデータのみを使用する。

        Args:
            result: 分析結果（cot_token_classificationに結果を格納）
            k_values: 上位k件のリスト
            n_examples: 具体例として抽出するサンプル数
        """
        if k_values is None:
            k_values = [5, 10, 20]

        logger.info("=== 分析2: 回答寄与トークンの共通/CoT固有分類 ===")

        common_ids = {sr.sample_id for sr in result.sample_results}
        per_sample_results: list[dict] = []
        skipped = 0

        for sample_id in sorted(common_ids):
            # 摂動前のCoT→Answer寄与度データを読み込み
            cot_data = self._load_importance_data(self.before_dir, sample_id, "cot")
            if cot_data is None:
                skipped += 1
                continue

            cot_token_start = cot_data.get("cot_token_start")
            cot_token_end = cot_data.get("cot_token_end")
            if cot_token_start is None or cot_token_end is None:
                skipped += 1
                continue

            scores = cot_data["scores"]
            tokens = cot_data["tokens"]

            # 質問文トークン範囲を取得（few-shot例を除外）
            q_range = self._get_question_token_range(self.before_dir, sample_id)
            if q_range is not None:
                q_start, q_end = q_range
                prompt_tokens_raw = tokens[q_start:q_end]
            else:
                # フォールバック: cot_token_start前の全トークン（従来方式）
                logger.debug(f"質問文範囲取得不可（{sample_id}）: 従来方式にフォールバック")
                prompt_tokens_raw = tokens[:cot_token_start]

            # 質問文トークン集合を構築（正規化済み）
            prompt_token_set = {normalize_token(t) for t in prompt_tokens_raw}
            # 空文字列を除外
            prompt_token_set.discard("")

            # CoT部分のトークンとスコアを抽出
            cot_tokens_raw = tokens[cot_token_start:cot_token_end]
            cot_scores = scores[cot_token_start:cot_token_end]

            if not cot_tokens_raw:
                skipped += 1
                continue

            # CoT内スコアの降順でインデックスをソート
            cot_indexed = sorted(enumerate(cot_scores), key=lambda x: x[1], reverse=True)

            # 各kについて分類
            sample_k_results: dict[str, dict] = {}
            for k in k_values:
                top_k = cot_indexed[:k]
                common_tokens: list[str] = []
                unique_tokens: list[str] = []

                for local_idx, _score in top_k:
                    raw_token = cot_tokens_raw[local_idx]
                    normalized = normalize_token(raw_token)
                    if not normalized:
                        # 空トークン（特殊文字のみ）はCoT固有として扱う
                        unique_tokens.append(raw_token)
                        continue
                    if normalized in prompt_token_set:
                        common_tokens.append(raw_token)
                    else:
                        unique_tokens.append(raw_token)

                actual_k = len(top_k)
                sample_k_results[f"top_{k}"] = {
                    "common_count": len(common_tokens),
                    "unique_count": len(unique_tokens),
                    "common_ratio": len(common_tokens) / actual_k if actual_k > 0 else 0.0,
                    "unique_ratio": len(unique_tokens) / actual_k if actual_k > 0 else 0.0,
                    "common_tokens": common_tokens,
                    "unique_tokens": unique_tokens,
                }

            per_sample_results.append(
                {
                    "sample_id": sample_id,
                    "prompt_token_count": len(prompt_tokens_raw),
                    "prompt_unique_token_count": len(prompt_token_set),
                    "cot_token_count": len(cot_tokens_raw),
                    "k_results": sample_k_results,
                }
            )

        logger.info(f"分析2: {len(per_sample_results)}サンプル処理完了 (スキップ: {skipped})")

        if not per_sample_results:
            logger.warning("分析2: 有効なサンプルがありません")
            result.cot_token_classification = {"error": "有効なサンプルなし"}
            return

        # 全サンプルの集計統計
        summary: dict[str, dict] = {"n_samples": len(per_sample_results), "n_skipped": skipped}

        for k in k_values:
            key = f"top_{k}"
            common_ratios = [s["k_results"][key]["common_ratio"] for s in per_sample_results]
            unique_ratios = [s["k_results"][key]["unique_ratio"] for s in per_sample_results]
            common_counts = [s["k_results"][key]["common_count"] for s in per_sample_results]
            unique_counts = [s["k_results"][key]["unique_count"] for s in per_sample_results]

            summary[key] = {
                "common_ratio": {
                    "mean": float(np.mean(common_ratios)),
                    "std": float(np.std(common_ratios, ddof=1)) if len(common_ratios) > 1 else 0.0,
                },
                "unique_ratio": {
                    "mean": float(np.mean(unique_ratios)),
                    "std": float(np.std(unique_ratios, ddof=1)) if len(unique_ratios) > 1 else 0.0,
                },
                "common_count": {
                    "mean": float(np.mean(common_counts)),
                    "std": float(np.std(common_counts, ddof=1)) if len(common_counts) > 1 else 0.0,
                },
                "unique_count": {
                    "mean": float(np.mean(unique_counts)),
                    "std": float(np.std(unique_counts, ddof=1)) if len(unique_counts) > 1 else 0.0,
                },
            }

        # 具体例を抽出（k=10でCoT固有トークン割合が高いサンプル上位n_examples件）
        examples: list[dict] = []
        if per_sample_results and "top_10" in per_sample_results[0]["k_results"]:
            sorted_by_unique = sorted(
                per_sample_results,
                key=lambda s: s["k_results"]["top_10"]["unique_ratio"],
                reverse=True,
            )
            for s in sorted_by_unique[:n_examples]:
                examples.append(
                    {
                        "sample_id": s["sample_id"],
                        "common_tokens": s["k_results"]["top_10"]["common_tokens"],
                        "unique_tokens": s["k_results"]["top_10"]["unique_tokens"],
                        "common_ratio": s["k_results"]["top_10"]["common_ratio"],
                        "unique_ratio": s["k_results"]["top_10"]["unique_ratio"],
                    }
                )

        # per_sampleからトークンリストを除外した軽量版を作成（JSON出力用）
        per_sample_compact: list[dict] = []
        for s in per_sample_results:
            compact = {
                "sample_id": s["sample_id"],
                "prompt_token_count": s["prompt_token_count"],
                "prompt_unique_token_count": s["prompt_unique_token_count"],
                "cot_token_count": s["cot_token_count"],
                "k_results": {},
            }
            for k in k_values:
                key = f"top_{k}"
                kr = s["k_results"][key]
                compact["k_results"][key] = {
                    "common_count": kr["common_count"],
                    "unique_count": kr["unique_count"],
                    "common_ratio": kr["common_ratio"],
                    "unique_ratio": kr["unique_ratio"],
                }
            per_sample_compact.append(compact)

        result.cot_token_classification = {
            "summary": summary,
            "examples": examples,
            "per_sample": per_sample_compact,
        }

        # ログ出力
        for k in k_values:
            key = f"top_{k}"
            common_r = summary[key]["common_ratio"]["mean"]
            unique_r = summary[key]["unique_ratio"]["mean"]
            logger.info(f"  @{k}: 共通={common_r:.1%}, CoT固有={unique_r:.1%}")

        if examples:
            logger.info(f"  具体例 (CoT固有割合が高い上位{n_examples}件):")
            for ex in examples:
                logger.info(
                    f"    {ex['sample_id']}: 共通={ex['common_ratio']:.0%}, "
                    f"固有={ex['unique_ratio']:.0%}"
                )

    def _analyze_cot_stratified_jaccard(
        self, result: AnalysisResult, k_values: list[int] | None = None
    ) -> None:
        """分析3: 共通/CoT固有トークンそれぞれのJaccard@kと回答変化の相関.

        摂動前後のCoT内回答寄与トークン（上位k件）を質問文トークンとの共起で分類し、
        - 共通トークン（質問文にも出現）のJaccard@k
        - CoT固有トークン（質問文に出現しない）のJaccard@k
        をそれぞれ計算し、回答変化との相関・偏相関を分析する。

        Args:
            result: 分析結果（cot_stratified_jaccard_analysisに結果を格納）
            k_values: 上位k件のリスト
        """
        from scipy import stats as sp_stats

        if k_values is None:
            k_values = [5, 10, 20]

        logger.info("=== 分析3: 層別Jaccard@kと回答変化の相関分析 ===")

        # sample_resultsからルックアップテーブルを作成
        sample_info: dict[str, dict] = {}
        for sr in result.sample_results:
            sample_info[sr.sample_id] = {
                "answer_changed": sr.answer_changed,
                "pattern": sr.pattern,
                "before_correct": sr.before_correct,
            }

        per_sample_results: list[dict] = []
        skipped = 0

        for sample_id in tqdm(sorted(sample_info.keys()), desc="分析3", unit="sample"):
            info = sample_info[sample_id]

            # 摂動前・後のCoT→Answer寄与度データを読み込み
            before_cot = self._load_importance_data(self.before_dir, sample_id, "cot")
            after_cot = self._load_importance_data(self.after_dir, sample_id, "cot")

            if before_cot is None or after_cot is None:
                skipped += 1
                continue

            before_start = before_cot.get("cot_token_start")
            before_end = before_cot.get("cot_token_end")
            after_start = after_cot.get("cot_token_start")
            after_end = after_cot.get("cot_token_end")

            if any(v is None for v in [before_start, before_end, after_start, after_end]):
                skipped += 1
                continue

            # 摂動前の質問文トークン集合を基準に共通/固有を判定（few-shot除外）
            q_range = self._get_question_token_range(self.before_dir, sample_id)
            if q_range is not None:
                q_start, q_end = q_range
                before_prompt_tokens = before_cot["tokens"][q_start:q_end]
            else:
                logger.debug(f"質問文範囲取得不可（{sample_id}）: 従来方式にフォールバック")
                before_prompt_tokens = before_cot["tokens"][:before_start]
            prompt_token_set = {normalize_token(t) for t in before_prompt_tokens}
            prompt_token_set.discard("")

            # 摂動前後のCoT部分を抽出
            before_cot_tokens = before_cot["tokens"][before_start:before_end]
            before_cot_scores = before_cot["scores"][before_start:before_end]
            after_cot_tokens = after_cot["tokens"][after_start:after_end]
            after_cot_scores = after_cot["scores"][after_start:after_end]

            if not before_cot_tokens or not after_cot_tokens:
                skipped += 1
                continue

            sample_k_results: dict[str, dict] = {}
            for k in k_values:
                # 摂動前: top-kトークンを共通/固有に分類
                before_indexed = sorted(
                    enumerate(before_cot_scores), key=lambda x: x[1], reverse=True
                )
                before_top_k = before_indexed[:k]
                before_common: set[str] = set()
                before_unique: set[str] = set()

                for idx, _score in before_top_k:
                    normalized = normalize_token(before_cot_tokens[idx])
                    if not normalized:
                        before_unique.add(before_cot_tokens[idx])
                        continue
                    if normalized in prompt_token_set:
                        before_common.add(normalized)
                    else:
                        before_unique.add(normalized)

                # 摂動後: top-kトークンを共通/固有に分類
                after_indexed = sorted(
                    enumerate(after_cot_scores), key=lambda x: x[1], reverse=True
                )
                after_top_k = after_indexed[:k]
                after_common: set[str] = set()
                after_unique: set[str] = set()

                for idx, _score in after_top_k:
                    normalized = normalize_token(after_cot_tokens[idx])
                    if not normalized:
                        after_unique.add(after_cot_tokens[idx])
                        continue
                    if normalized in prompt_token_set:
                        after_common.add(normalized)
                    else:
                        after_unique.add(normalized)

                # 共通トークン集合のJaccard
                common_union = len(before_common | after_common)
                jaccard_common = (
                    len(before_common & after_common) / common_union
                    if common_union > 0
                    else float("nan")
                )

                # CoT固有トークン集合のJaccard
                unique_union = len(before_unique | after_unique)
                jaccard_unique = (
                    len(before_unique & after_unique) / unique_union
                    if unique_union > 0
                    else float("nan")
                )

                # 全体のJaccard（参考値）
                before_all = before_common | before_unique
                after_all = after_common | after_unique
                all_union = len(before_all | after_all)
                jaccard_all = (
                    len(before_all & after_all) / all_union if all_union > 0 else float("nan")
                )

                sample_k_results[f"top_{k}"] = {
                    "jaccard_common": jaccard_common,
                    "jaccard_unique": jaccard_unique,
                    "jaccard_all": jaccard_all,
                    "before_common_count": len(before_common),
                    "before_unique_count": len(before_unique),
                    "after_common_count": len(after_common),
                    "after_unique_count": len(after_unique),
                }

            per_sample_results.append(
                {
                    "sample_id": sample_id,
                    "answer_changed": info["answer_changed"],
                    "pattern": info["pattern"],
                    "before_correct": info["before_correct"],
                    "k_results": sample_k_results,
                }
            )

        logger.info(f"分析3: {len(per_sample_results)}サンプル処理完了 (スキップ: {skipped})")

        if not per_sample_results:
            logger.warning("分析3: 有効なサンプルがありません")
            result.cot_stratified_jaccard_analysis = {"error": "有効なサンプルなし"}
            return

        # 集計統計
        summary: dict[str, dict | int] = {
            "n_samples": len(per_sample_results),
            "n_skipped": skipped,
        }

        for k in k_values:
            key = f"top_{k}"
            jc_commons = [s["k_results"][key]["jaccard_common"] for s in per_sample_results]
            jc_uniques = [s["k_results"][key]["jaccard_unique"] for s in per_sample_results]
            jc_alls = [s["k_results"][key]["jaccard_all"] for s in per_sample_results]

            # NaN除去した統計
            valid_c = [v for v in jc_commons if not math.isnan(v)]
            valid_u = [v for v in jc_uniques if not math.isnan(v)]
            valid_a = [v for v in jc_alls if not math.isnan(v)]

            summary[key] = {
                "jaccard_common": {
                    "mean": float(np.mean(valid_c)) if valid_c else float("nan"),
                    "std": float(np.std(valid_c, ddof=1)) if len(valid_c) > 1 else 0.0,
                    "n": len(valid_c),
                },
                "jaccard_unique": {
                    "mean": float(np.mean(valid_u)) if valid_u else float("nan"),
                    "std": float(np.std(valid_u, ddof=1)) if len(valid_u) > 1 else 0.0,
                    "n": len(valid_u),
                },
                "jaccard_all": {
                    "mean": float(np.mean(valid_a)) if valid_a else float("nan"),
                    "std": float(np.std(valid_a, ddof=1)) if len(valid_a) > 1 else 0.0,
                    "n": len(valid_a),
                },
            }

        # 相関分析（k=10を基準に実施）
        correlations: dict[str, dict] = {}
        partial_correlations: dict[str, dict] = {}
        k_for_corr = 10
        key_corr = f"top_{k_for_corr}"

        if key_corr in per_sample_results[0]["k_results"]:
            answer_changed_arr = np.array(
                [1.0 if s["answer_changed"] else 0.0 for s in per_sample_results]
            )

            jc_common_arr = np.array(
                [s["k_results"][key_corr]["jaccard_common"] for s in per_sample_results]
            )
            jc_unique_arr = np.array(
                [s["k_results"][key_corr]["jaccard_unique"] for s in per_sample_results]
            )
            jc_all_arr = np.array(
                [s["k_results"][key_corr]["jaccard_all"] for s in per_sample_results]
            )

            # NaN除去マスク
            valid_common_mask = ~np.isnan(jc_common_arr)
            valid_unique_mask = ~np.isnan(jc_unique_arr)
            valid_all_mask = ~np.isnan(jc_all_arr)
            valid_both_mask = valid_common_mask & valid_unique_mask

            # Spearman相関: jaccard_common vs answer_changed
            if np.sum(valid_common_mask) >= 20:
                rho, p = sp_stats.spearmanr(
                    jc_common_arr[valid_common_mask], answer_changed_arr[valid_common_mask]
                )
                correlations["jaccard_common_vs_answer_changed"] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(np.sum(valid_common_mask)),
                    "interpretation": interpret_correlation(float(rho)),
                }

            # Spearman相関: jaccard_unique vs answer_changed
            if np.sum(valid_unique_mask) >= 20:
                rho, p = sp_stats.spearmanr(
                    jc_unique_arr[valid_unique_mask], answer_changed_arr[valid_unique_mask]
                )
                correlations["jaccard_unique_vs_answer_changed"] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(np.sum(valid_unique_mask)),
                    "interpretation": interpret_correlation(float(rho)),
                }

            # Spearman相関: jaccard_all vs answer_changed（参考値）
            if np.sum(valid_all_mask) >= 20:
                rho, p = sp_stats.spearmanr(
                    jc_all_arr[valid_all_mask], answer_changed_arr[valid_all_mask]
                )
                correlations["jaccard_all_vs_answer_changed"] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": int(np.sum(valid_all_mask)),
                    "interpretation": interpret_correlation(float(rho)),
                }

            # correct→correct vs correct→incorrect のみの相関
            cc_ci_mask = np.array([s["before_correct"] for s in per_sample_results])
            if np.sum(cc_ci_mask) >= 20:
                cc_ci_answer = np.array(
                    [1.0 if s["pattern"] == "correct→correct" else 0.0 for s in per_sample_results]
                )

                cc_ci_valid_common = cc_ci_mask & valid_common_mask
                if np.sum(cc_ci_valid_common) >= 20:
                    rho, p = sp_stats.spearmanr(
                        jc_common_arr[cc_ci_valid_common], cc_ci_answer[cc_ci_valid_common]
                    )
                    correlations["jaccard_common_vs_correctness"] = {
                        "spearman_rho": float(rho),
                        "p_value": float(p),
                        "n": int(np.sum(cc_ci_valid_common)),
                        "interpretation": interpret_correlation(float(rho)),
                    }

                cc_ci_valid_unique = cc_ci_mask & valid_unique_mask
                if np.sum(cc_ci_valid_unique) >= 20:
                    rho, p = sp_stats.spearmanr(
                        jc_unique_arr[cc_ci_valid_unique], cc_ci_answer[cc_ci_valid_unique]
                    )
                    correlations["jaccard_unique_vs_correctness"] = {
                        "spearman_rho": float(rho),
                        "p_value": float(p),
                        "n": int(np.sum(cc_ci_valid_unique)),
                        "interpretation": interpret_correlation(float(rho)),
                    }

            # 偏相関: jaccard_common | jaccard_unique vs answer_changed
            if np.sum(valid_both_mask) >= 20:
                x_common = jc_common_arr[valid_both_mask]
                x_unique = jc_unique_arr[valid_both_mask]
                y_changed = answer_changed_arr[valid_both_mask]

                # jaccard_common vs answer_changed | jaccard_unique
                pc_result = self._compute_partial_correlation(
                    x=x_common,
                    y=y_changed,
                    z=x_unique,
                    variable_name="jaccard_common",
                    control_variable_name="jaccard_unique",
                    target_name="answer_changed",
                )
                if pc_result is not None:
                    partial_correlations["jaccard_common_given_unique"] = {
                        "partial_r": pc_result.partial_r,
                        "partial_p": pc_result.partial_p,
                        "zero_order_r": pc_result.zero_order_r,
                        "zero_order_p": pc_result.zero_order_p,
                        "n": pc_result.n,
                    }

                # jaccard_unique vs answer_changed | jaccard_common
                pc_result = self._compute_partial_correlation(
                    x=x_unique,
                    y=y_changed,
                    z=x_common,
                    variable_name="jaccard_unique",
                    control_variable_name="jaccard_common",
                    target_name="answer_changed",
                )
                if pc_result is not None:
                    partial_correlations["jaccard_unique_given_common"] = {
                        "partial_r": pc_result.partial_r,
                        "partial_p": pc_result.partial_p,
                        "zero_order_r": pc_result.zero_order_r,
                        "zero_order_p": pc_result.zero_order_p,
                        "n": pc_result.n,
                    }

                # 多重共線性チェック: jaccard_common vs jaccard_unique の相関
                rho_collin, p_collin = sp_stats.spearmanr(x_common, x_unique)
                partial_correlations["multicollinearity_check"] = {
                    "jaccard_common_vs_unique_rho": float(rho_collin),
                    "p_value": float(p_collin),
                    "n": int(np.sum(valid_both_mask)),
                    "interpretation": interpret_correlation(float(rho_collin)),
                }

        # per_sampleのコンパクト版
        per_sample_compact: list[dict] = []
        for s in per_sample_results:
            per_sample_compact.append(
                {
                    "sample_id": s["sample_id"],
                    "answer_changed": s["answer_changed"],
                    "pattern": s["pattern"],
                    "k_results": s["k_results"],
                }
            )

        result.cot_stratified_jaccard_analysis = {
            "summary": summary,
            "correlations": correlations,
            "partial_correlations": partial_correlations,
            "per_sample": per_sample_compact,
        }

        # ログ出力
        for k in k_values:
            key = f"top_{k}"
            stats_k = summary[key]
            jc_c = stats_k["jaccard_common"]["mean"]
            jc_u = stats_k["jaccard_unique"]["mean"]
            jc_a = stats_k["jaccard_all"]["mean"]
            logger.info(f"  @{k}: J(共通)={jc_c:.3f}, J(固有)={jc_u:.3f}, J(全体)={jc_a:.3f}")

        if correlations:
            logger.info("  相関分析:")
            for name, corr in correlations.items():
                logger.info(f"    {name}: ρ={corr['spearman_rho']:.3f} (p={corr['p_value']:.4f})")

        if partial_correlations:
            logger.info("  偏相関分析:")
            for name, pc in partial_correlations.items():
                if "partial_r" in pc:
                    logger.info(f"    {name}: r={pc['partial_r']:.3f} (p={pc['partial_p']:.4f})")
                elif "jaccard_common_vs_unique_rho" in pc:
                    logger.info(f"    多重共線性: ρ={pc['jaccard_common_vs_unique_rho']:.3f}")

    def _extract_example_samples(self, result: AnalysisResult, top_n: int = 5) -> None:
        """具体例サンプルを抽出する.

        定性分析のために、8パターンに該当するサンプルを各5事例ずつ抽出する。
        8パターン: (C→C / C→I) × (ROUGE-L 高/低) × (CoT-Jaccard@10 高/低)
        閾値は全サンプル（C→C + C→I）の中央値を使用。
        各パターンの特徴が最も顕著な事例を優先的に選択する。

        Args:
            result: 分析結果
            top_n: 各パターンで抽出するサンプル数（デフォルト: 5）
        """
        logger.info("具体例サンプルを抽出中...")

        # C→C と C→I のサンプルを抽出
        cc_samples = [sr for sr in result.sample_results if sr.pattern == "correct→correct"]
        ci_samples = [sr for sr in result.sample_results if sr.pattern == "correct→incorrect"]
        target_samples = cc_samples + ci_samples

        if not target_samples:
            logger.warning("C→C / C→I のサンプルがありません")
            return

        logger.info(f"対象サンプル数: C→C={len(cc_samples)}, C→I={len(ci_samples)}")

        def extract_sample_metrics(sr: SamplePairResult) -> dict:
            """サンプルから詳細メトリクスを抽出する."""
            return {
                "sample_id": sr.sample_id,
                "pattern": sr.pattern,
                "cot_jaccard_10": sr.cot_jaccard.get("top10", 0.0),
                "rouge_l": sr.cot_rouge_l["f1"],
                "q_jaccard_10": sr.question_jaccard.get("top10", 0.0),
            }

        # 全サンプルの中央値を閾値として計算
        rouge_l_values = [sr.cot_rouge_l["f1"] for sr in target_samples]
        jaccard_10_values = [sr.cot_jaccard.get("top10", 0.0) for sr in target_samples]
        rouge_l_median = float(np.median(rouge_l_values))
        jaccard_10_median = float(np.median(jaccard_10_values))
        logger.info(f"閾値 - ROUGE-L中央値: {rouge_l_median:.4f}")
        logger.info(f"閾値 - CoT-Jaccard@10中央値: {jaccard_10_median:.4f}")

        # 8パターンの定義: (パターン名, 対象サンプル, ROUGE-L条件, Jaccard条件, スコア計算)
        # スコア: 各パターンの特徴が顕著なほど高くなる
        pattern_defs: list[tuple[str, list[SamplePairResult], str, str]] = [
            ("cc_rouge_high_jaccard_high", cc_samples, "high", "high"),
            ("cc_rouge_high_jaccard_low", cc_samples, "high", "low"),
            ("cc_rouge_low_jaccard_high", cc_samples, "low", "high"),
            ("cc_rouge_low_jaccard_low", cc_samples, "low", "low"),
            ("ci_rouge_high_jaccard_high", ci_samples, "high", "high"),
            ("ci_rouge_high_jaccard_low", ci_samples, "high", "low"),
            ("ci_rouge_low_jaccard_high", ci_samples, "low", "high"),
            ("ci_rouge_low_jaccard_low", ci_samples, "low", "low"),
        ]

        for pattern_name, samples, rouge_cond, jaccard_cond in pattern_defs:
            candidates: list[tuple[SamplePairResult, float]] = []
            for sr in samples:
                rouge_l = sr.cot_rouge_l["f1"]
                jaccard_10 = sr.cot_jaccard.get("top10", 0.0)

                # 閾値による条件フィルタリング
                rouge_ok = (
                    rouge_l >= rouge_l_median if rouge_cond == "high" else rouge_l < rouge_l_median
                )
                jaccard_ok = (
                    jaccard_10 >= jaccard_10_median
                    if jaccard_cond == "high"
                    else jaccard_10 < jaccard_10_median
                )

                if rouge_ok and jaccard_ok:
                    # 特徴度スコア: 各条件の極端さの合計
                    rouge_score = rouge_l if rouge_cond == "high" else (1 - rouge_l)
                    jaccard_score = jaccard_10 if jaccard_cond == "high" else (1 - jaccard_10)
                    score = rouge_score + jaccard_score
                    candidates.append((sr, score))

            # スコア降順でソートし、上位top_n件を選択
            candidates.sort(key=lambda x: x[1], reverse=True)
            result.example_samples[pattern_name] = [
                extract_sample_metrics(c[0]) for c in candidates[:top_n]
            ]
            logger.info(
                f"  {pattern_name}: 候補={len(candidates)}件, 抽出={len(result.example_samples[pattern_name])}件"
            )

        total = sum(len(v) for v in result.example_samples.values())
        logger.info(f"具体例サンプル抽出完了: 合計{total}件（8パターン×{top_n}事例）")

    def analyze(self, excluded_sample_ids: set[str] | None = None) -> AnalysisResult:
        """分析を実行する.

        Args:
            excluded_sample_ids: 集計対象から除外する sample_id の集合（union 除外用）.
                呼び出し側で (model, bench) 単位の和集合を計算し渡すことを想定.
                None の場合は per-pair の strict チェックで除外判定する.

        Returns:
            分析結果
        """
        # 共通のサンプルIDを取得
        common_ids = set(self.before_results.keys()) & set(self.after_results.keys())
        logger.info(f"共通サンプル数: {len(common_ids)}")

        if not common_ids:
            raise ValueError("摂動前と摂動後に共通のサンプルがありません")

        # 除外判定:
        # - excluded_sample_ids が与えられた場合: union 除外 ((model, bench) 単位で
        #   全摂動条件にわたって、いずれかで strict 未検出だったサンプルの和集合)
        # - None の場合: per-pair の strict チェック (この pair の before/after
        #   両方で strict 未検出のサンプルのみ)
        analyzable_ids: list[str] = []
        excluded_no_answer_count = 0

        if excluded_sample_ids is not None:
            # Union 除外モード
            for sample_id in sorted(common_ids):
                if sample_id in excluded_sample_ids:
                    excluded_no_answer_count += 1
                    continue
                analyzable_ids.append(sample_id)
            mode_label = "union"
        else:
            # Per-pair strict モード (backward compatible)
            from typo_cot.evaluation.extractor import create_extractor
            strict_extractor = create_extractor(self.metadata["dataset"])
            for sample_id in sorted(common_ids):
                before_text = self.before_results[sample_id].get("generated_text", "") or ""
                after_text = self.after_results[sample_id].get("generated_text", "") or ""
                before_strict = strict_extractor.extract_strict(before_text).strip()
                after_strict = strict_extractor.extract_strict(after_text).strip()
                if not before_strict and not after_strict:
                    excluded_no_answer_count += 1
                    continue
                analyzable_ids.append(sample_id)
            mode_label = "strict per-pair"

        if excluded_no_answer_count > 0:
            logger.info(
                f"回答スパン未検出により除外 ({mode_label}): {excluded_no_answer_count} サンプル "
                f"(集計対象: {len(analyzable_ids)} サンプル)"
            )

        if not analyzable_ids:
            raise ValueError(
                "typo前・後のいずれかで canonical 回答スパンが検出されたサンプルがありません"
            )

        # 結果を格納
        result = AnalysisResult(
            before_dir=str(self.before_dir),
            after_dir=str(self.after_dir),
            total_samples=len(analyzable_ids),
            dataset=self.metadata["dataset"],
            model=self.metadata["model"],
            num_perturbations=self.metadata["num_perturbations"],
            perturbation_type=self.metadata["perturbation_type"],
            excluded_no_answer_count=excluded_no_answer_count,
        )

        # パターン別カウンタ
        pattern_counts: dict[str, int] = dict.fromkeys(self.PATTERNS, 0)
        pattern_samples: dict[str, list[SamplePairResult]] = {p: [] for p in self.PATTERNS}

        answer_changed_count = 0
        answer_unchanged_count = 0
        answer_changed_samples: list[SamplePairResult] = []
        answer_unchanged_samples: list[SamplePairResult] = []

        # 各サンプルを分析
        for sample_id in tqdm(analyzable_ids, desc="分析中", unit="sample"):
            before_data = self.before_results[sample_id]
            after_data = self.after_results[sample_id]

            sample_result = self._analyze_sample_pair(sample_id, before_data, after_data)
            if sample_result is None:
                continue

            result.sample_results.append(sample_result)

            # パターンカウント
            pattern_counts[sample_result.pattern] += 1
            pattern_samples[sample_result.pattern].append(sample_result)

            # 回答変化カウント
            if sample_result.answer_changed:
                answer_changed_count += 1
                answer_changed_samples.append(sample_result)
            else:
                answer_unchanged_count += 1
                answer_unchanged_samples.append(sample_result)

        # パターン別メトリクスを計算（平均と標準偏差）
        result.pattern_counts = pattern_counts
        result.answer_changed_count = answer_changed_count
        result.answer_unchanged_count = answer_unchanged_count

        for pattern in self.PATTERNS:
            if pattern_samples[pattern]:
                result.pattern_metrics[pattern] = self._compute_group_metrics(
                    pattern_samples[pattern]
                )
            else:
                result.pattern_metrics[pattern] = {}

        # answer_changed/unchanged別メトリクスを計算
        result.answer_change_metrics = {
            "answer_changed": self._compute_group_metrics(answer_changed_samples),
            "answer_unchanged": self._compute_group_metrics(answer_unchanged_samples),
        }

        # 全体メトリクスを計算
        result.overall_metrics = self._compute_group_metrics(result.sample_results)

        # 統計的検定を実行
        result.statistical_tests = self._run_statistical_tests(result)

        # 相関分析を実行
        result.correlation_results = self._run_correlation_analysis(result)

        # 偏相関分析を実行（ROUGE-Lの影響を統制）
        result.partial_correlation_results = self._run_partial_correlation_analysis(result)

        # 具体例サンプルを抽出
        self._extract_example_samples(result)

        # CoT寄与度分析（分析1: CoT vs 質問文の寄与スコア比較）
        self._analyze_cot_relevance_distribution(result)

        # CoTトークン分類（分析2: 共通/CoT固有トークンの割合）
        self._analyze_cot_token_classification(result)

        # 層別Jaccard相関分析（分析3: 共通/CoT固有のJaccardと回答変化の相関）
        self._analyze_cot_stratified_jaccard(result)

        return result

    def save_results(self, result: AnalysisResult, output_dir: Path) -> dict[str, Path]:
        """分析結果を実験シートごとに分割して保存する.

        出力ファイル:
        - exp2-a.json: 質問文への影響指標 (ΔToken Num, Q:Jaccard@m, Q:Spearman-ρ)
        - exp2-b.json: CoT推論過程への影響 (CoT:ROUGE-L, CoT:Jaccard@m)
        - exp3.json: Q指標とCoT指標の相関
        - exp4-a.json: CoT指標と回答変化の偏相関
        - exp4-b.json: CoT指標と不正解転落の偏相関
        - example_samples.json: 具体例サンプルID
        - cot_analysis_1.json: CoT vs 質問文の寄与スコア比較
        - cot_analysis_2.json: 回答寄与トークンの共通/CoT固有分類
        - cot_analysis_3.json: 層別Jaccard@kと回答変化の相関分析
        - full_results.json: 全データ（デバッグ/詳細分析用）

        Args:
            result: 分析結果
            output_dir: 出力ディレクトリ

        Returns:
            ファイル名 -> パスの辞書
        """
        output_dir = Path(output_dir)

        # サブフォルダ構造を作成: {dataset}/{model}/k{num}_{type}/
        subdir_name = f"k{result.num_perturbations}"
        if result.perturbation_type:
            subdir_name += f"_{result.perturbation_type}"

        output_subdir = output_dir / result.dataset / result.model / subdir_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 共通メタデータ
        common_metadata = {
            "dataset": result.dataset,
            "model": result.model,
            "num_perturbations": result.num_perturbations,
            "perturbation_type": result.perturbation_type,
            "total_samples": result.total_samples,
            "excluded_no_answer_count": result.excluded_no_answer_count,
            "before_dir": result.before_dir,
            "after_dir": result.after_dir,
        }

        saved_files: dict[str, Path] = {}

        # ========================================
        # 実験2-a: 質問文への影響指標
        # ΔToken Num, Q:Jaccard@m (m=3,5,10), Q:Spearman-ρ
        # ========================================
        exp2a_data = {
            "metadata": common_metadata,
            "overall_metrics": {
                "token_count_diff": result.overall_metrics.get("question_token_count_diff", {}),
                "q_jaccard_top3": result.overall_metrics.get("question_jaccard_top3", {}),
                "q_jaccard_top5": result.overall_metrics.get("question_jaccard_top5", {}),
                "q_jaccard_top10": result.overall_metrics.get("question_jaccard_top10", {}),
                "q_spearman_r": result.overall_metrics.get("question_spearman_r", {}),
            },
        }
        exp2a_path = output_subdir / "exp2-a.json"
        self._save_json(exp2a_data, exp2a_path)
        saved_files["exp2-a"] = exp2a_path

        # ========================================
        # 実験2-b: CoT推論過程への影響
        # CoT:ROUGE-L, CoT:Jaccard@m (m=3,5,10)
        # ========================================
        exp2b_data = {
            "metadata": common_metadata,
            "overall_metrics": {
                "cot_rouge_l_f1": result.overall_metrics.get("cot_rouge_l_f1", {}),
                "cot_jaccard_top3": result.overall_metrics.get("cot_jaccard_top3", {}),
                "cot_jaccard_top5": result.overall_metrics.get("cot_jaccard_top5", {}),
                "cot_jaccard_top10": result.overall_metrics.get("cot_jaccard_top10", {}),
                "cot_jaccard_top15": result.overall_metrics.get("cot_jaccard_top15", {}),
                "cot_jaccard_top20": result.overall_metrics.get("cot_jaccard_top20", {}),
            },
        }
        exp2b_path = output_subdir / "exp2-b.json"
        self._save_json(exp2b_data, exp2b_path)
        saved_files["exp2-b"] = exp2b_path

        # ========================================
        # 実験3: Q指標とCoT指標の相関
        # Heatmap用のQ指標×CoT指標の相関行列
        # ========================================
        # Q指標: Q:Jaccard@3, Q:Jaccard@5, Q:Jaccard@10, Q:Spearman-ρ
        # CoT指標: CoT:ROUGE-L, CoT:Jaccard@3, CoT:Jaccard@5, CoT:Jaccard@10
        exp3_correlations = [
            {
                "variable1": c.variable1,
                "variable2": c.variable2,
                "group_name": c.group_name,
                "n": c.n,
                "spearman_rho": c.spearman_rho,
                "spearman_p": c.spearman_p,
            }
            for c in result.correlation_results
        ]

        exp3_data = {
            "metadata": common_metadata,
            "correlations": exp3_correlations,
        }
        exp3_path = output_subdir / "exp3.json"
        self._save_json(exp3_data, exp3_path)
        saved_files["exp3"] = exp3_path

        # ========================================
        # 実験4-a: CoT指標と回答変化の偏相関
        # answer_changed vs answer_unchanged でグループ分け
        # ROUGE-L|Jaccard, Jaccard|ROUGE-L の偏相関
        # ========================================
        exp4a_partial_correlations = [
            {
                "variable": pc.variable,
                "control_variable": pc.control_variable,
                "target_variable": pc.target_variable,
                "group_name": pc.group_name if hasattr(pc, "group_name") else "all",
                "n": pc.n,
                "partial_r": pc.partial_r,
                "partial_p": pc.partial_p,
                "zero_order_r": pc.zero_order_r,
                "zero_order_p": pc.zero_order_p,
            }
            for pc in result.partial_correlation_results
            if pc.target_variable == "answer_changed"
        ]

        exp4a_data = {
            "metadata": common_metadata,
            "answer_change_counts": {
                "changed": result.answer_changed_count,
                "unchanged": result.answer_unchanged_count,
            },
            "partial_correlations": exp4a_partial_correlations,
        }
        exp4a_path = output_subdir / "exp4-a.json"
        self._save_json(exp4a_data, exp4a_path)
        saved_files["exp4-a"] = exp4a_path

        # ========================================
        # 実験4-b: CoT指標と不正解転落の偏相関
        # correct→correct vs correct→incorrect でグループ分け
        # ========================================
        exp4b_partial_correlations = [
            {
                "variable": pc.variable,
                "control_variable": pc.control_variable,
                "target_variable": pc.target_variable,
                "group_name": pc.group_name if hasattr(pc, "group_name") else "all",
                "n": pc.n,
                "partial_r": pc.partial_r,
                "partial_p": pc.partial_p,
                "zero_order_r": pc.zero_order_r,
                "zero_order_p": pc.zero_order_p,
            }
            for pc in result.partial_correlation_results
            if pc.target_variable == "answer_correctness"
        ]

        # correct→correct vs correct→incorrect のカウント
        cc_count = result.pattern_counts.get("correct→correct", 0)
        ci_count = result.pattern_counts.get("correct→incorrect", 0)

        exp4b_data = {
            "metadata": common_metadata,
            "pattern_counts": {
                "correct_to_correct": cc_count,
                "correct_to_incorrect": ci_count,
            },
            "partial_correlations": exp4b_partial_correlations,
        }
        exp4b_path = output_subdir / "exp4-b.json"
        self._save_json(exp4b_data, exp4b_path)
        saved_files["exp4-b"] = exp4b_path

        # ========================================
        # 具体例サンプルID（定性分析用）
        # ========================================
        example_samples_data = {
            "metadata": common_metadata,
            "metrics_included": [
                "sample_id",
                "pattern",
                "cot_jaccard_10",
                "rouge_l",
                "q_jaccard_10",
            ],
            "description": "8パターン分類: (C→C/C→I) × (ROUGE-L高/低) × (CoT-Jaccard@10高/低)、各5事例",
            "categories": {
                "cc_rouge_high_jaccard_high": {
                    "description": "C→C & ROUGE-L高 & CoT-Jaccard@10高",
                    "samples": result.example_samples["cc_rouge_high_jaccard_high"],
                },
                "cc_rouge_high_jaccard_low": {
                    "description": "C→C & ROUGE-L高 & CoT-Jaccard@10低",
                    "samples": result.example_samples["cc_rouge_high_jaccard_low"],
                },
                "cc_rouge_low_jaccard_high": {
                    "description": "C→C & ROUGE-L低 & CoT-Jaccard@10高",
                    "samples": result.example_samples["cc_rouge_low_jaccard_high"],
                },
                "cc_rouge_low_jaccard_low": {
                    "description": "C→C & ROUGE-L低 & CoT-Jaccard@10低",
                    "samples": result.example_samples["cc_rouge_low_jaccard_low"],
                },
                "ci_rouge_high_jaccard_high": {
                    "description": "C→I & ROUGE-L高 & CoT-Jaccard@10高",
                    "samples": result.example_samples["ci_rouge_high_jaccard_high"],
                },
                "ci_rouge_high_jaccard_low": {
                    "description": "C→I & ROUGE-L高 & CoT-Jaccard@10低",
                    "samples": result.example_samples["ci_rouge_high_jaccard_low"],
                },
                "ci_rouge_low_jaccard_high": {
                    "description": "C→I & ROUGE-L低 & CoT-Jaccard@10高",
                    "samples": result.example_samples["ci_rouge_low_jaccard_high"],
                },
                "ci_rouge_low_jaccard_low": {
                    "description": "C→I & ROUGE-L低 & CoT-Jaccard@10低",
                    "samples": result.example_samples["ci_rouge_low_jaccard_low"],
                },
            },
        }
        example_path = output_subdir / "example_samples.json"
        self._save_json(example_samples_data, example_path)
        saved_files["example_samples"] = example_path

        # ========================================
        # 分析1: CoT vs 質問文の寄与スコア比較
        # ========================================
        if result.cot_relevance_analysis:
            cot_analysis_1_data = {
                "metadata": common_metadata,
                "description": "分析1: CoT内トークン vs 質問文内トークンの回答への寄与スコア比較",
                "metrics_description": {
                    "prompt_mean/max/sum": "プロンプト（質問文）部分の寄与スコア統計量",
                    "cot_mean/max/sum": "CoT部分の寄与スコア統計量",
                    "cot_prompt_mean_ratio": "CoT平均寄与度 / プロンプト平均寄与度",
                    "top_k_cot_ratio": "寄与スコア上位k件中のCoTトークン割合",
                },
                "summary": result.cot_relevance_analysis.get("summary", {}),
                "per_sample": result.cot_relevance_analysis.get("per_sample", []),
            }
            cot_analysis_1_path = output_subdir / "cot_analysis_1.json"
            self._save_json(cot_analysis_1_data, cot_analysis_1_path)
            saved_files["cot_analysis_1"] = cot_analysis_1_path

        # ========================================
        # 分析2: 回答寄与トークンの共通/CoT固有分類
        # ========================================
        if result.cot_token_classification:
            cot_analysis_2_data = {
                "metadata": common_metadata,
                "description": "分析2: CoT内の回答寄与トークン(上位k件)を共通/CoT固有に分類",
                "metrics_description": {
                    "common_count/ratio": "質問文にも出現する回答寄与トークンの数/割合",
                    "unique_count/ratio": "CoTにのみ出現する回答寄与トークンの数/割合",
                    "normalization": "先頭特殊文字除去(▁, Ġ) + 小文字化",
                },
                "summary": result.cot_token_classification.get("summary", {}),
                "examples": result.cot_token_classification.get("examples", []),
                "per_sample": result.cot_token_classification.get("per_sample", []),
            }
            cot_analysis_2_path = output_subdir / "cot_analysis_2.json"
            self._save_json(cot_analysis_2_data, cot_analysis_2_path)
            saved_files["cot_analysis_2"] = cot_analysis_2_path

        # ========================================
        # 分析3: 層別Jaccard@kと回答変化の相関分析
        # ========================================
        if result.cot_stratified_jaccard_analysis:
            cot_analysis_3_data = {
                "metadata": common_metadata,
                "description": "分析3: 共通/CoT固有トークンそれぞれのJaccard@kと回答変化の相関",
                "metrics_description": {
                    "jaccard_common": "質問文にも出現するトークン集合の摂動前後Jaccard",
                    "jaccard_unique": "CoTにのみ出現するトークン集合の摂動前後Jaccard",
                    "jaccard_all": "全体（共通+固有）のJaccard（参考値）",
                    "correlations": "Spearman相関（vs answer_changed, vs correctness）",
                    "partial_correlations": "偏相関（jaccard_common|unique, unique|common）",
                },
                "summary": result.cot_stratified_jaccard_analysis.get("summary", {}),
                "correlations": result.cot_stratified_jaccard_analysis.get("correlations", {}),
                "partial_correlations": result.cot_stratified_jaccard_analysis.get(
                    "partial_correlations", {}
                ),
                "per_sample": result.cot_stratified_jaccard_analysis.get("per_sample", []),
            }
            cot_analysis_3_path = output_subdir / "cot_analysis_3.json"
            self._save_json(cot_analysis_3_data, cot_analysis_3_path)
            saved_files["cot_analysis_3"] = cot_analysis_3_path

        # ========================================
        # 全データ（デバッグ/詳細分析用）
        # ========================================
        full_data = {
            "metadata": common_metadata,
            "pattern_counts": result.pattern_counts,
            "answer_change": {
                "changed": result.answer_changed_count,
                "unchanged": result.answer_unchanged_count,
            },
            "example_samples": result.example_samples,
            "overall_metrics": result.overall_metrics,
            "pattern_metrics": result.pattern_metrics,
            "answer_change_metrics": result.answer_change_metrics,
            "correlations": [
                {
                    "variable1": c.variable1,
                    "variable2": c.variable2,
                    "group_name": c.group_name,
                    "n": c.n,
                    "spearman_rho": c.spearman_rho,
                    "spearman_p": c.spearman_p,
                }
                for c in result.correlation_results
            ],
            "partial_correlations": [
                {
                    "variable": pc.variable,
                    "control_variable": pc.control_variable,
                    "target_variable": pc.target_variable,
                    "n": pc.n,
                    "partial_r": pc.partial_r,
                    "partial_p": pc.partial_p,
                    "zero_order_r": pc.zero_order_r,
                    "zero_order_p": pc.zero_order_p,
                }
                for pc in result.partial_correlation_results
            ],
            "statistical_tests": [
                {
                    "metric_name": t.metric_name,
                    "group1_name": t.group1_name,
                    "group2_name": t.group2_name,
                    "group1_n": t.group1_n,
                    "group2_n": t.group2_n,
                    "group1_mean": t.group1_mean,
                    "group1_std": t.group1_std,
                    "group2_mean": t.group2_mean,
                    "group2_std": t.group2_std,
                    "delta": t.delta,
                    "mann_whitney_u": t.mann_whitney_u,
                    "mann_whitney_p": t.mann_whitney_p,
                    "cohens_d": t.cohens_d,
                    "significance": t.significance,
                }
                for t in result.statistical_tests
            ],
            "cot_relevance_analysis": result.cot_relevance_analysis.get("summary", {}),
            "cot_token_classification": result.cot_token_classification.get("summary", {}),
            "cot_stratified_jaccard": {
                "summary": result.cot_stratified_jaccard_analysis.get("summary", {}),
                "correlations": result.cot_stratified_jaccard_analysis.get("correlations", {}),
                "partial_correlations": result.cot_stratified_jaccard_analysis.get(
                    "partial_correlations", {}
                ),
            },
            "sample_results": [
                {
                    "sample_id": sr.sample_id,
                    "pattern": sr.pattern,
                    "answer_changed": sr.answer_changed,
                    "before_correct": sr.before_correct,
                    "after_correct": sr.after_correct,
                    "token_count": {
                        "before": sr.question_token_count_before,
                        "after": sr.question_token_count_after,
                        "diff": sr.question_token_count_diff,
                    },
                    "question_metrics": {
                        "spearman_r": sr.question_spearman_r,
                        "jaccard": sr.question_jaccard,
                    },
                    "cot_metrics": {
                        "rouge_l": sr.cot_rouge_l,
                        "jaccard": sr.cot_jaccard,
                    },
                }
                for sr in result.sample_results
            ],
        }
        full_path = output_subdir / "full_results.json"
        self._save_json(full_data, full_path)
        saved_files["full_results"] = full_path

        logger.info(f"分析結果を {output_subdir} に保存:")
        for path in saved_files.values():
            logger.info(f"  - {path.name}")

        return saved_files

    def _save_json(self, data: dict, path: Path) -> None:
        """JSONファイルを保存する."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def compute_unified_exclusion(
    before_dir: str | Path,
    after_dirs: list[str | Path],
    dataset: str,
) -> set[str]:
    """(model, bench) 単位で全摂動条件にわたる union 除外集合を計算する.

    1 つの before_dir と複数の after_dir (k=1,2,4,8 importance + k=4 random +
    k=4 bottom_k 等) を受け取り、いずれかの before/after で
    `extractor.extract_strict()` が空文字を返すサンプル ID の和集合を返す.

    これにより、その (model, bench) の全摂動条件で同一の集計対象サンプル集合
    が使われ、各 (k, type) の指標を直接比較可能になる.

    Args:
        before_dir: 摂動前 (baseline) の結果ディレクトリ
        after_dirs: 摂動後 (perturbed) の結果ディレクトリのリスト
        dataset: ベンチマーク名 (extractor 選択用)

    Returns:
        集計対象から除外すべき sample_id の集合
    """
    from typo_cot.evaluation.extractor import create_extractor

    extractor = create_extractor(dataset)
    excluded: set[str] = set()

    # before_dir の results.json を読み、strict 未検出のサンプルを除外候補に
    before_results_path = Path(before_dir) / "results.json"
    if not before_results_path.exists():
        logger.warning(f"before_dir の results.json が見つかりません: {before_results_path}")
        return excluded
    with before_results_path.open(encoding="utf-8") as f:
        before_results = json.load(f)
    # results.json は list 形式: [{sample_id, generated_text, ...}, ...]
    before_strict_failed: set[str] = set()
    before_text_by_id: dict[str, str] = {}
    for sr in before_results:
        sid = sr.get("sample_id")
        if not sid:
            continue
        text = sr.get("generated_text", "") or ""
        before_text_by_id[sid] = text
        if not extractor.extract_strict(text).strip():
            before_strict_failed.add(sid)

    # 各 after_dir についても strict チェック
    for after_dir in after_dirs:
        after_results_path = Path(after_dir) / "results.json"
        if not after_results_path.exists():
            logger.warning(f"after_dir の results.json が見つかりません: {after_results_path}")
            continue
        with after_results_path.open(encoding="utf-8") as f:
            after_results = json.load(f)
        for sr in after_results:
            sid = sr.get("sample_id")
            if not sid:
                continue
            text = sr.get("generated_text", "") or ""
            after_strict_ok = bool(extractor.extract_strict(text).strip())
            before_strict_ok = sid not in before_strict_failed
            # この (sample_id, after_dir) で before/after どちらかが strict 失敗なら除外
            if not before_strict_ok or not after_strict_ok:
                excluded.add(sid)

    return excluded


def run_analysis(
    before_dir: str | Path,
    after_dir: str | Path,
    output_dir: str | Path,
    excluded_sample_ids: set[str] | None = None,
) -> AnalysisResult:
    """分析を実行する便利関数.

    Args:
        before_dir: 摂動前の結果ディレクトリ
        after_dir: 摂動後の結果ディレクトリ
        output_dir: 出力ディレクトリ（実験シートごとに複数ファイルを出力）
        excluded_sample_ids: 集計対象から除外する sample_id の集合（union 除外用）.
            None の場合は per-pair の strict チェックで除外判定する.

    Returns:
        分析結果
    """
    analyzer = PerturbationAnalyzer(
        before_dir=Path(before_dir),
        after_dir=Path(after_dir),
    )
    result = analyzer.analyze(excluded_sample_ids=excluded_sample_ids)
    analyzer.save_results(result, Path(output_dir))
    return result
