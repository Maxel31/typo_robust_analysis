"""Phase 4: 分析用メトリクス計算モジュール.

摂動前後のRelevance分布やCoT推論過程の変化を測定するための
各種メトリクスを計算する。
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


def normalize_distribution(scores: Sequence[float]) -> np.ndarray:
    """スコアを確率分布に正規化する.

    Args:
        scores: スコアのリスト（負の値も許容）

    Returns:
        正規化された確率分布（合計が1）
    """
    arr = np.array(scores, dtype=np.float64)

    # 負の値がある場合はシフト
    if arr.min() < 0:
        arr = arr - arr.min()

    # 合計が0の場合は一様分布を返す
    total = arr.sum()
    if total == 0 or np.isnan(total):
        return np.ones_like(arr) / len(arr)

    return arr / total


def shannon_entropy(scores: Sequence[float], normalize: bool = True) -> float:
    """Shannon Entropyを計算する（自然対数使用）.

    Args:
        scores: スコアのリスト
        normalize: Trueの場合、ln(n)で正規化して0-1にスケール

    Returns:
        Shannon Entropy（normalizeがTrueの場合は0-1の値）
    """
    if len(scores) == 0:
        return 0.0

    p = normalize_distribution(scores)

    # エントロピー計算（0の要素は0*log(0)=0として扱う）
    entropy = 0.0
    for pi in p:
        if pi > 0:
            entropy -= pi * math.log(pi)  # 自然対数

    if normalize and len(scores) > 1:
        # ln(n)で正規化
        max_entropy = math.log(len(scores))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return entropy


def js_divergence(scores1: Sequence[float], scores2: Sequence[float]) -> float:
    """Jensen-Shannon Divergenceを計算する.

    Args:
        scores1: 分布1のスコア
        scores2: 分布2のスコア

    Returns:
        JS-Divergence（0-1の値、同じ分布なら0）

    Note:
        2つの分布の長さが異なる場合、短い方を0でパディングする
    """
    # 長さを揃える
    max_len = max(len(scores1), len(scores2))
    if max_len == 0:
        return 0.0

    arr1 = np.zeros(max_len)
    arr2 = np.zeros(max_len)
    arr1[: len(scores1)] = scores1
    arr2[: len(scores2)] = scores2

    p = normalize_distribution(arr1)
    q = normalize_distribution(arr2)

    # 中間分布
    m = (p + q) / 2

    # KLダイバージェンス計算
    def kl_divergence(p_dist: np.ndarray, q_dist: np.ndarray) -> float:
        kl = 0.0
        for pi, qi in zip(p_dist, q_dist, strict=True):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)
        return kl

    js = (kl_divergence(p, m) + kl_divergence(q, m)) / 2

    return js


def top_k_concentration(
    scores: Sequence[float],
    k: int | None = None,
    k_percentage: float | None = None,
) -> float:
    """Top-k Concentration（上位k個のスコア集中度）を計算する.

    Args:
        scores: スコアのリスト
        k: 上位何個を使用するか（固定値）
        k_percentage: トークン長に対する割合（0-1）

    Returns:
        Top-k Concentration（0-1の値、上位kに全て集中で1）

    Note:
        k と k_percentage のどちらか一方を指定する。
        k_percentage を指定した場合、k = ceil(len(scores) * k_percentage) となる。
    """
    if len(scores) == 0:
        return 0.0

    # kを決定
    if k_percentage is not None:
        k = max(1, math.ceil(len(scores) * k_percentage))
    elif k is None:
        raise ValueError("k または k_percentage のどちらかを指定してください")

    k = min(k, len(scores))

    # 正規化
    p = normalize_distribution(scores)

    # 上位kのスコアを合計
    sorted_p = np.sort(p)[::-1]  # 降順ソート
    concentration = float(np.sum(sorted_p[:k]))

    return concentration


def top_k_jaccard(
    scores1: Sequence[float],
    scores2: Sequence[float],
    k: int | None = None,
    k_percentage: float | None = None,
) -> float:
    """Top-K Jaccard係数を計算する.

    Args:
        scores1: 分布1のスコア
        scores2: 分布2のスコア
        k: 上位何個を比較するか（固定値）
        k_percentage: トークン長に対する割合（0-1）

    Returns:
        Jaccard係数（0-1の値、完全一致で1）

    Note:
        k_percentage を指定した場合、各分布ごとに
        k = ceil(len(scores) * k_percentage) を計算する。
    """
    if len(scores1) == 0 or len(scores2) == 0:
        return 0.0

    # 各分布のkを決定
    if k_percentage is not None:
        k1 = max(1, math.ceil(len(scores1) * k_percentage))
        k2 = max(1, math.ceil(len(scores2) * k_percentage))
    elif k is not None:
        k1 = min(k, len(scores1))
        k2 = min(k, len(scores2))
    else:
        raise ValueError("k または k_percentage のどちらかを指定してください")

    # 上位kのインデックスを取得
    indices1 = set(np.argsort(scores1)[-k1:])
    indices2 = set(np.argsort(scores2)[-k2:])

    # Jaccard係数
    intersection = len(indices1 & indices2)
    union = len(indices1 | indices2)

    if union == 0:
        return 0.0

    return intersection / union


def top_k_jaccard_by_token(
    tokens1: Sequence[str],
    scores1: Sequence[float],
    tokens2: Sequence[str],
    scores2: Sequence[float],
    k: int,
) -> float:
    """トークンベースのTop-K Jaccard係数を計算する.

    同じトークンが複数回出現する場合は、最も高いスコアのトークンのみを残して
    重複を排除してから計算を行う。

    Args:
        tokens1: 分布1のトークンリスト
        scores1: 分布1のスコアリスト
        tokens2: 分布2のトークンリスト
        scores2: 分布2のスコアリスト
        k: 上位何個を比較するか

    Returns:
        Jaccard係数（0-1の値、完全一致で1）
    """
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0

    def get_top_k_unique_tokens(tokens: Sequence[str], scores: Sequence[float], k: int) -> set[str]:
        """重複トークンを排除し、Top-kトークン集合を取得する.

        同じトークンが複数回出現する場合は、最も高いスコアのもののみを残す。
        """
        # トークンごとの最大スコアを取得
        token_max_scores: dict[str, float] = {}
        for token, score in zip(tokens, scores, strict=True):
            if token not in token_max_scores or score > token_max_scores[token]:
                token_max_scores[token] = score

        # スコアで降順ソートしてTop-kを取得
        sorted_tokens = sorted(token_max_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(k, len(sorted_tokens))
        return {token for token, _ in sorted_tokens[:top_k]}

    # 各分布からTop-kユニークトークン集合を取得
    top_tokens1 = get_top_k_unique_tokens(tokens1, scores1, k)
    top_tokens2 = get_top_k_unique_tokens(tokens2, scores2, k)

    # Jaccard係数を計算
    intersection = len(top_tokens1 & top_tokens2)
    union = len(top_tokens1 | top_tokens2)

    if union == 0:
        return 0.0

    return intersection / union


def top_k_rbo(
    scores1: Sequence[float],
    scores2: Sequence[float],
    k: int | None = None,
    k_percentage: float | None = None,
    p: float = 0.9,
) -> float:
    """Top-K RBO (Rank-Biased Overlap) を計算する.

    RBOはランキングの類似度を測定し、上位の要素により大きな重みを付ける。
    Jaccard係数と異なり、順位を考慮した比較が可能。

    Args:
        scores1: 分布1のスコア
        scores2: 分布2のスコア
        k: 上位何個を比較するか（固定値）
        k_percentage: トークン長に対する割合（0-1）
        p: 持続パラメータ（0-1、大きいほど下位の要素も重視）

    Returns:
        RBOスコア（0-1の値、完全一致で1）

    Note:
        p=0.9の場合、上位10要素で約86%の重みが付く。
        p=0.8の場合、上位10要素で約89%の重みが付く。
    """
    if len(scores1) == 0 or len(scores2) == 0:
        return 0.0

    # 各分布のkを決定
    if k_percentage is not None:
        k1 = max(1, math.ceil(len(scores1) * k_percentage))
        k2 = max(1, math.ceil(len(scores2) * k_percentage))
    elif k is not None:
        k1 = min(k, len(scores1))
        k2 = min(k, len(scores2))
    else:
        raise ValueError("k または k_percentage のどちらかを指定してください")

    # スコアが高い順にインデックスを取得（降順）
    ranked1 = list(np.argsort(scores1)[::-1][:k1])
    ranked2 = list(np.argsort(scores2)[::-1][:k2])

    # 比較する深さの最大値
    max_depth = min(k1, k2)

    if max_depth == 0:
        return 0.0

    # RBOの計算
    # RBO = (1-p) * Σ_{d=1}^{max_depth} p^(d-1) * overlap_at_d / d
    rbo_sum = 0.0
    set1 = set()
    set2 = set()

    for d in range(1, max_depth + 1):
        set1.add(ranked1[d - 1])
        set2.add(ranked2[d - 1])
        overlap = len(set1 & set2)
        agreement = overlap / d
        rbo_sum += (p ** (d - 1)) * agreement

    rbo = (1 - p) * rbo_sum

    # 残りの要素に対する補正項（extrapolation）
    # 最終的な重なりの割合を使って無限級数を近似
    final_overlap = len(set1 & set2) / max_depth
    rbo += (p**max_depth) * final_overlap

    return min(1.0, rbo)  # 数値誤差で1を超える場合を防ぐ


def rouge_l_score(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-Lスコアを計算する.

    Args:
        reference: 参照テキスト
        hypothesis: 仮説テキスト

    Returns:
        precision, recall, f1を含む辞書
    """
    if not reference or not hypothesis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 文字単位でLCSを計算
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    lcs_length = _lcs_length(ref_chars, hyp_chars)

    precision = lcs_length / len(hyp_chars) if hyp_chars else 0.0
    recall = lcs_length / len(ref_chars) if ref_chars else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(seq1: list, seq2: list) -> int:
    """最長共通部分列（LCS）の長さを計算する.

    Args:
        seq1: シーケンス1
        seq2: シーケンス2

    Returns:
        LCSの長さ
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    # メモリ効率のため2行だけ保持
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


# ============================================================
# 統計的検定関数
# ============================================================


def mann_whitney_u_test(
    group1: Sequence[float],
    group2: Sequence[float],
) -> dict[str, float]:
    """Mann-Whitney U検定を実行する.

    Args:
        group1: グループ1のサンプル
        group2: グループ2のサンプル

    Returns:
        statistic, p_value を含む辞書
    """
    if len(group1) < 2 or len(group2) < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}

    try:
        result = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan")}


def cohens_d(
    group1: Sequence[float],
    group2: Sequence[float],
) -> float:
    """Cohen's d（効果量）を計算する.

    Args:
        group1: グループ1のサンプル
        group2: グループ2のサンプル

    Returns:
        Cohen's d（正の値はgroup1 > group2）
    """
    if len(group1) < 2 or len(group2) < 2:
        return float("nan")

    arr1 = np.array(group1, dtype=np.float64)
    arr2 = np.array(group2, dtype=np.float64)

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    n1, n2 = len(arr1), len(arr2)

    # プールされた標準偏差
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return float("nan")

    d = (mean1 - mean2) / pooled_std
    return float(d)


# ============================================================
# 相関分析関数
# ============================================================


def pearson_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Pearson相関係数を計算する.

    Args:
        x: 変数1のサンプル
        y: 変数2のサンプル

    Returns:
        correlation, p_value を含む辞書
    """
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return {"correlation": float("nan"), "p_value": float("nan")}

    try:
        result = stats.pearsonr(x, y)
        return {"correlation": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception:
        return {"correlation": float("nan"), "p_value": float("nan")}


def spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Spearman順位相関係数を計算する.

    Args:
        x: 変数1のサンプル
        y: 変数2のサンプル

    Returns:
        correlation, p_value を含む辞書
    """
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return {"correlation": float("nan"), "p_value": float("nan")}

    try:
        result = stats.spearmanr(x, y)
        return {"correlation": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception:
        return {"correlation": float("nan"), "p_value": float("nan")}
