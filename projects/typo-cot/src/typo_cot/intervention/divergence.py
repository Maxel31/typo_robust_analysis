"""実験3: forced-decoding divergence の計算.

(clean質問, clean CoT) と (typo質問, clean CoTを強制) の 2 forward で、
CoT 各位置の次トークン分布を比較する。logits は保存せず、位置ごとに
KL / log-prob / rank をその場でスカラー化する (チャンク処理)。

すべて GPU 非依存の純関数。モデル forward は runner 側が行い、
ここには logits テンソルとトークン列だけが渡される。
"""

import random as _random
from dataclasses import dataclass

import torch


@dataclass
class AlignedTargets:
    """clean run / typo run の CoT トークン位置対応.

    CoT は同一文字列を teacher-forcing するため、トークン列は質問長差の
    オフセットを除いて 1:1 対応するはず。トークン化の境界効果で一致しない
    サンプルは ok=False でフラグし、divergence 計算から除外する。
    """

    ok: bool
    cot_ids: list[int]
    start_clean: int
    start_typo: int


def align_cot_targets(
    clean_ids: list[int],
    prompt_len_clean: int,
    typo_ids: list[int],
    prompt_len_typo: int,
) -> AlignedTargets:
    """両 run の CoT トークン列を突き合わせる (質問長差のオフセット補正のみ)."""
    cot_clean = list(clean_ids[prompt_len_clean:])
    cot_typo = list(typo_ids[prompt_len_typo:])
    ok = len(cot_clean) == len(cot_typo) and cot_clean == cot_typo
    return AlignedTargets(
        ok=ok,
        cot_ids=cot_clean if ok else [],
        start_clean=prompt_len_clean,
        start_typo=prompt_len_typo,
    )


@dataclass
class DivergenceProfile:
    """位置別 divergence プロファイル (すべて CoT 位置に対して同長のリスト).

    Attributes:
        kl: KL(p_clean ‖ p_typo)
        logp_clean: clean 実トークンの clean run での log-prob
        logp_typo: clean 実トークンの typo run での log-prob
        rank_clean: clean 実トークンの clean run での順位 (1-indexed)
        rank_typo: clean 実トークンの typo run での順位 (1-indexed)
    """

    kl: list[float]
    logp_clean: list[float]
    logp_typo: list[float]
    rank_clean: list[int]
    rank_typo: list[int]


def positionwise_divergence(
    logits_clean: torch.Tensor,
    logits_typo: torch.Tensor,
    target_ids: list[int],
    chunk_size: int = 256,
) -> DivergenceProfile:
    """位置別に KL / log-prob / rank をスカラー化する.

    Args:
        logits_clean: [T, V]。位置 t の行が target_ids[t] を予測する logits
        logits_typo: [T, V]。同上 (typo run 側)
        target_ids: clean 実トークンの ID 列 (長さ T)
        chunk_size: 位置方向のチャンクサイズ (メモリ節約)

    Returns:
        DivergenceProfile
    """
    assert logits_clean.shape == logits_typo.shape
    t_total = logits_clean.shape[0]
    assert len(target_ids) == t_total

    kl_list: list[float] = []
    lpc_list: list[float] = []
    lpt_list: list[float] = []
    rc_list: list[int] = []
    rt_list: list[int] = []

    targets = torch.tensor(target_ids, dtype=torch.long, device=logits_clean.device)

    for start in range(0, t_total, chunk_size):
        end = min(start + chunk_size, t_total)
        lc = logits_clean[start:end].float()
        lt = logits_typo[start:end].float()
        tg = targets[start:end]

        logp_c = torch.log_softmax(lc, dim=-1)
        logp_t = torch.log_softmax(lt, dim=-1)

        kl = (logp_c.exp() * (logp_c - logp_t)).sum(-1)
        lp_c = logp_c.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        lp_t = logp_t.gather(-1, tg.unsqueeze(-1)).squeeze(-1)

        tgt_logit_c = lc.gather(-1, tg.unsqueeze(-1))
        tgt_logit_t = lt.gather(-1, tg.unsqueeze(-1))
        rank_c = (lc > tgt_logit_c).sum(-1) + 1
        rank_t = (lt > tgt_logit_t).sum(-1) + 1

        kl_list.extend(kl.cpu().tolist())
        lpc_list.extend(lp_c.cpu().tolist())
        lpt_list.extend(lp_t.cpu().tolist())
        rc_list.extend(rank_c.cpu().tolist())
        rt_list.extend(rank_t.cpu().tolist())

    return DivergenceProfile(
        kl=kl_list,
        logp_clean=lpc_list,
        logp_typo=lpt_list,
        rank_clean=rc_list,
        rank_typo=rt_list,
    )


def divergence_onset(ranks: list[int], threshold: int = 5) -> int | None:
    """発散オンセット: clean 実トークンの順位が threshold を割る最初の位置.

    Args:
        ranks: typo run での clean 実トークンの順位列 (1-indexed)
        threshold: 順位しきい値 (これを超えたら「割った」とみなす)

    Returns:
        最初の位置 (0-indexed)。存在しなければ None
    """
    for i, r in enumerate(ranks):
        if r > threshold:
            return i
    return None


def _top_k_types(words: list[tuple[str, float]], k: int) -> set[str]:
    """|score| 降順で上位 k 語の語タイプ (小文字化) を返す."""
    ranked = sorted(words, key=lambda w: abs(w[1]), reverse=True)[:k]
    return {w.lower() for w, _ in ranked}


def precision_at_k(
    kl_words: list[tuple[str, float]],
    rc_words: list[tuple[str, float]],
    k: int = 10,
) -> float:
    """KL 上位 k 語タイプのうち R_C 上位 k 語タイプと重なる割合.

    スコアは絶対値でランクする (AttnLRP の R_C は負スコアも重要のため)。
    語タイプは小文字化して比較する。

    Args:
        kl_words: (語, KL値) のリスト
        rc_words: (語, R_C スコア) のリスト
        k: 上位語数

    Returns:
        precision@k (KL 上位語タイプ数で正規化)
    """
    top_kl = _top_k_types(kl_words, k)
    top_rc = _top_k_types(rc_words, k)
    if not top_kl:
        return 0.0
    return len(top_kl & top_rc) / len(top_kl)


@dataclass
class ShuffleNullResult:
    """precision@k のシャッフル帰無検定の結果."""

    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_shuffles: int


def shuffle_null_precision(
    kl_words: list[tuple[str, float]],
    rc_words: list[tuple[str, float]],
    k: int = 10,
    n_shuffles: int = 1000,
    seed: int = 42,
) -> ShuffleNullResult:
    """位置ラベルのシャッフルによる precision@k の帰無分布を計算する.

    KL 値を語 (位置) ラベルに対してシャッフルし、そのたびに precision@k を
    再計算する。p 値は「帰無 ≥ 実測」の割合 (+1 平滑化)。

    Args:
        kl_words: (語, KL値) のリスト
        rc_words: (語, R_C スコア) のリスト
        k: 上位語数
        n_shuffles: シャッフル回数 B
        seed: 乱数シード

    Returns:
        ShuffleNullResult
    """
    observed = precision_at_k(kl_words, rc_words, k)

    words = [w for w, _ in kl_words]
    values = [v for _, v in kl_words]
    rng = _random.Random(seed)

    null_values: list[float] = []
    n_ge = 0
    for _ in range(n_shuffles):
        shuffled = values[:]
        rng.shuffle(shuffled)
        p = precision_at_k(list(zip(words, shuffled, strict=True)), rc_words, k)
        null_values.append(p)
        if p >= observed:
            n_ge += 1

    n = len(null_values)
    mean = sum(null_values) / n if n else float("nan")
    var = sum((v - mean) ** 2 for v in null_values) / n if n else float("nan")
    p_value = (n_ge + 1) / (n + 1) if n else float("nan")

    return ShuffleNullResult(
        observed=observed,
        null_mean=mean,
        null_std=var**0.5,
        p_value=p_value,
        n_shuffles=n,
    )
