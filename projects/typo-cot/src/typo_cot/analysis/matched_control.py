"""実験5(双子語統制): 対応のある条件間比較の統計.

LXT-4 と Matched-Rnd-4 の精度低下差を、対応のある McNemar 検定 (exact,
二項両側) とリスク差 + 対応のある Wald 95% CI で評価する。

条件記法: A, B は同一サンプル集合上の 2 条件 (例: A=Matched-Rnd-4,
B=LXT-4) の正解フラグ列。risk_diff = acc_A - acc_B。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import binomtest

Z_95 = 1.959963984540054  # Phi^{-1}(0.975)


def mcnemar_exact_p(n01: int, n10: int) -> float:
    """McNemar 検定の exact p 値 (不一致セルの二項両側検定).

    Args:
        n01: A 正解 / B 不正解 の件数
        n10: A 不正解 / B 正解 の件数

    Returns:
        両側 p 値。不一致が 0 件のときは 1.0。
    """
    n_discordant = n01 + n10
    if n_discordant == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n_discordant, 0.5).pvalue)


def paired_condition_comparison(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> dict:
    """対応のある 2 条件の正解率比較 (McNemar + リスク差 Wald CI).

    Args:
        correct_a: 条件 A の正解フラグ (サンプル順)
        correct_b: 条件 B の正解フラグ (同一サンプル順)

    Returns:
        dict: n / n01 / n10 / acc_a / acc_b / risk_diff / risk_diff_ci95 /
        mcnemar_p。risk_diff = acc_a - acc_b = (n01 - n10) / n。
        CI は対応のある Wald 標準誤差
        se = sqrt(n01 + n10 - (n01 - n10)^2 / n) / n による 95% 区間。

    Raises:
        ValueError: 長さ不一致または空列。
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"条件 A/B の長さが一致しません: {len(correct_a)} != {len(correct_b)}"
        )
    n = len(correct_a)
    if n == 0:
        raise ValueError("空のサンプル列は比較できません")

    n01 = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    n10 = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if not a and b)

    acc_a = sum(bool(a) for a in correct_a) / n
    acc_b = sum(bool(b) for b in correct_b) / n
    risk_diff = (n01 - n10) / n

    se = math.sqrt(max(0.0, n01 + n10 - (n01 - n10) ** 2 / n)) / n

    return {
        "n": n,
        "n01": n01,
        "n10": n10,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "risk_diff": risk_diff,
        "risk_diff_ci95": (risk_diff - Z_95 * se, risk_diff + Z_95 * se),
        "mcnemar_p": mcnemar_exact_p(n01, n10),
    }
