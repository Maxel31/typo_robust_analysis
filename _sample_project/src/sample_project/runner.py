"""再現/提案の両 run.py が共有するパイプライン。

ここでは実モデル無しで動く toy 評価器を例にしている。
実モデルを使う場合は ``typo_utils.models.load_causal_lm`` を呼び出す箇所に差し替える。
"""

from __future__ import annotations

from typo_utils.data.typo import TypoConfig, inject_typos
from typo_utils.eval.metrics import accuracy, relative_robustness, robustness_gap

# toy データ: (入力文, ラベル)。実際は typo_utils.data.loaders で読み込む。
TOY_DATA: list[tuple[str, int]] = [
    ("the service was excellent and friendly", 1),
    ("a wonderful delightful pleasant experience", 1),
    ("terrible awful and very disappointing", 0),
    ("the worst horrible broken product", 0),
]

_POSITIVE = {"excellent", "wonderful", "delightful", "pleasant", "friendly", "great"}
_NEGATIVE = {"terrible", "awful", "disappointing", "worst", "horrible", "broken"}


def toy_classify(text: str) -> int:
    """語彙ベースの極性分類（typo に弱い = 頑健性を観察する題材）。"""
    tokens = set(text.lower().split())
    return int(len(tokens & _POSITIVE) >= len(tokens & _NEGATIVE))


def evaluate(typo: TypoConfig | None = None) -> dict[str, float]:
    """clean / typo 双方で評価し、メトリクスを返す。"""
    texts = [t for t, _ in TOY_DATA]
    golds = [y for _, y in TOY_DATA]

    clean_preds = [toy_classify(t) for t in texts]
    clean_acc = accuracy(clean_preds, golds)

    if typo is None:
        return {"clean_acc": clean_acc}

    typo_preds = [toy_classify(inject_typos(t, typo)) for t in texts]
    typo_acc = accuracy(typo_preds, golds)
    return {
        "clean_acc": clean_acc,
        "typo_acc": typo_acc,
        "robustness_gap": robustness_gap(clean_acc, typo_acc),
        "relative_robustness": relative_robustness(clean_acc, typo_acc),
    }
