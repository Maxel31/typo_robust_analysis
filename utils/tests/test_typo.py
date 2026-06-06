"""typo 注入と評価メトリクスの基本テスト。"""

from typo_utils.data.typo import TypoConfig, inject_typos
from typo_utils.eval.metrics import accuracy, relative_robustness, robustness_gap


def test_inject_typos_deterministic():
    text = "the quick brown fox jumps"
    cfg = TypoConfig(rate=1.0, type="swap", seed=0)
    out1 = inject_typos(text, cfg)
    out2 = inject_typos(text, cfg)
    assert out1 == out2  # 同一シードで決定論的
    assert out1 != text  # rate=1.0 なので必ず変化


def test_inject_typos_rate_zero_is_identity():
    text = "no change expected here"
    assert inject_typos(text, TypoConfig(rate=0.0)) == text


def test_accuracy_and_gap():
    assert accuracy([1, 1, 0], [1, 0, 0]) == 2 / 3
    assert robustness_gap(0.9, 0.7) == 0.9 - 0.7
    assert relative_robustness(0.8, 0.4) == 0.5
