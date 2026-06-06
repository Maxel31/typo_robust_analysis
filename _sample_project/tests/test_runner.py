"""sample_project の評価パイプラインのスモークテスト。"""

from sample_project.runner import evaluate

from typo_utils.data.typo import TypoConfig


def test_evaluate_clean_only():
    m = evaluate(None)
    assert "clean_acc" in m
    assert 0.0 <= m["clean_acc"] <= 1.0


def test_evaluate_with_typo_has_gap_keys():
    m = evaluate(TypoConfig(rate=0.5, type="swap", seed=1))
    assert {"clean_acc", "typo_acc", "robustness_gap", "relative_robustness"} <= set(m)
