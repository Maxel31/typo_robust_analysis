"""typo 注入と評価メトリクスの基本テスト。"""

from typo_utils.data.typo import (
    TypoAnnotation,
    TypoConfig,
    inject_typos,
    inject_typos_by_count,
)
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


# --- count-based typo injection ---


def test_inject_typos_by_count_deterministic():
    text = "the quick brown fox jumps over the lazy dog"
    result1 = inject_typos_by_count(text, num_typos=2, typo_type="swap", seed=42)
    result2 = inject_typos_by_count(text, num_typos=2, typo_type="swap", seed=42)
    assert result1 == result2
    assert result1[0] != text


def test_inject_typos_by_count_returns_annotations():
    text = "the quick brown fox jumps"
    modified, annotations = inject_typos_by_count(text, num_typos=2, typo_type="swap", seed=0)
    assert isinstance(modified, str)
    assert len(annotations) == 2
    for ann in annotations:
        assert isinstance(ann, TypoAnnotation)
        assert ann.original_word != ann.typo_word
        assert ann.typo_type == "swap"


def test_inject_typos_by_count_respects_exclude():
    text = "the quick brown fox jumps"
    modified, annotations = inject_typos_by_count(
        text, num_typos=3, typo_type="swap", seed=0, exclude_indices={0, 1, 2, 3, 4}
    )
    assert modified == text
    assert annotations == []


def test_inject_typos_by_count_random_type():
    text = "the quick brown fox jumps over the lazy dog"
    _, annotations = inject_typos_by_count(text, num_typos=5, typo_type="random", seed=123)
    types_used = {a.typo_type for a in annotations}
    assert len(types_used) >= 2


def test_inject_typos_by_count_replace_type():
    text = "the quick brown fox jumps"
    _, annotations = inject_typos_by_count(text, num_typos=2, typo_type="replace", seed=0)
    for ann in annotations:
        assert ann.typo_type == "replace"


def test_inject_typos_by_count_short_words_skipped():
    text = "I a the"
    modified, annotations = inject_typos_by_count(text, num_typos=3, typo_type="swap", seed=0)
    assert len(annotations) <= 1


def test_inject_typos_by_count_word_indices_consistent_across_types():
    text = "the quick brown fox jumps over the lazy dog"
    seed = 42
    num_typos = 3
    types: list[str] = ["swap", "insert", "delete", "replace", "random"]
    index_sets = []
    for typo_type in types:
        _, annotations = inject_typos_by_count(
            text, num_typos=num_typos, typo_type=typo_type, seed=seed
        )
        index_sets.append({a.word_index for a in annotations})
    for i in range(1, len(index_sets)):
        assert index_sets[0] == index_sets[i], (
            f"typo_type={types[i]}: indices {index_sets[i]} != {types[0]}: {index_sets[0]}"
        )


def test_inject_typos_by_count_monotonic_inclusion():
    text = "the quick brown fox jumps over the lazy dog"
    seed = 42
    prev_indices: set[int] = set()
    for n in range(1, 5):
        _, annotations = inject_typos_by_count(
            text, num_typos=n, typo_type="swap", seed=seed
        )
        current_indices = {a.word_index for a in annotations}
        assert prev_indices <= current_indices, (
            f"num_typos={n}: {prev_indices} is not a subset of {current_indices}"
        )
        prev_indices = current_indices
