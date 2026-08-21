import numpy as np
import pytest
from ucc_inj_reproduction.exp6 import (
    Exp6Config,
    cosine_per_layer,
    last_non_padding_index,
    run_exp6,
    stable_example_seed,
    summarise_layer_cosines,
)


class FakeExtractor:
    def states(self, text: str) -> np.ndarray:
        # Depend on text length so noise variants are observably distinct.
        return np.asarray([[1.0, float(len(text))], [2.0, float(len(text) + 1)]])


def test_last_non_padding_index_handles_left_and_right_padding() -> None:
    assert last_non_padding_index([0, 0, 1, 1]) == 3
    assert last_non_padding_index([1, 1, 0, 0]) == 1
    with pytest.raises(ValueError, match="no non-padding"):
        last_non_padding_index([0, 0])


def test_cosine_per_layer_and_zero_norm() -> None:
    result = cosine_per_layer(np.asarray([[1.0, 0.0], [0.0, 0.0]]), np.asarray([[0.0, 1.0], [1.0, 1.0]]))
    assert result[0] == 0.0
    assert np.isnan(result[1])


def test_exp6_records_every_question_and_noise_level() -> None:
    config = Exp6Config(model="unit-test", noise_levels=(1, 3), device="cpu", limit=None)
    records, summary = run_exp6(config, questions=["one", "two"], extractor=FakeExtractor())
    assert [(record["example_index"], record["noise_level"]) for record in records] == [
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 3),
    ]
    assert {(row["noise_level"], row["layer"], row["n"]) for row in summary} == {
        (1, 0, 2),
        (1, 1, 2),
        (3, 0, 2),
        (3, 1, 2),
    }


def test_stable_seed_ignores_python_process_hash_randomisation() -> None:
    assert stable_example_seed(root_seed=42, example_index=1, noise_level=2) == stable_example_seed(
        root_seed=42, example_index=1, noise_level=2
    )
    assert stable_example_seed(root_seed=42, example_index=1, noise_level=2) != stable_example_seed(
        root_seed=42, example_index=1, noise_level=3
    )


def test_summary_excludes_undefined_cosines() -> None:
    summary = summarise_layer_cosines(
        [{"noise_level": 1, "cosine_similarity": [1.0, float("nan")]}, {"noise_level": 1, "cosine_similarity": [0.5, 0.1]}]
    )
    assert summary == [
        {"noise_level": 1, "layer": 0, "n": 2, "mean_cosine": 0.75, "median_cosine": 0.75, "std_cosine": 0.25},
        {"noise_level": 1, "layer": 1, "n": 1, "mean_cosine": 0.1, "median_cosine": 0.1, "std_cosine": 0.0},
    ]
