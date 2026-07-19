"""実験5: target_word_list 引数と Matched-Rnd データセット作成のテスト (GPU 不要)."""

import json
from pathlib import Path

import pytest
import torch

from typo_cot.perturbation.dataset import PerturbedDatasetCreator


@pytest.fixture
def baseline_dir(tmp_path: Path) -> Path:
    """合成ベースラインディレクトリ (GSM8K 形式: 選択肢なし)."""
    config = {"model": "test-org/test-model", "benchmark": "gsm8k"}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    # 質問: "alpha beta gamma delta epsilon zeta"
    #        0-5   6-10  11-16 17-22 23-30   31-35
    question = "alpha beta gamma delta epsilon zeta"
    results = [
        {
            "sample_id": "sample_001",
            "question": question,
            "correct_answer": "42",
            "subset": None,
        }
    ]
    with open(tmp_path / "results.json", "w") as f:
        json.dump(results, f)

    scores_dir = tmp_path / "importance_scores"
    scores_dir.mkdir()
    importance_data = {
        "tokens": ["alpha", " beta", " gamma", " delta", " epsilon", " zeta"],
        "token_scores": [
            ("alpha", 0.10),
            (" beta", 0.90),  # top-2
            (" gamma", 0.05),
            (" delta", 0.80),  # top-2
            (" epsilon", 0.03),
            (" zeta", 0.02),
        ],
        "offset_mapping": [
            (0, 5),
            (5, 10),
            (10, 16),
            (16, 22),
            (22, 30),
            (30, 35),
        ],
        "question_char_start": 0,
        "question_char_end": 35,
    }
    torch.save(importance_data, scores_dir / "sample_001.pt")
    return tmp_path


class TestTargetWordList:
    def test_target_word_list_selects_specified_tokens(self, baseline_dir: Path) -> None:
        """target_word_list で指定したトークンだけが摂動される."""
        creator = PerturbedDatasetCreator(
            baseline_dir=baseline_dir,
            num_perturbations=2,
            seed=42,
            target_word_list={"sample_001": [4, 2]},  # epsilon, gamma
        )
        dataset = creator.create()
        assert len(dataset.samples) == 1
        indices = {pt.token_index for pt in dataset.samples[0].perturbed_tokens}
        assert indices == {4, 2}

    def test_target_word_list_metadata_mode(self, baseline_dir: Path) -> None:
        creator = PerturbedDatasetCreator(
            baseline_dir=baseline_dir,
            num_perturbations=2,
            seed=42,
            target_word_list={"sample_001": [4, 2]},
        )
        dataset = creator.create()
        assert dataset.metadata["perturbation_mode"] == "target_list"

    def test_default_mode_unchanged(self, baseline_dir: Path) -> None:
        """target_word_list なしでは従来どおり重要度 top-k が摂動される."""
        creator = PerturbedDatasetCreator(
            baseline_dir=baseline_dir,
            num_perturbations=2,
            seed=42,
        )
        dataset = creator.create()
        indices = {pt.token_index for pt in dataset.samples[0].perturbed_tokens}
        assert indices == {1, 3}  # beta, delta
        assert dataset.metadata["perturbation_mode"] == "importance"

    def test_missing_sample_falls_back_to_importance(self, baseline_dir: Path) -> None:
        """target_word_list に載っていないサンプルは従来の重要度選択."""
        creator = PerturbedDatasetCreator(
            baseline_dir=baseline_dir,
            num_perturbations=2,
            seed=42,
            target_word_list={"other_sample": [0]},
        )
        dataset = creator.create()
        indices = {pt.token_index for pt in dataset.samples[0].perturbed_tokens}
        assert indices == {1, 3}


class TestMatchedTwinDatasetCreator:
    def _make_creator(self, baseline_dir: Path):
        from typo_cot.perturbation.matched_dataset import MatchedTwinDatasetCreator
        from typo_cot.perturbation.matched_sampler import (
            FeatureExtractor,
            MatchedTwinSampler,
        )

        extractor = FeatureExtractor(
            tokenizer=None,
            zipf_fn=lambda w: 4.0,
            seed=42,
        )
        sampler = MatchedTwinSampler(extractor, num_perturbations=2, seed=42)
        return MatchedTwinDatasetCreator(
            baseline_dir=baseline_dir,
            num_perturbations=2,
            sampler=sampler,
            seed=42,
        )

    def test_twins_exclude_lxt_targets(self, baseline_dir: Path) -> None:
        creator = self._make_creator(baseline_dir)
        dataset = creator.create()
        indices = {pt.token_index for pt in dataset.samples[0].perturbed_tokens}
        # LXT top-2 は beta(1), delta(3)。双子語はそれ以外から選ばれる。
        assert len(indices) == 2
        assert not indices & {1, 3}

    def test_metadata_mode_matched_rnd(self, baseline_dir: Path) -> None:
        creator = self._make_creator(baseline_dir)
        dataset = creator.create()
        assert dataset.metadata["perturbation_mode"] == "matched_rnd"

    def test_match_records_collected(self, baseline_dir: Path) -> None:
        creator = self._make_creator(baseline_dir)
        creator.create()
        assert len(creator.match_records) == 2
        assert all(r.sample_id == "sample_001" for r in creator.match_records)

    def test_output_schema_compatible(self, baseline_dir: Path) -> None:
        """出力が既存 perturbed_dataset.json スキーマと互換."""
        from typo_cot.perturbation.dataset import PerturbedDataset

        creator = self._make_creator(baseline_dir)
        dataset = creator.create()
        out = baseline_dir / "out" / "perturbed_dataset.json"
        dataset.save(out)
        loaded = PerturbedDataset.load(out)
        assert loaded.metadata["perturbation_mode"] == "matched_rnd"
        assert loaded.samples[0].sample_id == "sample_001"
        assert len(loaded.samples[0].perturbed_tokens) == 2
