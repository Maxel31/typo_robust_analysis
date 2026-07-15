"""intervention.archive_loader のテスト.

アーカイブの baseline/perturbed results.json (スキーマは実データで確認済み)
から PairRecord を構築する薄いデータアクセス層。GPU 不要・合成 fixture のみで
検証する。後で Step 0 の master table に 1 行で差し替えられるよう、この層に
アーカイブ依存を閉じ込める。
"""

import json
from pathlib import Path

import pytest

from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.records import PairRecord


@pytest.fixture
def archive_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """アーカイブ schema を模した合成 baseline/perturbed ディレクトリ."""
    baseline = [
        {
            "sample_id": "gsm8k_00000",
            "question": "Janet has 16 eggs. How many dollars?",
            "correct_answer": "18",
            "choices": None,
            "context": None,
            "generated_text": "\n9 * 2 = 18.\nThe answer is 18.\n",
            "extracted_answer": "18",
            "is_correct": True,
            "subset": "default",
            "question_top_k_words": [{"word": "Janet", "score": 7.3}],
            "cot_top_k_words": [{"word": "18", "score": 2.0}],
        },
        {
            "sample_id": "gsm8k_00001",
            "question": "Second question?",
            "correct_answer": "5",
            "choices": None,
            "context": None,
            "generated_text": "2+3=5.\nThe answer is 5.\n",
            "extracted_answer": "5",
            "is_correct": False,
            "subset": "default",
            "question_top_k_words": [],
            "cot_top_k_words": [],
        },
        {
            "sample_id": "gsm8k_00002",
            "question": "Only in baseline?",
            "correct_answer": "1",
            "choices": None,
            "context": None,
            "generated_text": "The answer is 1.",
            "extracted_answer": "1",
            "is_correct": True,
            "subset": "default",
            "question_top_k_words": [],
            "cot_top_k_words": [],
        },
    ]
    perturbed = [
        {
            "sample_id": "gsm8k_00000",
            "question": "Janeet has 16 egs. How many dollars?",
            "correct_answer": "18",
            "choices": None,
            "context": None,
            "generated_text": " 9 * 2 = 17.\nThe answer is 17.\n",
            "extracted_answer": "17",
            "is_correct": False,
            "subset": "default",
            "question_top_k_words": [{"word": "Janeet", "score": 5.0}],
            "cot_top_k_words": [{"word": "17", "score": 1.0}],
            "original_question": "Janet has 16 eggs. How many dollars?",
            "perturbed_tokens": [
                {
                    "token_index": 838,
                    "original_token": " Janet",
                    "perturbed_token": "Janeet",
                    "importance_score": 7.3,
                    "perturbation_type": "double_typing",
                }
            ],
        },
        {
            "sample_id": "gsm8k_00001",
            "question": "Secnod question?",
            "correct_answer": "5",
            "choices": None,
            "context": None,
            "generated_text": "2+3=5.\nThe answer is 5.\n",
            "extracted_answer": "5",
            "is_correct": False,
            "subset": "default",
            "question_top_k_words": [],
            "cot_top_k_words": [],
            "original_question": "Second question?",
            "perturbed_tokens": [],
        },
    ]

    baseline_dir = tmp_path / "baseline" / "gemma-3-4b-it_gsm8k"
    perturbed_dir = tmp_path / "perturbed" / "gemma-3-4b-it_gsm8k_k4_importance"
    baseline_dir.mkdir(parents=True)
    perturbed_dir.mkdir(parents=True)
    (baseline_dir / "results.json").write_text(json.dumps(baseline), encoding="utf-8")
    (perturbed_dir / "results.json").write_text(json.dumps(perturbed), encoding="utf-8")
    (baseline_dir / "config.json").write_text(
        json.dumps({"model": "google/gemma-3-4b-it", "benchmark": "gsm8k"}), encoding="utf-8"
    )
    (perturbed_dir / "config.json").write_text(
        json.dumps({"model": "google/gemma-3-4b-it", "benchmark": "gsm8k"}), encoding="utf-8"
    )
    return baseline_dir, perturbed_dir


class TestLoadPairRecords:
    def test_joins_on_sample_id(self, archive_dirs):
        baseline_dir, perturbed_dir = archive_dirs
        pairs = load_pair_records(baseline_dir, perturbed_dir)
        # baseline のみに存在する gsm8k_00002 は除外される
        assert [p.sample_id for p in pairs] == ["gsm8k_00000", "gsm8k_00001"]
        assert all(isinstance(p, PairRecord) for p in pairs)

    def test_fields_mapped(self, archive_dirs):
        baseline_dir, perturbed_dir = archive_dirs
        pairs = load_pair_records(baseline_dir, perturbed_dir)
        p = pairs[0]
        assert p.model == "google/gemma-3-4b-it"
        assert p.benchmark == "gsm8k"
        assert p.question_clean == "Janet has 16 eggs. How many dollars?"
        assert p.question_typo == "Janeet has 16 egs. How many dollars?"
        assert p.cot_clean == "\n9 * 2 = 18.\nThe answer is 18.\n"
        assert p.cot_typo == " 9 * 2 = 17.\nThe answer is 17.\n"
        assert p.answer_clean == "18"
        assert p.answer_typo == "17"
        assert p.correct_answer == "18"
        assert p.is_correct_clean is True
        assert p.subset == "default"

    def test_extra_carries_attribution(self, archive_dirs):
        baseline_dir, perturbed_dir = archive_dirs
        pairs = load_pair_records(baseline_dir, perturbed_dir)
        p = pairs[0]
        # 実験3 の precision@10 用に R_Q/R_C (baseline側) を持ち回る
        assert p.extra["rq_top_words"] == [{"word": "Janet", "score": 7.3}]
        assert p.extra["rc_top_words"] == [{"word": "18", "score": 2.0}]
        assert p.extra["perturbed_tokens"][0]["perturbed_token"] == "Janeet"

    def test_clean_correct_only_filter(self, archive_dirs):
        baseline_dir, perturbed_dir = archive_dirs
        pairs = load_pair_records(baseline_dir, perturbed_dir, clean_correct_only=True)
        assert [p.sample_id for p in pairs] == ["gsm8k_00000"]

    def test_limit(self, archive_dirs):
        baseline_dir, perturbed_dir = archive_dirs
        pairs = load_pair_records(baseline_dir, perturbed_dir, limit=1)
        assert len(pairs) == 1
