"""実験6-(iv) 本番 CLI (scripts/run_loo_scoring.py) の純粋ヘルパーのテスト.

test_exp2_cli.py と同じくスクリプトをパスから import して検証する。
対象:
- select_sample_ids: clean 正解サンプルからの決定論的サンプル選定 (seed 固定)
- load_rc_ranking: R_C ランキングのロード + Mistral 結合不良アーカイブの
  token_scores 再構築フォールバック (full_text 配線)
"""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("run_loo_scoring")


def _entries(n: int, correct_every: int = 2) -> list[dict]:
    """sample_id 順のエントリ列。correct_every 件に1件だけ正解."""
    return [
        {"sample_id": f"s{i:04d}", "is_correct": i % correct_every == 0}
        for i in range(n)
    ]


class TestSelectSampleIds:
    def test_only_correct_samples_are_selected(self, cli):
        ids = cli.select_sample_ids(_entries(100), n=10, seed=42)
        correct = {e["sample_id"] for e in _entries(100) if e["is_correct"]}
        assert set(ids) <= correct
        assert len(ids) == 10

    def test_deterministic_for_same_seed(self, cli):
        a = cli.select_sample_ids(_entries(100), n=10, seed=42)
        b = cli.select_sample_ids(_entries(100), n=10, seed=42)
        assert a == b

    def test_different_seed_changes_selection(self, cli):
        a = cli.select_sample_ids(_entries(200), n=10, seed=42)
        b = cli.select_sample_ids(_entries(200), n=10, seed=43)
        assert a != b

    def test_n_larger_than_pool_returns_all_correct(self, cli):
        ids = cli.select_sample_ids(_entries(20), n=300, seed=42)
        correct = [e["sample_id"] for e in _entries(20) if e["is_correct"]]
        assert ids == correct

    def test_n_none_returns_all_correct(self, cli):
        ids = cli.select_sample_ids(_entries(20), n=None, seed=42)
        correct = [e["sample_id"] for e in _entries(20) if e["is_correct"]]
        assert ids == correct

    def test_returned_in_original_entry_order(self, cli):
        ids = cli.select_sample_ids(_entries(100), n=10, seed=42)
        assert ids == sorted(ids)  # sample_id は 0 詰め連番なので順序=辞書順

    def test_no_duplicates(self, cli):
        ids = cli.select_sample_ids(_entries(100), n=30, seed=42)
        assert len(ids) == len(set(ids))


class TestLoadRcRanking:
    FULL_TEXT = "Q: add 3 and 4.\nA: 3 + 4 = 7. The answer is 7."
    TOKENS = [
        "<s>", "Q", ":", "add", "3", "and", "4", ".", "\n", "A", ":",
        "3", "+", "4", "=", "7", ".", "The", "answer", "is", "7", ".",
    ]

    def _save(self, tmp_path: Path, sid: str, data: dict) -> None:
        scores_dir = tmp_path / "importance_scores"
        scores_dir.mkdir(exist_ok=True)
        torch.save(data, scores_dir / f"{sid}_cot.pt")

    def _degenerate_data(self) -> dict:
        by_index = {11: 0.5, 12: 1.0, 13: 2.0, 14: 3.0, 15: 5.0, 16: 0.25}
        return {
            "word_scores": [
                {
                    "word": self.FULL_TEXT.replace(" ", "").replace("\n", ""),
                    "score": 1.23,
                    "token_indices": list(range(1, len(self.TOKENS))),
                }
            ],
            "token_scores": [
                (t, by_index.get(i, 0.1)) for i, t in enumerate(self.TOKENS)
            ],
            "cot_token_start": 11,
            "cot_token_end": 16,
        }

    def test_missing_pt_returns_none(self, cli, tmp_path):
        ranking, degenerate = cli.load_rc_ranking(tmp_path, "nope")
        assert ranking is None
        assert degenerate is False

    def test_healthy_pt_returns_word_ranking(self, cli, tmp_path):
        self._save(
            tmp_path,
            "sid",
            {
                "word_scores": [
                    {"word": "eggs", "score": 2.0, "token_indices": [10]},
                    {"word": "16", "score": 5.0, "token_indices": [12]},
                ],
                "cot_token_start": 5,
                "cot_token_end": 15,
            },
        )
        ranking, degenerate = cli.load_rc_ranking(
            tmp_path, "sid", full_text="whatever"
        )
        assert degenerate is False
        assert [r["word"] for r in ranking] == ["16", "eggs"]

    def test_degenerate_pt_rebuilds_from_token_scores(self, cli, tmp_path):
        """Mistral アーカイブ不良: full_text 配線で token_scores から再構築."""
        self._save(tmp_path, "sid", self._degenerate_data())
        ranking, degenerate = cli.load_rc_ranking(
            tmp_path, "sid", full_text=self.FULL_TEXT
        )
        assert degenerate is True
        assert [r["word"] for r in ranking] == ["7.", "=", "4", "+", "3"]
