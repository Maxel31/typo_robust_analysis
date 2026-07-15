"""実験2 CLI (scripts/exp2/run_target_deletion.py ほか) の純粋ヘルパーのテスト.

test_loo_scorer.py と同じくスクリプトをパスから import して検証する。
シャード (--start/--end)・resume (sample_id ベースの冪等スキップ)・
R_C/LOO ランキングのロード・原子的保存を対象とする。
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "exp2"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("run_target_deletion")


@pytest.fixture(scope="module")
def recovery_cli():
    return _load("run_recovery_curve")


class TestResolveGpuId:
    def test_env_takes_precedence(self, cli):
        assert cli.resolve_gpu_id("0", {"CUDA_VISIBLE_DEVICES": "3"}) == "3"

    def test_cli_when_env_unset(self, cli):
        assert cli.resolve_gpu_id("1", {}) == "1"

    def test_empty_env_falls_back(self, cli):
        assert cli.resolve_gpu_id("0", {"CUDA_VISIBLE_DEVICES": ""}) == "0"


class TestShardEntries:
    ENTRIES = [{"sample_id": f"s{i}"} for i in range(10)]

    def test_start_end(self, cli):
        shard = cli.shard_entries(self.ENTRIES, 2, 5)
        assert [e["sample_id"] for e in shard] == ["s2", "s3", "s4"]

    def test_none_bounds_take_all(self, cli):
        assert cli.shard_entries(self.ENTRIES, None, None) == self.ENTRIES

    def test_end_beyond_length(self, cli):
        assert len(cli.shard_entries(self.ENTRIES, 8, 100)) == 2


class TestResolveArms:
    def test_presets(self, cli):
        assert len(cli.resolve_arms("core")) == 2
        assert len(cli.resolve_arms("smoke")) == 4
        assert len(cli.resolve_arms("full")) == 33

    def test_loo_preset_for_m3b2(self, cli):
        # LOO 腕 (修正B): M3×B2 で top_loo × delete × k∈{1,2,4} を単独実行する用
        arms = cli.resolve_arms("loo")
        assert [(a.target_kind, a.op, a.k) for a in arms] == [
            ("top_loo", "delete", 1),
            ("top_loo", "delete", 2),
            ("top_loo", "delete", 4),
        ]

    def test_unknown_raises(self, cli):
        with pytest.raises(ValueError):
            cli.resolve_arms("everything")


class TestFilterPending:
    def test_done_ids_are_skipped(self, cli):
        entries = [{"sample_id": f"s{i}"} for i in range(4)]
        existing = [{"sample_id": "s1"}, {"sample_id": "s3"}]
        pending = cli.filter_pending(entries, existing)
        assert [e["sample_id"] for e in pending] == ["s0", "s2"]


class TestRankingLoaders:
    def test_rc_from_results_entry(self, cli):
        entry = {"cot_top_k_words": [{"word": "eggs", "score": 1.0}]}
        ranking = cli.load_rc_ranking(Path("/nonexistent"), "sid", entry, "results")
        assert ranking == [{"word": "eggs", "score": 1.0}]

    def test_rc_from_cot_pt(self, cli, tmp_path):
        scores_dir = tmp_path / "importance_scores"
        scores_dir.mkdir()
        torch.save(
            {
                "word_scores": [
                    {"word": "eggs", "score": 1.5, "token_indices": [7]},
                    {"word": "outside", "score": 9.0, "token_indices": [99]},
                ],
                "cot_token_start": 5,
                "cot_token_end": 10,
            },
            scores_dir / "sid_cot.pt",
        )
        ranking = cli.load_rc_ranking(tmp_path, "sid", {}, "cot_pt")
        assert ranking == [{"word": "eggs", "score": 1.5}]

    def test_rc_missing_pt_returns_none(self, cli, tmp_path):
        assert cli.load_rc_ranking(tmp_path, "nope", {}, "cot_pt") is None

    def test_loo_rankings_from_file(self, cli, tmp_path):
        path = tmp_path / "loo_results.json"
        path.write_text(
            json.dumps(
                [{"sample_id": "s0", "loo_word_scores": [{"word": "a", "score": 0.1}]}]
            )
        )
        loo = cli.load_loo_rankings(path)
        assert loo["s0"] == [{"word": "a", "score": 0.1}]


class TestAtomicSave:
    def test_save_and_reload_roundtrip(self, cli, tmp_path):
        records = [{"sample_id": "s0", "arms": {}}]
        cli.save_results_atomic(tmp_path, records)
        loaded = json.loads((tmp_path / "results.json").read_text())
        assert loaded == records
        # 再保存 (追記後) しても壊れない
        cli.save_results_atomic(tmp_path, records + [{"sample_id": "s1", "arms": {}}])
        assert len(json.loads((tmp_path / "results.json").read_text())) == 2


class TestRecoveryAssembleCase:
    def test_case_fields(self, recovery_cli):
        cot = "Janet lay 16 eggs per day. She eats 3 eggs for breakfast."
        recovered = {0: False, 25: False, 50: True, 75: True, 100: True}
        rc = [{"word": "eggs", "score": 1.0}, {"word": "16", "score": 5.0}]
        case = recovery_cli.assemble_case(cot, recovered, rc)
        assert case["interval"] == (25, 50)
        # 最上位 content 語 = eggs (数値 16 は除外)
        assert case["target_word"] == "eggs"
        assert case["target_frac"] == pytest.approx(cot.index("eggs") / len(cot))
        # 帰無分布候補は content 候補の初出位置比
        assert all(0 <= f <= 1 for f in case["candidate_fracs"])
        assert len(case["candidate_fracs"]) >= 3

    def test_no_recovery_gives_none_interval(self, recovery_cli):
        cot = "Janet lay 16 eggs."
        case = recovery_cli.assemble_case(
            cot, {p: False for p in (0, 25, 50, 75, 100)}, [{"word": "eggs", "score": 1.0}]
        )
        assert case["interval"] is None
