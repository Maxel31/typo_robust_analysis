"""repair.archive_access のテスト (実験9: アーカイブ読み出しの薄い隔離層).

小さな合成 fixture (tmp_path) でアーカイブのディレクトリ構造を模す。
後で Step 0 の master table に一行で差し替えられるよう、
データアクセスはこのモジュールに隔離される。
"""

import json
from pathlib import Path

import pytest

from typo_cot.repair.archive_access import RepairInputRecord, load_condition_records


MODEL = "gemma-3-4b-it"
BENCH = "gsm8k"


@pytest.fixture()
def mini_archive(tmp_path: Path) -> Path:
    """2 サンプル分の合成アーカイブを作る."""
    root = tmp_path / "archive"

    # baseline (clean 生成)
    base_dir = root / "outputs" / "baseline" / f"{MODEL}_{BENCH}"
    base_dir.mkdir(parents=True)
    baseline = [
        {"sample_id": "gsm8k_00000", "extracted_answer": "18", "is_correct": True},
        {"sample_id": "gsm8k_00001", "extracted_answer": "7", "is_correct": False},
        {"sample_id": "gsm8k_00002", "extracted_answer": "3", "is_correct": True},
    ]
    (base_dir / "results.json").write_text(json.dumps(baseline))

    # perturbed 生成 (LXT-4 = importance)
    pert_dir = root / "outputs" / "perturbed" / f"{MODEL}_{BENCH}_k4_importance"
    pert_dir.mkdir(parents=True)
    perturbed = [
        {"sample_id": "gsm8k_00000", "extracted_answer": "18", "is_correct": True},
        {"sample_id": "gsm8k_00001", "extracted_answer": "9", "is_correct": False},
        # gsm8k_00002 は摂動側に存在しない (欠損ケース)
    ]
    (pert_dir / "results.json").write_text(json.dumps(perturbed))

    # 摂動データセット
    ds_dir = root / "datasets" / "perturbed" / f"{MODEL}_{BENCH}_k4_with_choices"
    ds_dir.mkdir(parents=True)
    dataset = {
        "metadata": {"perturbation_mode": "importance", "num_perturbations": 4},
        "samples": [
            {
                "sample_id": "gsm8k_00000",
                "original_question": "Janet has five ducks.",
                "perturbed_question": "Janet has five dicks.",
                "perturbed_tokens": [
                    {
                        "token_index": 10,
                        "original_token": " ducks",
                        "perturbed_token": "dicks",
                        "importance_score": 0.6,
                        "perturbation_type": "proximity",
                    }
                ],
                "choices": None,
                "perturbed_choices": None,
                "correct_answer": "18",
                "subset": "default",
            },
            {
                "sample_id": "gsm8k_00001",
                "original_question": "The lay of the land.",
                "perturbed_question": "The ly of the land.",
                "perturbed_tokens": [
                    {
                        "token_index": 3,
                        "original_token": " lay",
                        "perturbed_token": "ly",
                        "importance_score": 0.1,
                        "perturbation_type": "omission",
                    }
                ],
                "choices": None,
                "perturbed_choices": None,
                "correct_answer": "7",
                "subset": "default",
            },
            {
                "sample_id": "gsm8k_00002",
                "original_question": "Three cats sat.",
                "perturbed_question": "Three cats st.",
                "perturbed_tokens": [],
                "choices": None,
                "perturbed_choices": None,
                "correct_answer": "3",
                "subset": "default",
            },
        ],
    }
    (ds_dir / "perturbed_dataset.json").write_text(json.dumps(dataset))
    return root


class TestLoadConditionRecords:
    def test_lxt4_records(self, mini_archive: Path) -> None:
        records = load_condition_records(mini_archive, MODEL, BENCH, "lxt4")
        # 摂動側 results に存在する 2 サンプルのみ
        assert len(records) == 2
        r0 = records[0]
        assert isinstance(r0, RepairInputRecord)
        assert r0.sample_id == "gsm8k_00000"
        assert r0.condition == "lxt4"
        assert r0.clean_answer == "18"
        assert r0.typo_answer == "18"
        assert r0.flip is False
        assert r0.clean_correct is True
        assert r0.perturbed_tokens[0]["original_token"] == " ducks"

        r1 = records[1]
        assert r1.flip is True  # "7" -> "9"
        assert r1.clean_correct is False

    def test_limit(self, mini_archive: Path) -> None:
        records = load_condition_records(mini_archive, MODEL, BENCH, "lxt4", limit=1)
        assert len(records) == 1

    def test_unknown_condition_raises(self, mini_archive: Path) -> None:
        with pytest.raises(ValueError):
            load_condition_records(mini_archive, MODEL, BENCH, "unknown")

    def test_random4_uses_random_dirs(self, mini_archive: Path, tmp_path: Path) -> None:
        # random 条件のディレクトリを複製して確認
        root = mini_archive
        src_out = root / "outputs" / "perturbed" / f"{MODEL}_{BENCH}_k4_importance"
        dst_out = root / "outputs" / "perturbed" / f"{MODEL}_{BENCH}_k4_random"
        dst_out.mkdir(parents=True)
        (dst_out / "results.json").write_text((src_out / "results.json").read_text())
        src_ds = root / "datasets" / "perturbed" / f"{MODEL}_{BENCH}_k4_with_choices"
        dst_ds = root / "datasets" / "perturbed" / f"{MODEL}_{BENCH}_k4_random_with_choices"
        dst_ds.mkdir(parents=True)
        (dst_ds / "perturbed_dataset.json").write_text(
            (src_ds / "perturbed_dataset.json").read_text()
        )
        records = load_condition_records(root, MODEL, BENCH, "random4")
        assert len(records) == 2
        assert records[0].condition == "random4"


class TestOverrideRoots:
    """拡張シャード (Qwen B5 / MATH-500) 用の複数ルート解決.

    exp-10-scope worktree に生成された baseline/perturbed/datasets を、
    アーカイブ (jsai2026_root) より優先して参照する。ファイル単位で
    解決するため、Qwen のように baseline のみアーカイブ・摂動側のみ
    exp-10 という混在構成を単一の呼び出しで扱える。
    """

    def _clone_tree(self, src_root: Path, dst_root: Path, parts: list[str]) -> None:
        src = src_root.joinpath(*parts)
        dst = dst_root.joinpath(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

    def test_file_level_fallback_across_roots(
        self, mini_archive: Path, tmp_path: Path
    ) -> None:
        """baseline はアーカイブ、摂動側+データセットは override 側 (Qwen 構成)."""
        override = tmp_path / "wt10"
        # 摂動側とデータセットを override 側へ移動 (アーカイブ側から削除)
        for parts in (
            ["outputs", "perturbed", f"{MODEL}_{BENCH}_k4_importance", "results.json"],
            [
                "datasets",
                "perturbed",
                f"{MODEL}_{BENCH}_k4_with_choices",
                "perturbed_dataset.json",
            ],
        ):
            self._clone_tree(mini_archive, override, parts)
            mini_archive.joinpath(*parts).unlink()

        records = load_condition_records(
            mini_archive, MODEL, BENCH, "lxt4", override_roots=[override]
        )
        assert len(records) == 2
        assert records[0].sample_id == "gsm8k_00000"

    def test_override_root_takes_precedence(
        self, mini_archive: Path, tmp_path: Path
    ) -> None:
        """両ルートに存在する場合は override 側を採用する.

        (Qwen の k4_with_choices はアーカイブと exp-10 で内容が異なり、
        WT10 摂動生成に使われたのは exp-10 側のため)
        """
        override = tmp_path / "wt10"
        ds_parts = [
            "datasets",
            "perturbed",
            f"{MODEL}_{BENCH}_k4_with_choices",
            "perturbed_dataset.json",
        ]
        self._clone_tree(mini_archive, override, ds_parts)
        # override 側のデータセットは 1 サンプルのみに書き換える
        ds = json.loads(override.joinpath(*ds_parts).read_text())
        ds["samples"] = ds["samples"][:1]
        override.joinpath(*ds_parts).write_text(json.dumps(ds))

        records = load_condition_records(
            mini_archive, MODEL, BENCH, "lxt4", override_roots=[override]
        )
        assert len(records) == 1  # override 側データセット (1 サンプル) が優先

    def test_no_override_keeps_existing_behavior(self, mini_archive: Path) -> None:
        records = load_condition_records(mini_archive, MODEL, BENCH, "lxt4")
        assert len(records) == 2

    def test_missing_everywhere_raises_with_tried_paths(
        self, mini_archive: Path, tmp_path: Path
    ) -> None:
        override = tmp_path / "wt10"
        base = mini_archive / "outputs" / "baseline" / f"{MODEL}_{BENCH}" / "results.json"
        base.unlink()
        with pytest.raises(FileNotFoundError) as exc:
            load_condition_records(
                mini_archive, MODEL, BENCH, "lxt4", override_roots=[override]
            )
        msg = str(exc.value)
        assert str(override) in msg
        assert str(mini_archive) in msg
