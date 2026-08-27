from pathlib import Path

import pytest
from ucc_inj_reproduction import cli
from ucc_inj_reproduction.cli import _load_config, build_parser
from ucc_inj_reproduction.exp6 import Exp6Config

def test_load_exp6_config_converts_noise_levels_to_tuple(tmp_path: Path) -> None:
    config_path = tmp_path / "exp6.yaml"
    config_path.write_text(
        "exp6:\n"
        "  protocol_scope: adaptation\n"
        "  model: tests/test-model\n"
        "  noise_levels: [0, 1, 2, 3]\n",
        encoding="utf-8",
    )
    config = _load_config(config_path)
    assert config.model == "tests/test-model"
    assert config.protocol_scope == "adaptation"
    assert config.noise_levels == (0, 1, 2, 3)

def test_load_exp6_config_fails_closed_without_level_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "exp6.yaml"
    config_path.write_text(
        "exp6:\n  model: tests/test-model\n  noise_levels: [1, 2, 3]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="level-0"):
        _load_config(config_path)

@pytest.mark.parametrize("field", ["trust_remote_code", "add_generation_prompt"])
def test_load_exp6_config_rejects_quoted_false_booleans(
    tmp_path: Path,
    field: str,
) -> None:
    config_path = tmp_path / "quoted-false.yaml"
    config_path.write_text(
        "exp6:\n"
        "  model: tests/test-model\n"
        "  noise_levels: [0]\n"
        f'  {field}: "false"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{field} must be a boolean"):
        _load_config(config_path)

def test_load_exp6_config_rejects_enabling_remote_code(tmp_path: Path) -> None:
    config_path = tmp_path / "remote-code.yaml"
    config_path.write_text(
        "exp6:\n"
        "  model: tests/test-model\n"
        "  noise_levels: [0]\n"
        "  trust_remote_code: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain disabled"):
        _load_config(config_path)

def test_exp6_cli_contract() -> None:
    parsed = build_parser().parse_args(
        [
            "exp6-cosine",
            "--config",
            "config.yaml",
            "--output-dir",
            "result",
            "--limit",
            "2",
        ]
    )
    assert parsed.command == "exp6-cosine"
    assert parsed.limit == 2

def test_cli_rejects_existing_output_before_expensive_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    config = Exp6Config(model="tests/unit-test", noise_levels=(0,), device="cpu")
    monkeypatch.setattr(cli, "_load_config", lambda _path: config)
    calls: list[str] = []

    def fake_run_exp6(_config: Exp6Config, **_kwargs: object) -> tuple[list, list, dict]:
        calls.append("gpu-run")
        return [], [], {}

    monkeypatch.setattr(cli, "run_exp6", fake_run_exp6)
    with pytest.raises(FileExistsError, match="already occupied"):
        cli.main(
            [
                "exp6-cosine",
                "--config",
                str(tmp_path / "unused.yaml"),
                "--output-dir",
                str(existing),
            ]
        )
    assert calls == []

def test_cli_rejects_dangling_output_symlink_before_expensive_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "dangling"
    output_dir.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    config = Exp6Config(model="tests/unit-test", noise_levels=(0,), device="cpu")
    monkeypatch.setattr(cli, "_load_config", lambda _path: config)
    calls: list[str] = []

    def fake_run_exp6(_config: Exp6Config, **_kwargs: object) -> tuple[list, list, dict]:
        calls.append("gpu-run")
        return [], [], {}

    monkeypatch.setattr(cli, "run_exp6", fake_run_exp6)
    with pytest.raises(FileExistsError, match="already occupied"):
        cli.main(
            [
                "exp6-cosine",
                "--config",
                str(tmp_path / "unused.yaml"),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert calls == []

def test_cli_rejects_output_below_a_regular_file_before_expensive_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "regular-file"
    parent.write_text("not a directory", encoding="utf-8")
    output_dir = parent / "result"
    config = Exp6Config(model="tests/unit-test", noise_levels=(0,), device="cpu")
    monkeypatch.setattr(cli, "_load_config", lambda _path: config)
    calls: list[str] = []

    def fake_run_exp6(_config: Exp6Config, **_kwargs: object) -> tuple[list, list, dict]:
        calls.append("gpu-run")
        return [], [], {}

    monkeypatch.setattr(cli, "run_exp6", fake_run_exp6)
    with pytest.raises(OSError):
        cli.main(
            [
                "exp6-cosine",
                "--config",
                str(tmp_path / "unused.yaml"),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert calls == []
