from pathlib import Path

from ucc_inj_reproduction.cli import _load_config, build_parser


def test_load_exp6_config_converts_noise_levels_to_tuple(tmp_path: Path) -> None:
    config_path = tmp_path / "exp6.yaml"
    config_path.write_text("exp6:\n  model: test-model\n  noise_levels: [1, 2, 3]\n", encoding="utf-8")
    config = _load_config(config_path)
    assert config.model == "test-model"
    assert config.noise_levels == (1, 2, 3)


def test_exp6_cli_contract() -> None:
    parsed = build_parser().parse_args(
        ["exp6-cosine", "--config", "config.yaml", "--output-dir", "result", "--limit", "2"]
    )
    assert parsed.command == "exp6-cosine"
    assert parsed.limit == 2
