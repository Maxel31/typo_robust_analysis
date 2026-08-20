"""The SAE workflow is executable from both English and Japanese READMEs."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_NAMES = ("README.md", "README.ja.md")


def _bash_blocks(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL))


def _sae_gpu_blocks(text: str) -> tuple[str, ...]:
    sae_section = text.split("## 7.", maxsplit=1)[1]
    return tuple(block for block in _bash_blocks(sae_section) if "CUDA_VISIBLE_DEVICES=" in block)


def _assert_every_sae_gpu_block_pins_gpu_zero(text: str) -> None:
    gpu_blocks = _sae_gpu_blocks(text)
    assert len(gpu_blocks) == 3
    for block_index, block in enumerate(gpu_blocks):
        assignments = re.findall(r'(?m)^SAE_GPU_ID="([^"]+)"$', block)
        assert assignments == ["0"], f"SAE GPU block {block_index} must assign literal SAE_GPU_ID=0"


def test_readmes_document_separate_wp1_and_wp2_commands_and_gpu_zero() -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "build-sae-clean-corpus" in text
        assert "calibrate-sparse-autoencoder-l1" in text
        assert "train-sparse-autoencoders" in text
        assert "validate-sparse-autoencoders" in text
        assert "GPU 5/6" in text
        assert "--training-budget minimum" in text
        assert ': "${ROOTED_REGISTRY:?Set ROOTED_REGISTRY' in text
        assert text.count('--registry "${ROOTED_REGISTRY}"') == 4
        assert '--registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml"' not in text
        assert text.count('--training-data "${SAE_SUPPLEMENT_DATA}"') == 2
        assert '--validation-data "${SAE_SUPPLEMENT_DATA}"' in text


def test_sae_subshell_pins_gpu_zero_without_clobbering_caller_state(
    tmp_path: Path,
) -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        blocks = _bash_blocks(text)
        environment_block = next(
            block for block in blocks if "TRAIN_PROJECT=" in block and "GPU_SELECT=" in block
        )
        environment_assignments = [
            line for line in environment_block.splitlines() if re.match(r"^[A-Z_]+=", line)
        ]
        sae_section = text.split("## 7.", maxsplit=1)[1]
        sae_blocks = _bash_blocks(sae_section)
        gpu_blocks = _sae_gpu_blocks(text)

        assert len(sae_blocks) == 4
        assert all(block.strip().startswith("(") for block in sae_blocks)
        assert all(block.strip().endswith(")") for block in sae_blocks)
        assert gpu_blocks

        sae_assignments = [
            line for line in gpu_blocks[0].splitlines() if re.match(r"^SAE_[A-Z_]+=", line)
        ]
        script = "\n".join(
            environment_assignments
            + ['EVALUATION_DATA="protected-evaluation"', "("]
            + sae_assignments
            + [
                'printf "INSIDE=%s|%s\\n" "${SAE_GPU_ID}" "${SAE_WANDB_PROJECT}"',
                ")",
                'printf "OUTSIDE=%s|%s|%s\\n" "${GPU_ID}" "${WANDB_PROJECT}" "${EVALUATION_DATA}"',
            ]
        )

        result = subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "INSIDE=0|typo-robustness-sae",
            "OUTSIDE=5|typo-robustness-training|protected-evaluation",
        ]
        assert not re.search(r"(?m)^(GPU_ID|WANDB_PROJECT|EVALUATION_DATA)=", sae_section)

        guard_block = next(block for block in sae_blocks if "build-sae-clean-corpus" in block)
        marker = tmp_path / f"{filename}.uv-invoked"
        failure = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    ['uv() { : > "${UV_MARKER}"; }']
                    + environment_assignments
                    + [
                        "unset SAE_ROOT ROOTED_REGISTRY",
                        guard_block,
                        'printf "CALLER_SURVIVED=%s\\n" "${GPU_ID}"',
                    ]
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "UV_MARKER": str(marker)},
        )
        assert failure.returncode == 0, failure.stderr
        assert failure.stdout.strip() == "CALLER_SURVIVED=5"
        assert not marker.exists()


def test_every_sae_gpu_block_fails_closed_before_invoking_uv(
    tmp_path: Path,
) -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        sae_section = text.split("## 7.", maxsplit=1)[1]
        gpu_blocks = [
            block for block in _bash_blocks(sae_section) if "CUDA_VISIBLE_DEVICES=" in block
        ]
        assert len(gpu_blocks) == 3

        for block_index, block in enumerate(gpu_blocks):
            block_root = tmp_path / f"{filename}-{block_index}-sae"
            block_root.mkdir()
            registry = tmp_path / f"{filename}-{block_index}-registry.yaml"
            registry.write_text("reviewed: true\n", encoding="utf-8")
            training_root = tmp_path / f"{filename}-{block_index}-results"
            training_source = training_root / "data/gemma4b-cycle3-64m/training_sources.jsonl"
            training_source.parent.mkdir(parents=True)
            training_source.write_text('{"record_id":"clean-1"}\n', encoding="utf-8")

            command_specific_root = block_root / "command-specific-missing"
            command_specific_root.mkdir()
            supplement = command_specific_root / "clean-corpus/sae_clean_supplement.jsonl"
            if "calibrate-sparse-autoencoder-l1" not in block:
                supplement.parent.mkdir(parents=True)
                supplement.write_text('{"record_id":"clean-2"}\n', encoding="utf-8")

            scenarios = (
                ("relative", "relative/sae-artifacts", "relative/registry.yaml"),
                (
                    "missing-root",
                    str(tmp_path / f"{filename}-{block_index}-missing-root"),
                    str(registry),
                ),
                (
                    "missing-registry",
                    str(block_root),
                    str(tmp_path / f"{filename}-{block_index}-missing-registry.yaml"),
                ),
                (
                    "missing-command-input",
                    str(command_specific_root),
                    str(registry),
                ),
            )
            for scenario, sae_root, registry in scenarios:
                marker = tmp_path / f"{filename}-{block_index}-{scenario}.uv-invoked"
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "\n".join(('uv() { : > "${UV_MARKER}"; }', block)),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={
                        **os.environ,
                        "TRAIN_PROJECT": str(PROJECT_ROOT),
                        "TRAIN_ROOT": str(training_root),
                        "GPU_ID": "5",
                        "WANDB_PROJECT": "typo-robustness-training",
                        "SAE_ROOT": sae_root,
                        "ROOTED_REGISTRY": registry,
                        "UV_MARKER": str(marker),
                    },
                )

                assert result.returncode != 0, (
                    f"{filename} block {block_index} accepted {scenario} inputs"
                )
                assert not marker.exists(), (
                    f"{filename} block {block_index} invoked uv for {scenario} inputs"
                )


def test_every_sae_gpu_reference_uses_the_gpu_zero_pin() -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        sae_section = text.split("## 7.", maxsplit=1)[1]
        cuda_references = re.findall(r"CUDA_VISIBLE_DEVICES=(\"[^\"]*\"|\S+)", sae_section)
        gpu_id_references = re.findall(r"--gpu-id\s+(\"[^\"]*\"|\S+)", sae_section)
        wandb_references = re.findall(r"--wandb-project\s+(\"[^\"]*\"|\S+)", sae_section)

        assert cuda_references
        assert gpu_id_references
        assert wandb_references
        assert set(cuda_references) == {'"${SAE_GPU_ID}"'}
        assert set(gpu_id_references) == {'"${SAE_GPU_ID}"'}
        assert set(wandb_references) == {'"${SAE_WANDB_PROJECT}"'}
        _assert_every_sae_gpu_block_pins_gpu_zero(text)


def test_gpu_zero_contract_rejects_each_sae_block_pinned_to_a_training_gpu() -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        for block_index, block in enumerate(_sae_gpu_blocks(text)):
            mutated_block = block.replace('SAE_GPU_ID="0"', 'SAE_GPU_ID="5"', 1)
            assert mutated_block != block
            assert re.findall(r'(?m)^SAE_GPU_ID="([^"]+)"$', mutated_block) == ["5"]
            mutated = text.replace(block, mutated_block, 1)

            with pytest.raises(
                AssertionError,
                match=f"SAE GPU block {block_index} must assign literal SAE_GPU_ID=0",
            ):
                _assert_every_sae_gpu_block_pins_gpu_zero(mutated)
