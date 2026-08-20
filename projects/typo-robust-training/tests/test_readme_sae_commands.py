"""The SAE workflow is executable from both English and Japanese READMEs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_NAMES = ("README.md", "README.ja.md")


def _bash_blocks(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL))


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


def test_sae_subshell_pins_gpu_zero_without_clobbering_caller_state() -> None:
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
        gpu_blocks = [block for block in sae_blocks if "CUDA_VISIBLE_DEVICES=" in block]

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
        failure = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    environment_assignments
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
        )
        assert failure.returncode == 0, failure.stderr
        assert failure.stdout.strip() == "CALLER_SURVIVED=5"


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
