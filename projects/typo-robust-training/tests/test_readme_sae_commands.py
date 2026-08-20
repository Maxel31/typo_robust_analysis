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
        assert text.count('--training-data "${SUPPLEMENT_DATA}"') == 2
        assert '--validation-data "${SUPPLEMENT_DATA}"' in text


def test_sae_gpu_zero_pin_overrides_the_prior_environment_assignment() -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        blocks = _bash_blocks(text)
        environment_block = next(
            block for block in blocks if "TRAIN_PROJECT=" in block and "GPU_SELECT=" in block
        )
        sae_assignment_block = next(block for block in blocks if "SAE_ROOT:?" in block)
        environment_assignments = [
            line for line in environment_block.splitlines() if re.match(r"^[A-Z_]+=", line)
        ]
        sae_gpu_assignment = next(
            line for line in sae_assignment_block.splitlines() if line.startswith("GPU_ID=")
        )
        script = "\n".join(
            environment_assignments + [sae_gpu_assignment, 'printf "RESOLVED=%s\\n" "${GPU_ID}"']
        )

        result = subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "RESOLVED=0", (
            f"{filename}: SAE GPU pin resolved incorrectly after the environment block: "
            f"{result.stdout.strip()!r}"
        )


def test_every_sae_gpu_reference_uses_the_gpu_zero_pin() -> None:
    for filename in README_NAMES:
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        sae_section = text.split("## 7.", maxsplit=1)[1]
        cuda_references = re.findall(r"CUDA_VISIBLE_DEVICES=(\"[^\"]*\"|\S+)", sae_section)
        gpu_id_references = re.findall(r"--gpu-id\s+(\"[^\"]*\"|\S+)", sae_section)

        assert cuda_references
        assert gpu_id_references
        assert set(cuda_references) == {'"${GPU_ID}"'}
        assert set(gpu_id_references) == {'"${GPU_ID}"'}
