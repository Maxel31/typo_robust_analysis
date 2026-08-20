"""The SAE workflow is executable from both English and Japanese READMEs."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readmes_document_separate_wp1_and_wp2_commands_and_gpu_zero() -> None:
    for filename in ("README.md", "README.ja.md"):
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "build-sae-clean-corpus" in text
        assert "calibrate-sparse-autoencoder-l1" in text
        assert "train-sparse-autoencoders" in text
        assert "validate-sparse-autoencoders" in text
        assert 'GPU_ID="0"' in text
        assert "GPU 5/6" in text
        assert "--training-budget minimum" in text
        assert 'ROOTED_REGISTRY="/absolute/path/to/reviewed/' in text
        assert text.count('--registry "${ROOTED_REGISTRY}"') == 4
        assert '--registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml"' not in text
        assert text.count('--training-data "${SUPPLEMENT_DATA}"') == 2
        assert '--validation-data "${SUPPLEMENT_DATA}"' in text
