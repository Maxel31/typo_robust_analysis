"""llm-compressor ベースの量子化実装 (GPTQ, AWQ, SmoothQuant)。"""

from __future__ import annotations

import os
from pathlib import Path

from typo_utils.quant.base import QuantConfig, Quantizer


class GPTQQuantizer(Quantizer):
    def quantize(
        self,
        model_name_or_path: str,
        output_dir: str | Path,
        config: QuantConfig,
        *,
        calibration_data: list[str] | None = None,
        gpu_ids: list[int] | None = None,
    ) -> Path:
        _set_gpu_ids(gpu_ids)
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        recipe = GPTQModifier(
            targets="Linear",
            scheme=f"W{config.bits}A16",
            ignore=["lm_head"],
        )

        oneshot(
            model=model,
            dataset=config.calibration_dataset,
            recipe=recipe,
            max_seq_length=2048,
            num_calibration_samples=config.num_calibration_samples,
            output_dir=str(output_dir),
        )
        tokenizer.save_pretrained(output_dir)
        return output_dir


class AWQQuantizer(Quantizer):
    def quantize(
        self,
        model_name_or_path: str,
        output_dir: str | Path,
        config: QuantConfig,
        *,
        calibration_data: list[str] | None = None,
        gpu_ids: list[int] | None = None,
    ) -> Path:
        _set_gpu_ids(gpu_ids)
        from llmcompressor.modifiers.quantization import AWQModifier
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        recipe = AWQModifier(
            targets="Linear",
            scheme=f"W{config.bits}A16",
            ignore=["lm_head"],
        )

        oneshot(
            model=model,
            dataset=config.calibration_dataset,
            recipe=recipe,
            max_seq_length=2048,
            num_calibration_samples=config.num_calibration_samples,
            output_dir=str(output_dir),
        )
        tokenizer.save_pretrained(output_dir)
        return output_dir


class SmoothQuantQuantizer(Quantizer):
    def quantize(
        self,
        model_name_or_path: str,
        output_dir: str | Path,
        config: QuantConfig,
        *,
        calibration_data: list[str] | None = None,
        gpu_ids: list[int] | None = None,
    ) -> Path:
        _set_gpu_ids(gpu_ids)
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if config.bits == 8:
            scheme = "W8A8"
        else:
            scheme = f"W{config.bits}A16"

        recipe = [
            SmoothQuantModifier(smoothing_strength=0.5),
            GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=config.calibration_dataset,
            recipe=recipe,
            max_seq_length=2048,
            num_calibration_samples=config.num_calibration_samples,
            output_dir=str(output_dir),
        )
        tokenizer.save_pretrained(output_dir)
        return output_dir


_QUANTIZERS: dict[str, type[Quantizer]] = {
    "gptq": GPTQQuantizer,
    "awq": AWQQuantizer,
    "smoothquant": SmoothQuantQuantizer,
}


def create_quantizer(method: str) -> Quantizer:
    if method not in _QUANTIZERS:
        raise ValueError(f"Unknown quantization method: {method}. Available: {list(_QUANTIZERS)}")
    return _QUANTIZERS[method]()


def _set_gpu_ids(gpu_ids: list[int] | None) -> None:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
