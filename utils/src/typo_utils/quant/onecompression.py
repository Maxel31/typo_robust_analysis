"""FujitsuResearch/OneCompression ベースの QEP 量子化。"""

from __future__ import annotations

import os
from pathlib import Path

from typo_utils.quant.base import QuantConfig, Quantizer


class QEPQuantizer(Quantizer):
    def quantize(
        self,
        model_name_or_path: str,
        output_dir: str | Path,
        config: QuantConfig,
        *,
        calibration_data: list[str] | None = None,
        gpu_ids: list[int] | None = None,
    ) -> Path:
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

        from onecompression.quantization import qep_quantize
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        qep_quantize(
            model=model,
            tokenizer=tokenizer,
            bits=config.bits,
            group_size=config.group_size,
            calibration_data=calibration_data,
            num_calibration_samples=config.num_calibration_samples,
            output_dir=str(output_dir),
        )
        tokenizer.save_pretrained(output_dir)
        return output_dir
