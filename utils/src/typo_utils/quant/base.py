"""量子化インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuantConfig:
    method: str
    bits: int
    group_size: int = 128
    num_calibration_samples: int = 512
    calibration_dataset: str = "wikitext"
    desc_act: bool = True


class Quantizer(ABC):
    @abstractmethod
    def quantize(
        self,
        model_name_or_path: str,
        output_dir: str | Path,
        config: QuantConfig,
        *,
        calibration_data: list[str] | None = None,
        gpu_ids: list[int] | None = None,
    ) -> Path:
        ...
