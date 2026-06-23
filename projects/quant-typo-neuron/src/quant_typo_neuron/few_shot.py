"""few-shot 例のキャッシュ機構。"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from quant_typo_neuron.benchmarks.base import BenchmarkExample


class FewShotCache:
    def __init__(self, cache_dir: Path | str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, benchmark: str, num_shots: int, seed: int) -> Path:
        return self._cache_dir / f"{benchmark}_n{num_shots}_s{seed}.json"

    def get_or_create(
        self,
        benchmark: str,
        num_shots: int,
        seed: int,
        pool: list[BenchmarkExample],
    ) -> list[BenchmarkExample]:
        if num_shots == 0:
            return []

        path = self._cache_path(benchmark, num_shots, seed)
        if path.exists():
            return self._load(path)

        rng = random.Random(seed)
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        selected = indices[: min(num_shots, len(indices))]
        examples = [pool[i] for i in selected]

        self._save(examples, path)
        return examples

    def _save(self, examples: list[BenchmarkExample], path: Path) -> None:
        data = [asdict(ex) for ex in examples]
        path.write_text(json.dumps(data, ensure_ascii=False))

    def _load(self, path: Path) -> list[BenchmarkExample]:
        data = json.loads(path.read_text())
        return [BenchmarkExample(**item) for item in data]
