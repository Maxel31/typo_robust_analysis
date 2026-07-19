"""実験5(双子語統制): Matched-Rnd-4 摂動データセット作成.

LXT-4 の各標的語に対し MatchedTwinSampler で層化マッチングした双子語を選び、
LXT と同一の抽選手続き (per-token seed = hash((seed, sample_id, token))) で
typo を注入する。出力は既存 perturbed_dataset.json と完全互換
(perturbation_mode="matched_rnd")。

選択部以外 (トークン範囲の決定・選択肢の扱い・適用ループ) はすべて
PerturbedDatasetCreator を継承して共有する。
"""

from __future__ import annotations

import logging
from pathlib import Path

from typo_cot.perturbation.dataset import PerturbedDatasetCreator, PerturbedToken
from typo_cot.perturbation.matched_sampler import MatchedTwinSampler, MatchRecord

logger = logging.getLogger(__name__)


class MatchedTwinDatasetCreator(PerturbedDatasetCreator):
    """Matched-Rnd モード: 層化マッチした双子語に摂動を適用する.

    _apply_perturbations の候補順序決定のみを MatchedTwinSampler に差し替え、
    適用ループは基底クラスの _apply_candidate_perturbations を用いる。
    マッチ記録は self.match_records に蓄積される (SMD 表の入力)。
    """

    def __init__(
        self,
        baseline_dir: Path,
        num_perturbations: int,
        sampler: MatchedTwinSampler,
        seed: int = 42,
        include_choices: bool = True,
    ) -> None:
        super().__init__(
            baseline_dir=baseline_dir,
            num_perturbations=num_perturbations,
            seed=seed,
            include_choices=include_choices,
        )
        self.sampler = sampler
        self.match_records: list[MatchRecord] = []

    def _apply_perturbations(
        self,
        question: str,
        question_tokens: list[tuple[int, str, float]],
        question_char_start: int,
        offset_mapping: list[tuple[int, int]],
        sample_id: str,
    ) -> tuple[str, list[PerturbedToken]]:
        if not question_tokens:
            return question, []

        # question はこの時点で摂動対象テキスト全体 (MMLU 系は選択肢込み)
        candidate_tokens, records = self.sampler.select(
            sample_id=sample_id,
            question_tokens=question_tokens,
            question_text=question,
        )
        self.match_records.extend(records)

        return self._apply_candidate_perturbations(
            question=question,
            candidate_tokens=candidate_tokens,
            question_char_start=question_char_start,
            offset_mapping=offset_mapping,
            sample_id=sample_id,
        )

    def create(self):
        dataset = super().create()
        dataset.metadata["perturbation_mode"] = "matched_rnd"
        return dataset
