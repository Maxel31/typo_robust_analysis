"""アーカイブ (JSAI2026 outputs) からの PairRecord 構築.

薄いデータアクセス層。アーカイブの baseline / perturbed `results.json`
(スキーマ: sample_id / question / correct_answer / choices / context /
generated_text / extracted_answer / is_correct / subset /
question_top_k_words / cot_top_k_words [+ perturbed 側: original_question /
perturbed_tokens]) を sample_id で結合して PairRecord のリストを返す。

Step 0 の master table が完成したら、この関数だけを master table 読み込みに
差し替えれば下流 (cell_builder / runner / analysis) は変更不要。
アーカイブは読み取り専用 — 書き込みは一切行わない。
"""

import json
import logging
from pathlib import Path

from typo_cot.intervention.records import PairRecord

logger = logging.getLogger(__name__)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pair_records(
    baseline_dir: str | Path,
    perturbed_dir: str | Path,
    clean_correct_only: bool = False,
    limit: int | None = None,
    start: int = 0,
) -> list[PairRecord]:
    """baseline / perturbed の results.json を結合して PairRecord を構築する.

    Args:
        baseline_dir: baseline 実行ディレクトリ (results.json / config.json)
        perturbed_dir: 摂動実行ディレクトリ (results.json / config.json)
        clean_correct_only: clean 条件で正解だったサンプルのみに絞る
        limit: start 位置から limit 件のみ返す (スモーク・シャード分割用)
        start: 結合済みペア列の先頭 start 件を読み飛ばす (大設定のシャード分割用。
            start/limit はフィルタ適用後の結合ペア列に対する [start, start+limit) 切り出し)

    Returns:
        sample_id で結合された PairRecord のリスト (両方に存在するもののみ)
    """
    baseline_dir = Path(baseline_dir)
    perturbed_dir = Path(perturbed_dir)

    config = _read_json(baseline_dir / "config.json")
    model = config.get("model", "unknown")
    benchmark = config.get("benchmark", "unknown")

    baseline_records = {r["sample_id"]: r for r in _read_json(baseline_dir / "results.json")}
    perturbed_records = {r["sample_id"]: r for r in _read_json(perturbed_dir / "results.json")}

    pairs: list[PairRecord] = []
    n_seen = 0
    for sample_id, base in baseline_records.items():
        pert = perturbed_records.get(sample_id)
        if pert is None:
            continue
        if clean_correct_only and not base.get("is_correct", False):
            continue

        n_seen += 1
        if n_seen <= start:
            continue

        pairs.append(
            PairRecord(
                sample_id=sample_id,
                model=model,
                benchmark=benchmark,
                question_clean=base["question"],
                question_typo=pert["question"],
                choices_clean=base.get("choices"),
                choices_typo=pert.get("choices"),
                subset=base.get("subset"),
                correct_answer=base["correct_answer"],
                cot_clean=base["generated_text"],
                cot_typo=pert["generated_text"],
                answer_clean=base.get("extracted_answer", ""),
                answer_typo=pert.get("extracted_answer", ""),
                is_correct_clean=bool(base.get("is_correct", False)),
                extra={
                    "rq_top_words": base.get("question_top_k_words", []),
                    "rc_top_words": base.get("cot_top_k_words", []),
                    "perturbed_tokens": pert.get("perturbed_tokens", []),
                    "is_correct_typo": bool(pert.get("is_correct", False)),
                },
            )
        )
        if limit is not None and len(pairs) >= limit:
            break

    logger.info(
        "PairRecord %d 件を構築 (baseline=%d, perturbed=%d, clean_correct_only=%s)",
        len(pairs),
        len(baseline_records),
        len(perturbed_records),
        clean_correct_only,
    )
    return pairs
