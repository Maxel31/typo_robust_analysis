"""実験9 パイプラインの組み立て層.

- HF_MODEL_NAMES: アーカイブのモデル短縮名 -> HuggingFace モデル名
- build_prompt_pair: clean/typo の完全プロンプト対を既存テンプレートで構築
- build_word_rows: 摂動語スパンごとの分析行 (修復スコア・logit lens・特徴量) を組み立て

GPU forward (extract_span_hiddens) の呼び出しは scripts/exp9/run_inner_repair.py が担い、
本モジュールは純粋なデータ変換に留める (ユニットテスト可能性のため)。
"""

import torch

from typo_cot.models.prompts import create_prompt_template
from typo_cot.repair.archive_access import RepairInputRecord
from typo_cot.repair.lexicon_probe import LogitLens, repair_score
from typo_cot.repair.span_align import AlignedSpan

# アーカイブのディレクトリ名に使われる短縮名 -> HF モデル名 (実験9 対象6モデル)
HF_MODEL_NAMES: dict[str, str] = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def build_prompt_pair(record: RepairInputRecord) -> tuple[str, str]:
    """clean/typo の完全プロンプト対を構築する.

    生成時 (scripts/run_inference.py) と同一のテンプレートを使うため、
    few-shot 文脈は両プロンプトで完全一致し、差分は typo 編集のみになる。

    Returns:
        (clean_prompt, typo_prompt)
    """
    template = create_prompt_template(record.benchmark)

    clean_result = template.generate(
        question=record.original_question,
        choices=record.choices,
        subject=record.subset,
    )
    # 生成時 (scripts/run_inference.py) と同じ扱い:
    # 選択肢つきベンチマークの perturbed_question は選択肢行を埋め込み済み
    # (include_choices=True) で perturbed_choices は None のため、
    # typo 側には clean 選択肢を再付加しない。
    typo_result = template.generate(
        question=record.perturbed_question,
        choices=record.perturbed_choices,
        subject=record.subset,
    )
    return clean_result.get_full_prompt(), typo_result.get_full_prompt()


def _min_rank(ranks: list[int] | None, skip_input_embedding: bool = True) -> int | None:
    if not ranks:
        return None
    sub = ranks[1:] if skip_input_embedding and len(ranks) > 1 else ranks
    return min(sub)


def build_word_rows(
    record: RepairInputRecord,
    spans: list[AlignedSpan],
    cos_curves: torch.Tensor,
    typo_target_ranks: list[list[int] | None],
    clean_self_ranks: list[list[int] | None],
    split_increments: list[int],
    zipf_freqs: list[float],
    lens_top_k: int = 5,
) -> list[dict]:
    """摂動語スパンごとの分析行を組み立てる.

    Args:
        record: 入力レコード (flip・R_Q 等のメタデータ)
        spans: 整列済み摂動語スパン (n_spans 個)
        cos_curves: [L+1, n_spans] の層別 cos
        typo_target_ranks: span ごとの層別 rank
            (typo 側 hidden から clean 語先頭トークンを復号)。整列不能なら None。
        clean_self_ranks: span ごとの層別 rank
            (clean 側 hidden から clean 語自身の先頭トークンを復号; サニティ用)
        split_increments: span ごとの分割数増分
        zipf_freqs: span ごとの clean 語 Zipf 頻度
        lens_top_k: 復号一致層の判定に使う top-k

    Returns:
        1 摂動語 = 1 dict の行リスト。cos 曲線が NaN の語 (整列失敗) は落とす。
    """
    rows: list[dict] = []
    for j, span in enumerate(spans):
        curve = cos_curves[:, j]
        if torch.isnan(curve).any():
            continue
        score, layer = repair_score(curve)
        t_ranks = typo_target_ranks[j]
        c_ranks = clean_self_ranks[j]
        rows.append(
            {
                "sample_id": record.sample_id,
                "model": record.model,
                "benchmark": record.benchmark,
                "condition": record.condition,
                "subset": record.subset,
                "clean_word": span.clean_word,
                "typo_word": span.typo_word,
                "perturbation_type": span.perturbation_type,
                "flip": record.flip,
                "clean_correct": record.clean_correct,
                "span_extract_ok": record.span_extract_ok,
                "r_q": span.importance_score,
                "repair_score": score,
                "repair_layer": layer,
                "cos_curve": [round(float(c), 6) for c in curve.tolist()],
                "lens_min_rank": _min_rank(t_ranks),
                "lens_first_hit_layer_top5": (
                    LogitLens.first_hit_layer(t_ranks, top_k=lens_top_k)
                    if t_ranks
                    else None
                ),
                "clean_self_min_rank": _min_rank(c_ranks),
                "clean_self_first_hit_layer_top5": (
                    LogitLens.first_hit_layer(c_ranks, top_k=lens_top_k)
                    if c_ranks
                    else None
                ),
                "split_increment": split_increments[j],
                "zipf_freq": zipf_freqs[j],
            }
        )
    return rows
