"""Step 0 統合テーブル (master table) のスキーマ定義と io 層.

1 行 = 1 サンプル × 1 モデル × 1 ベンチマーク × 1 条件。
parquet は `data/{model}/{benchmark}/{condition}.parquet` に保存する。
読み書きは本モジュールに集約し、他モジュールは pandas DataFrame とだけやり取りする。

- 条件は `CONDITIONS` に凍結 (clean, lxt1, lxt2, lxt4, lxt8, random4)
- 指標の本文/付録ラベル (修正C) は `METRIC_SCOPE` に凍結
- アーカイブのディレクトリ suffix との対応は `CONDITION_TO_ARCHIVE_SUFFIX`
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 生成条件 (凍結). lxt{k} = AttnLRP importance top-k 摂動, random4 = ランダム4語摂動.
CONDITIONS: tuple[str, ...] = ("clean", "lxt1", "lxt2", "lxt4", "lxt8", "random4")

# アーカイブ outputs/perturbed/{model}_{bench}_{suffix} / outputs/analysis/.../{suffix} との対応.
CONDITION_TO_ARCHIVE_SUFFIX: dict[str, str] = {
    "lxt1": "k1_importance",
    "lxt2": "k2_importance",
    "lxt4": "k4_importance",
    "lxt8": "k8_importance",
    "random4": "k4_random",
}

# 統合テーブルの列 (列名 -> pandas dtype).
# object 列は文字列 (nullable), Float64/boolean/Int32 は pandas nullable dtype.
MASTER_COLUMNS: dict[str, str] = {
    "sample_id": "object",
    "model": "object",
    "benchmark": "object",
    "condition": "object",
    "question_text": "object",
    "cot_text": "object",
    "answer_span": "object",
    "answer_pred": "object",
    "answer_gold": "object",
    "is_correct": "boolean",
    "flip": "boolean",
    "pattern": "object",
    "cot_rouge_l_f1": "Float64",
    "cot_jaccard_top3": "Float64",
    "cot_jaccard_top5": "Float64",
    "cot_jaccard_top10": "Float64",
    "cot_jaccard_top15": "Float64",
    "cot_jaccard_top20": "Float64",
    "r_q": "object",
    "r_c": "object",
    "span_extract_ok": "boolean",
    "seed": "Int32",
    "prompt_id": "object",
    "subset": "object",
    "original_question": "object",
    "perturbed_tokens": "object",
    "source_path": "object",
}

# 修正C: 本文 (main) は ROUGE-L・Jaccard@10・flip のみ。他は付録 (appendix)。
METRIC_SCOPE: dict[str, str] = {
    "cot_rouge_l_f1": "main",
    "cot_jaccard_top10": "main",
    "flip": "main",
    "cot_jaccard_top3": "appendix",
    "cot_jaccard_top5": "appendix",
    "cot_jaccard_top15": "appendix",
    "cot_jaccard_top20": "appendix",
}


def empty_master_df() -> pd.DataFrame:
    """スキーマ通りの空 DataFrame を返す."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in MASTER_COLUMNS.items()}
    )


def coerce_master_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """MASTER_COLUMNS の dtype に揃える (列順も揃える)."""
    out = df.copy()
    for col, dtype in MASTER_COLUMNS.items():
        if col not in out.columns:
            raise ValueError(f"missing column: {col}")
        if dtype != "object":
            out[col] = out[col].astype(dtype)
    return out[list(MASTER_COLUMNS)]


def validate_master_df(df: pd.DataFrame) -> None:
    """スキーマ検証. 不正な場合 ValueError.

    - 列の欠落・過剰
    - condition が CONDITIONS 外
    - (sample_id, model, benchmark, condition) キーの重複
    """
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in MASTER_COLUMNS]
    if extra:
        raise ValueError(f"unexpected columns: {extra}")
    if len(df) == 0:
        return
    bad_conditions = sorted(set(df["condition"]) - set(CONDITIONS))
    if bad_conditions:
        raise ValueError(f"unknown condition values: {bad_conditions}")
    key = ["sample_id", "model", "benchmark", "condition"]
    dup = df.duplicated(subset=key)
    if dup.any():
        dup_keys = df.loc[dup, key].to_dict("records")[:5]
        raise ValueError(f"duplicate keys (first 5): {dup_keys}")


def master_parquet_path(
    root: Path | str, model: str, benchmark: str, condition: str
) -> Path:
    """`{root}/{model}/{benchmark}/{condition}.parquet` を返す."""
    return Path(root) / model / benchmark / f"{condition}.parquet"


def write_condition_parquet(df: pd.DataFrame, root: Path | str) -> Path:
    """単一 (model, benchmark, condition) の DataFrame を parquet に保存する.

    Returns:
        書き込んだ parquet のパス
    """
    validate_master_df(df)
    if len(df) == 0:
        raise ValueError("empty DataFrame cannot be written")
    for col in ("model", "benchmark", "condition"):
        values = set(df[col])
        if len(values) != 1:
            raise ValueError(
                f"DataFrame must contain a single {col} (got {sorted(values)})"
            )
    df = coerce_master_dtypes(df)
    model = df.iloc[0]["model"]
    benchmark = df.iloc[0]["benchmark"]
    condition = df.iloc[0]["condition"]
    path = master_parquet_path(root, model, benchmark, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"wrote {len(df)} rows -> {path}")
    return path


def read_master_table(
    root: Path | str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    """`{root}/{model}/{benchmark}/{condition}.parquet` を横断して読み込む.

    Args:
        root: 統合テーブルのルートディレクトリ
        models / benchmarks / conditions: 指定時はその集合に限定 (None = 全て)

    Returns:
        連結した DataFrame (該当なしの場合は空スキーマ)
    """
    root = Path(root)
    frames: list[pd.DataFrame] = []
    if not root.is_dir():
        return empty_master_df()
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if models is not None and model_dir.name not in models:
            continue
        for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if benchmarks is not None and bench_dir.name not in benchmarks:
                continue
            for pq in sorted(bench_dir.glob("*.parquet")):
                condition = pq.stem
                if conditions is not None and condition not in conditions:
                    continue
                frames.append(pd.read_parquet(pq))
    if not frames:
        return empty_master_df()
    df = pd.concat(frames, ignore_index=True)
    return coerce_master_dtypes(df)
