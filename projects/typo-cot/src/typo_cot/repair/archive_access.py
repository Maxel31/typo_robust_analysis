"""実験9 のためのアーカイブ読み出し (薄い隔離層).

JSAI2026 アーカイブ (configs/paths.yaml の jsai2026_root) から、
モデル x ベンチマーク x 摂動条件 (lxt4 / random4) の
clean/typo 質問対・摂動トークン・flip 判定を読み出す。

このモジュールは読み取り専用であり、アーカイブには一切書き込まない。

NOTE: Step 0 (master table) が完成したら、load_condition_records() の実装を
master table の読み出しに差し替えるだけで上流を変えずに移行できるよう、
データアクセスはこの 1 関数に隔離してある。
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# 摂動条件 -> (outputs/perturbed のディレクトリ接尾辞, datasets/perturbed の接尾辞)
_CONDITION_DIRS: dict[str, tuple[str, str]] = {
    "lxt4": ("k4_importance", "k4_with_choices"),
    "random4": ("k4_random", "k4_random_with_choices"),
}


@dataclass
class RepairInputRecord:
    """実験9 の 1 サンプル分の入力.

    Attributes:
        sample_id: サンプル ID
        model: モデル短縮名 (アーカイブのディレクトリ名に使う形)
        benchmark: ベンチマーク名
        condition: 摂動条件 ("lxt4" / "random4")
        original_question: clean 質問文
        perturbed_question: typo 質問文
        perturbed_tokens: アーカイブ形式の摂動トークンリスト
        choices: clean 選択肢 (無ければ None)
        perturbed_choices: 摂動後選択肢 (無ければ None)
        subset: サブセット名
        correct_answer: 正解
        clean_answer: clean 生成の抽出答え
        typo_answer: typo 生成の抽出答え
        clean_correct: clean 生成が正解か
        flip: clean と typo の抽出答えが異なるか
        span_extract_ok: 両条件で答えスパン抽出に成功しているか
    """

    sample_id: str
    model: str
    benchmark: str
    condition: str
    original_question: str
    perturbed_question: str
    perturbed_tokens: list[dict] = field(default_factory=list)
    choices: list[str] | None = None
    perturbed_choices: list[str] | None = None
    subset: str | None = None
    correct_answer: str | None = None
    clean_answer: str | None = None
    typo_answer: str | None = None
    clean_correct: bool | None = None
    flip: bool | None = None
    span_extract_ok: bool = True


def _as_bool(value: object) -> bool | None:
    """JSON 中の bool / "True" / "False" を bool に正規化する."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def _load_results_index(path: Path) -> dict[str, dict]:
    """results.json を sample_id -> record の辞書にする."""
    with open(path) as f:
        results = json.load(f)
    return {r["sample_id"]: r for r in results}


def _resolve_input_path(roots: Sequence[Path], relative: Path) -> Path:
    """ルート候補を順に試し、最初に存在するファイルパスを返す.

    拡張シャード (Qwen B5 / MATH-500) では baseline がアーカイブ、
    摂動側が exp-10-scope worktree という混在構成になるため、
    ディレクトリ単位ではなくファイル単位で解決する。

    Raises:
        FileNotFoundError: どのルートにも存在しない場合 (試行パスを列挙)
    """
    tried: list[str] = []
    for root in roots:
        candidate = Path(root) / relative
        if candidate.exists():
            return candidate
        tried.append(str(candidate))
    raise FileNotFoundError("実験9の入力が見つかりません。試行: " + " | ".join(tried))


def load_condition_records(
    archive_root: str | Path,
    model: str,
    benchmark: str,
    condition: str,
    limit: int | None = None,
    override_roots: Sequence[str | Path] = (),
) -> list[RepairInputRecord]:
    """モデル x ベンチマーク x 摂動条件の実験9入力レコードを読み出す.

    Args:
        archive_root: アーカイブルート (jsai2026_root)
        model: モデル短縮名 (例: "gemma-3-4b-it")
        benchmark: ベンチマーク名 (例: "gsm8k")
        condition: "lxt4" (LXT-4=importance) または "random4" (Random-4)
        limit: 先頭から読み出す最大サンプル数 (None なら全件)
        override_roots: アーカイブより優先して探すルート群 (先頭が最優先)。
            拡張シャードでは exp-10-scope worktree のプロジェクトルートを渡す。
            ファイル単位で解決するため、baseline のみアーカイブという
            混在構成も単一呼び出しで扱える。

    Returns:
        baseline / perturbed の両方に生成結果があるサンプルのレコードリスト
        (perturbed_dataset.json の順序)
    """
    if condition not in _CONDITION_DIRS:
        raise ValueError(
            f"不明な摂動条件: {condition}. 利用可能: {sorted(_CONDITION_DIRS)}"
        )
    out_suffix, ds_suffix = _CONDITION_DIRS[condition]
    roots = [Path(r) for r in override_roots] + [Path(archive_root)]

    baseline = _load_results_index(
        _resolve_input_path(
            roots, Path("outputs") / "baseline" / f"{model}_{benchmark}" / "results.json"
        )
    )
    perturbed = _load_results_index(
        _resolve_input_path(
            roots,
            Path("outputs") / "perturbed" / f"{model}_{benchmark}_{out_suffix}" / "results.json",
        )
    )
    with open(
        _resolve_input_path(
            roots,
            Path("datasets")
            / "perturbed"
            / f"{model}_{benchmark}_{ds_suffix}"
            / "perturbed_dataset.json",
        )
    ) as f:
        dataset = json.load(f)

    records: list[RepairInputRecord] = []
    for s in dataset["samples"]:
        sid = s["sample_id"]
        base = baseline.get(sid)
        pert = perturbed.get(sid)
        if base is None or pert is None:
            continue
        clean_answer = base.get("extracted_answer")
        typo_answer = pert.get("extracted_answer")
        span_ok = bool(clean_answer) and bool(typo_answer)
        flip = (clean_answer != typo_answer) if span_ok else None
        records.append(
            RepairInputRecord(
                sample_id=sid,
                model=model,
                benchmark=benchmark,
                condition=condition,
                original_question=s["original_question"],
                perturbed_question=s["perturbed_question"],
                perturbed_tokens=s.get("perturbed_tokens") or [],
                choices=s.get("choices"),
                perturbed_choices=s.get("perturbed_choices"),
                subset=s.get("subset"),
                correct_answer=s.get("correct_answer"),
                clean_answer=clean_answer,
                typo_answer=typo_answer,
                clean_correct=_as_bool(base.get("is_correct")),
                flip=flip,
                span_extract_ok=span_ok,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records
