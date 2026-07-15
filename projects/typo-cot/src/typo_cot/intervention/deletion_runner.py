"""実験2: 腕構築 → CoT 編集 → teacher-forcing 短生成 → flip 判定の実行コア.

モデル依存部分は generate_fn (コンテキストリスト → 継続テキストリスト) として
注入する (実験1 runner と同型)。ユニットテストではモック、GPU 実行時は
ModelWrapper.generate_batch を包んだクロージャを渡す。

- 各腕のコンテキスト = prompt + 編集後 prefix + 答えトリガー (trigger_text)。
  答えスパンのみの短生成で安価 (計画 §2.3 昇格2)。
- flip = 基準腕 (無編集 prefix の再生成) の抽出答えとの不一致。
- 答え抽出は evaluation.extractor の該当ベンチマーク抽出器を全腕で完全同一に適用。

設計メモ: docs/dev_notes_02_target_deletion.md
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.cot_editor import apply_edit
from typo_cot.intervention.loo_scorer import (
    ANSWER_PATTERNS,
    CotSplit,
    split_generated_text,
)
from typo_cot.intervention.target_selector import (
    build_candidates,
    normalize_ranking,
    rng_for_sample,
    select_matched_random,
    select_top,
)

logger = logging.getLogger(__name__)

BASELINE_ARM = "baseline"
GenerateFn = Callable[[list[str]], list[str]]

TARGET_KINDS = ("top_rc", "matched_random", "bottom_rc", "top_loo")


@dataclass(frozen=True)
class ArmSpec:
    """要因計画の1セル (腕) の仕様.

    Attributes:
        name: 腕名 (results.json のキー)
        target_kind: "top_rc" | "matched_random" | "bottom_rc" | "top_loo"
        op: "delete" | "mask" | "replace"
        k: 用量 (標的語タイプ数)
        stratum: "content" (主) | "numeric" (別枠)
    """

    name: str
    target_kind: str
    op: str
    k: int
    stratum: str = "content"


def core_arms() -> list[ArmSpec]:
    """コア対比 (M5×B5 昇格分): top vs 一致ランダム、削除のみ、k=1."""
    return [
        ArmSpec("top_rc_delete_k1", "top_rc", "delete", 1),
        ArmSpec("matched_random_delete_k1", "matched_random", "delete", 1),
    ]


def smoke_arms() -> list[ArmSpec]:
    """GPU スモーク: コア対比 + LOO 腕1セル + 数値層別枠1セル."""
    return core_arms() + [
        ArmSpec("top_loo_delete_k1", "top_loo", "delete", 1),
        ArmSpec("numeric_top_rc_delete_k1", "top_rc", "delete", 1, stratum="numeric"),
    ]


def full_grid_arms() -> list[ArmSpec]:
    """完全グリッド (Gemma-3-4B×B2): 標的3×操作3×用量3 + 数値層 + LOO 腕."""
    arms: list[ArmSpec] = []
    for kind in ("top_rc", "matched_random", "bottom_rc"):
        for op in ("delete", "mask", "replace"):
            for k in (1, 2, 4):
                arms.append(ArmSpec(f"{kind}_{op}_k{k}", kind, op, k))
    for k in (1, 2, 4):
        arms.append(
            ArmSpec(f"numeric_top_rc_delete_k{k}", "top_rc", "delete", k, stratum="numeric")
        )
    for k in (1, 2, 4):
        arms.append(ArmSpec(f"top_loo_delete_k{k}", "top_loo", "delete", k))
    return arms


@dataclass
class ArmPlan:
    """1サンプル×1腕の編集計画."""

    spec: ArmSpec
    target_words: list[str] = field(default_factory=list)
    matched_to: list[str] | None = None
    replacements: dict[str, str] = field(default_factory=dict)
    edited_text: str = ""
    n_spans_edited: int = 0
    skip_reason: str | None = None


@dataclass
class SamplePlan:
    """1サンプルの全腕編集計画 + メタデータ."""

    sample_id: str
    prompt: str
    split: CotSplit | None
    skip_reason: str | None
    residual_answer_in_prefix: bool
    clean_correct: bool
    correct_answer: str
    archive_answer: str
    arm_plans: dict[str, ArmPlan]


def _has_answer_pattern(text: str) -> bool:
    """切断後 prefix に答え句 (断片) が残っているか (残留→主分析から除外)."""
    return any(re.search(pattern, text) for pattern, _ in ANSWER_PATTERNS)


def _plan_arm(
    spec: ArmSpec,
    cot_text: str,
    candidates,
    rc_scores: dict[str, float] | None,
    loo_scores: dict[str, float] | None,
    seed: int,
    sample_id: str,
    replacement_sampler,
) -> ArmPlan:
    plan = ArmPlan(spec=spec)
    # 腕ごとに独立の決定論 RNG (腕構成の変更が他腕の選定に影響しない=冪等)
    rng = rng_for_sample(seed, f"{sample_id}/{spec.name}")

    scores = loo_scores if spec.target_kind == "top_loo" else rc_scores
    if scores is None:
        plan.skip_reason = "missing_ranking"
        return plan

    if spec.target_kind == "matched_random":
        top = select_top(scores, candidates, spec.k, stratum=spec.stratum)
        if len(top) < spec.k:
            plan.skip_reason = "insufficient_candidates"
            return plan
        targets = select_matched_random(top, candidates, rng, stratum=spec.stratum)
        plan.matched_to = top
    else:
        targets = select_top(
            scores,
            candidates,
            spec.k,
            stratum=spec.stratum,
            bottom=spec.target_kind == "bottom_rc",
        )
    if len(targets) < spec.k:
        plan.skip_reason = "insufficient_candidates"
        return plan

    replacement_map: dict[str, str] | None = None
    if spec.op == "replace":
        if replacement_sampler is None:
            plan.skip_reason = "missing_replacement_sampler"
            return plan
        replacement_map = {}
        for word in targets:
            repl = replacement_sampler.sample(word, rng)
            if repl is None:
                plan.skip_reason = "no_replacement"
                return plan
            replacement_map[word] = repl

    edit = apply_edit(cot_text, targets, spec.op, replacement_map)
    plan.target_words = targets
    plan.edited_text = edit.edited_text
    plan.n_spans_edited = edit.n_spans_edited
    plan.replacements = edit.replacements
    if not edit.changed:
        plan.skip_reason = "edit_no_change"
    return plan


def prepare_sample(
    entry: dict,
    prompt: str,
    rc_ranking: list[dict] | None,
    loo_ranking: list[dict] | None,
    arms: list[ArmSpec],
    seed: int,
    replacement_sampler=None,
) -> SamplePlan:
    """1サンプルの全腕について標的選定と編集を行い、生成前の計画を返す.

    Args:
        entry: アーカイブ results.json の1行 (sample_id / generated_text /
            extracted_answer / is_correct / correct_answer)
        prompt: 再構築済みの生成プロンプト (few-shot 込み、既存規約と同一)
        rc_ranking: R_C 語ランキング [{"word","score"}] (fixed-target 版が本番)
        loo_ranking: LOO 語ランキング (top_loo 腕にのみ必要)
        arms: 腕仕様リスト
        seed: グローバル seed (sample_id と併せて決定論 RNG を導出)
        replacement_sampler: replace 操作用サンプラ (sample(word, rng) -> str|None)
    """
    sample_id = entry["sample_id"]
    common = dict(
        sample_id=sample_id,
        prompt=prompt,
        clean_correct=bool(entry.get("is_correct", False)),
        correct_answer=str(entry.get("correct_answer", "")),
        archive_answer=str(entry.get("extracted_answer") or ""),
    )
    split = split_generated_text(entry["generated_text"])
    if split is None:
        return SamplePlan(
            **common,
            split=None,
            skip_reason="no_answer_pattern",
            residual_answer_in_prefix=False,
            arm_plans={},
        )

    candidates = build_candidates(split.cot_text)
    rc_scores = normalize_ranking(rc_ranking) if rc_ranking else None
    loo_scores = normalize_ranking(loo_ranking) if loo_ranking else None

    arm_plans = {
        spec.name: _plan_arm(
            spec, split.cot_text, candidates, rc_scores, loo_scores,
            seed, sample_id, replacement_sampler,
        )
        for spec in arms
    }
    return SamplePlan(
        **common,
        split=split,
        skip_reason=None,
        residual_answer_in_prefix=_has_answer_pattern(split.cot_text),
        arm_plans=arm_plans,
    )


def judge_sample(plan: SamplePlan, continuations: dict[str, str], extractor) -> dict:
    """生成継続から答えを抽出し、flip / correct→incorrect を判定した record を作る.

    Returns:
        JSON 直列化可能な dict (スキーマは dev notes §4 で凍結)
    """
    record: dict = {
        "sample_id": plan.sample_id,
        "skip_reason": plan.skip_reason,
        "clean_correct": plan.clean_correct,
        "correct_answer": plan.correct_answer,
        "residual_answer_in_prefix": plan.residual_answer_in_prefix,
        "baseline": None,
        "arms": {},
    }
    if plan.skip_reason is not None or plan.split is None:
        return record

    trigger = plan.split.trigger_text
    base_cont = continuations.get(BASELINE_ARM, "")
    base_answer = extractor.extract(trigger + base_cont).extracted_answer.strip()
    record["baseline"] = {
        "generated": base_cont,
        "answer": base_answer,
        "matches_archive": base_answer == plan.archive_answer.strip(),
        "is_correct": extractor.is_correct(base_answer, plan.correct_answer),
    }

    for name, arm_plan in plan.arm_plans.items():
        arm_record: dict = {
            "target_kind": arm_plan.spec.target_kind,
            "op": arm_plan.spec.op,
            "k": arm_plan.spec.k,
            "stratum": arm_plan.spec.stratum,
            "skip_reason": arm_plan.skip_reason,
            "target_words": arm_plan.target_words,
            "matched_to": arm_plan.matched_to,
            "replacements": arm_plan.replacements,
            "n_spans_edited": arm_plan.n_spans_edited,
            "generated": None,
            "answer": None,
            "flip": None,
            "is_correct": None,
            "correct_to_incorrect": None,
        }
        if arm_plan.skip_reason is None:
            cont = continuations.get(name, "")
            answer = extractor.extract(trigger + cont).extracted_answer.strip()
            is_correct = extractor.is_correct(answer, plan.correct_answer)
            arm_record.update(
                generated=cont,
                answer=answer,
                flip=answer != base_answer,
                is_correct=is_correct,
                correct_to_incorrect=bool(plan.clean_correct and not is_correct),
            )
        record["arms"][name] = arm_record
    return record


def run_samples(
    entries: list[dict],
    prompts: list[str],
    rc_rankings: list[list[dict] | None],
    loo_rankings: list[list[dict] | None],
    arms: list[ArmSpec],
    generate_fn: GenerateFn,
    benchmark: str,
    seed: int = 42,
    batch_size: int = 8,
    replacement_sampler=None,
) -> list[dict]:
    """全サンプル × 全腕を teacher-forcing 短生成で流し、record リストを返す.

    Args:
        generate_fn: コンテキストリスト → 継続テキストリスト (greedy を想定)。
            1回の呼び出しは batch_size 以下。
    """
    extractor = create_extractor(benchmark)
    plans = [
        prepare_sample(e, p, rc, loo, arms, seed, replacement_sampler)
        for e, p, rc, loo in zip(entries, prompts, rc_rankings, loo_rankings, strict=True)
    ]

    # (サンプル idx, 腕名, コンテキスト) を平坦化してバッチ処理
    tasks: list[tuple[int, str, str]] = []
    for i, plan in enumerate(plans):
        if plan.skip_reason is not None or plan.split is None:
            continue
        base_ctx = plan.prompt + plan.split.cot_text + plan.split.trigger_text
        tasks.append((i, BASELINE_ARM, base_ctx))
        for name, arm_plan in plan.arm_plans.items():
            if arm_plan.skip_reason is not None:
                continue
            tasks.append(
                (i, name, plan.prompt + arm_plan.edited_text + plan.split.trigger_text)
            )

    continuations: dict[tuple[int, str], str] = {}
    for j in range(0, len(tasks), batch_size):
        batch = tasks[j : j + batch_size]
        outputs = generate_fn([ctx for _, _, ctx in batch])
        for (i, name, _), out in zip(batch, outputs, strict=True):
            continuations[(i, name)] = out

    records = []
    for i, plan in enumerate(plans):
        conts = {name: c for (idx, name), c in continuations.items() if idx == i}
        records.append(judge_sample(plan, conts, extractor))
    return records
