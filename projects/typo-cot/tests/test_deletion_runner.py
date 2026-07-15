"""実験2: deletion_runner (腕構築 + teacher-forcing 再生成 + flip 判定) のテスト.

生成はモック generate_fn (プロンプトリスト → 継続テキストリスト) で駆動する
(実験1 runner と同じ注入方式)。flip = 基準腕 (無編集 prefix 再生成) との
抽出答え不一致。主推定量は clean 正解条件付き (correct→incorrect)。
"""

from typo_cot.intervention.deletion_runner import (
    BASELINE_ARM,
    ArmSpec,
    core_arms,
    full_grid_arms,
    prepare_sample,
    run_samples,
    smoke_arms,
)

COT_FULL = (
    "\nJanet's ducks lay 16 eggs per day.\n"
    "She eats 3 eggs for breakfast and bakes muffins.\n"
    "So she has 16 - 3 = 13 eggs left, selling each for $2.\n"
    "The answer is 18.\n"
)

ENTRY = {
    "sample_id": "gsm8k_00000",
    "generated_text": COT_FULL,
    "extracted_answer": "18",
    "is_correct": True,
    "correct_answer": "18",
}

PROMPT = "Solve the problem.\nQ: Janet has ducks.\nA:"

RC_RANKING = [
    {"word": "16", "score": 3.7},
    {"word": "eggs", "score": 1.2},
    {"word": "muffins", "score": 0.8},
    {"word": "breakfast", "score": 0.4},
    {"word": "ducks", "score": 0.1},
]

LOO_RANKING = [
    {"word": "breakfast", "score": 2.0},
    {"word": "eggs", "score": 1.0},
]


def _arm(name, kind, op="delete", k=1, stratum="content"):
    return ArmSpec(name=name, target_kind=kind, op=op, k=k, stratum=stratum)


class TestArmFactories:
    def test_core_arms(self):
        arms = core_arms()
        kinds = {a.target_kind for a in arms}
        assert kinds == {"top_rc", "matched_random"}
        assert all(a.op == "delete" and a.k == 1 and a.stratum == "content" for a in arms)

    def test_smoke_arms_include_loo_and_numeric(self):
        arms = smoke_arms()
        kinds = {(a.target_kind, a.stratum) for a in arms}
        assert ("top_loo", "content") in kinds
        assert ("top_rc", "numeric") in kinds

    def test_full_grid_has_27_content_cells(self):
        arms = full_grid_arms()
        content_grid = [
            a
            for a in arms
            if a.stratum == "content"
            and a.target_kind in ("top_rc", "matched_random", "bottom_rc")
        ]
        assert len(content_grid) == 27  # 標的3 × 操作3 × 用量3
        names = [a.name for a in arms]
        assert len(names) == len(set(names))


class TestPrepareSample:
    def test_no_answer_pattern_skips_sample(self):
        entry = dict(ENTRY, generated_text="I do not know.")
        plan = prepare_sample(
            entry, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_delete_k1", "top_rc")], seed=42,
        )
        assert plan.skip_reason == "no_answer_pattern"

    def test_top_rc_targets_top_content_word(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_delete_k1", "top_rc")], seed=42,
        )
        arm = plan.arm_plans["top_rc_delete_k1"]
        assert arm.target_words == ["eggs"]  # 数値 16 は content 層で飛ぶ
        assert "eggs" not in arm.edited_text

    def test_numeric_arm_targets_numeric_word(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("numeric_top_rc_delete_k1", "top_rc", stratum="numeric")], seed=42,
        )
        arm = plan.arm_plans["numeric_top_rc_delete_k1"]
        assert arm.target_words == ["16"]

    def test_matched_random_excludes_top_words(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("matched_random_delete_k1", "matched_random")], seed=42,
        )
        arm = plan.arm_plans["matched_random_delete_k1"]
        assert arm.target_words[0] != "eggs"
        assert arm.matched_to == ["eggs"]

    def test_matched_random_deterministic_across_calls(self):
        kwargs = dict(
            rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("matched_random_delete_k1", "matched_random")], seed=42,
        )
        p1 = prepare_sample(ENTRY, PROMPT, **kwargs)
        p2 = prepare_sample(ENTRY, PROMPT, **kwargs)
        assert (
            p1.arm_plans["matched_random_delete_k1"].target_words
            == p2.arm_plans["matched_random_delete_k1"].target_words
        )

    def test_top_loo_uses_loo_ranking(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=LOO_RANKING,
            arms=[_arm("top_loo_delete_k1", "top_loo")], seed=42,
        )
        assert plan.arm_plans["top_loo_delete_k1"].target_words == ["breakfast"]

    def test_missing_loo_ranking_skips_arm(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_loo_delete_k1", "top_loo")], seed=42,
        )
        assert plan.arm_plans["top_loo_delete_k1"].skip_reason == "missing_ranking"

    def test_insufficient_candidates_skips_arm(self):
        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_delete_k4", "top_rc", k=4)], seed=42,
        )
        # ランキング上の content 候補は eggs/muffins/breakfast/ducks の 4 語 —
        # ここでは k=4 は成立する。k=8 は不足で skip
        plan8 = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_delete_k8", "top_rc", k=8)], seed=42,
        )
        assert plan.arm_plans["top_rc_delete_k4"].skip_reason is None
        assert plan8.arm_plans["top_rc_delete_k8"].skip_reason == "insufficient_candidates"

    def test_residual_answer_flag(self):
        entry = dict(
            ENTRY,
            generated_text=(
                "The answer is 5. Wait, recompute.\n16 - 3 = 13.\nThe answer is 18.\n"
            ),
        )
        plan = prepare_sample(
            entry, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_delete_k1", "top_rc")], seed=42,
        )
        assert plan.residual_answer_in_prefix is True

    def test_replace_arm_uses_injected_sampler(self):
        class FakeSampler:
            def sample(self, word, rng):
                return "cards"

        plan = prepare_sample(
            ENTRY, PROMPT, rc_ranking=RC_RANKING, loo_ranking=None,
            arms=[_arm("top_rc_replace_k1", "top_rc", op="replace")], seed=42,
            replacement_sampler=FakeSampler(),
        )
        arm = plan.arm_plans["top_rc_replace_k1"]
        assert arm.replacements == {"eggs": "cards"}
        assert "cards" in arm.edited_text


def fake_generate(prompts: list[str]) -> list[str]:
    """eggs が文脈に残っていれば 18、消えていれば 99 を答えるモック."""
    return ["18.\n" if "eggs" in p else "99.\n" for p in prompts]


class TestRunSamples:
    ARMS = [
        _arm("top_rc_delete_k1", "top_rc"),
        _arm("matched_random_delete_k1", "matched_random"),
    ]

    def _run(self, entries=None, batch_size=8):
        entries = entries or [ENTRY]
        return run_samples(
            entries=entries,
            prompts=[PROMPT] * len(entries),
            rc_rankings=[RC_RANKING] * len(entries),
            loo_rankings=[None] * len(entries),
            arms=self.ARMS,
            generate_fn=fake_generate,
            benchmark="gsm8k",
            seed=42,
            batch_size=batch_size,
        )

    def test_flip_direction_top_flips_random_does_not(self):
        rec = self._run()[0]
        assert rec["baseline"]["answer"] == "18"
        assert rec["baseline"]["matches_archive"] is True
        assert rec["arms"]["top_rc_delete_k1"]["answer"] == "99"
        assert rec["arms"]["top_rc_delete_k1"]["flip"] is True
        assert rec["arms"]["matched_random_delete_k1"]["flip"] is False

    def test_correct_to_incorrect(self):
        rec = self._run()[0]
        assert rec["clean_correct"] is True
        assert rec["arms"]["top_rc_delete_k1"]["correct_to_incorrect"] is True
        assert rec["arms"]["matched_random_delete_k1"]["correct_to_incorrect"] is False

    def test_record_schema_has_arm_metadata(self):
        rec = self._run()[0]
        arm = rec["arms"]["top_rc_delete_k1"]
        for key in ("target_kind", "op", "k", "stratum", "target_words", "n_spans_edited"):
            assert key in arm
        assert arm["stratum"] == "content"

    def test_batching_respects_batch_size(self):
        calls: list[int] = []

        def counting_generate(prompts):
            calls.append(len(prompts))
            return fake_generate(prompts)

        run_samples(
            entries=[ENTRY, dict(ENTRY, sample_id="gsm8k_00001")],
            prompts=[PROMPT, PROMPT],
            rc_rankings=[RC_RANKING, RC_RANKING],
            loo_rankings=[None, None],
            arms=self.ARMS,
            generate_fn=counting_generate,
            benchmark="gsm8k",
            seed=42,
            batch_size=4,
        )
        assert all(n <= 4 for n in calls)
        # 2 サンプル × (baseline + 2 腕) = 6 コンテキスト
        assert sum(calls) == 6

    def test_skipped_sample_emits_record_without_arms(self):
        rec = self._run(entries=[dict(ENTRY, generated_text="no answer here")])[0]
        assert rec["skip_reason"] == "no_answer_pattern"
        assert rec["arms"] == {}

    def test_baseline_arm_constant(self):
        assert BASELINE_ARM == "baseline"
