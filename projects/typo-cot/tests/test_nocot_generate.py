"""実験14: no-CoT 生成コアのテスト (generate_fn を注入して GPU 不要検証)."""

from typo_cot.nocot.generate import build_nocot_prompt, generate_nocot_records


class TestBuildPrompt:
    def test_gsm8k_prompt_direct(self) -> None:
        p = build_nocot_prompt(
            {"question": "What is 2+2?", "choices": None, "subset": None}, "gsm8k"
        )
        assert "Problem: What is 2+2?" in p
        assert p.rstrip().endswith("Solution:")
        # no-CoT 例示 (reasoning 無し) が含まれる
        assert "Solution: The answer is 11." in p

    def test_mmlu_prompt_uses_choices(self) -> None:
        p = build_nocot_prompt(
            {
                "question": "Q?",
                "choices": ["x", "y", "z", "w"],
                "subset": "math",
            },
            "mmlu",
        )
        assert "(A) x" in p and "(B) y" in p
        assert p.rstrip().endswith("Step-by-step reasoning:")

    def test_mmlu_typo_choices_inline(self) -> None:
        # 摂動データは choices=None で question に選択肢が内包される
        p = build_nocot_prompt(
            {"question": "Q?\n(A) x (B) y (C) z (D) w", "choices": None, "subset": "math"},
            "mmlu",
        )
        assert "(A) x" in p


class TestGenerateRecords:
    def _samples(self) -> list[dict]:
        return [
            {"sample_id": "s1", "question": "Q1", "choices": ["a", "b", "c", "d"],
             "correct_answer": "B", "subset": "math"},
            {"sample_id": "s2", "question": "Q2", "choices": ["a", "b", "c", "d"],
             "correct_answer": "A", "subset": "math"},
        ]

    def test_extracts_and_scores(self) -> None:
        # mock: s1 -> B (correct), s2 -> C (wrong)
        outs = ["The answer is (B).", "The answer is (C)."]

        def gen_fn(prompts: list[str]) -> list[str]:
            assert len(prompts) == 2
            return outs

        recs = generate_nocot_records(self._samples(), "mmlu", gen_fn, batch_size=8)
        assert recs["s1"]["answer"] == "B"
        assert recs["s1"]["is_correct"] is True
        assert recs["s2"]["answer"] == "C"
        assert recs["s2"]["is_correct"] is False

    def test_batching_covers_all(self) -> None:
        calls = []

        def gen_fn(prompts: list[str]) -> list[str]:
            calls.append(len(prompts))
            return ["The answer is (A)."] * len(prompts)

        recs = generate_nocot_records(self._samples(), "mmlu", gen_fn, batch_size=1)
        assert set(recs.keys()) == {"s1", "s2"}
        assert calls == [1, 1]  # batch_size=1 -> 2 calls
        # s2 correct_answer is A
        assert recs["s2"]["is_correct"] is True
