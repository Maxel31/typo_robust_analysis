"""repair.pipeline のテスト (実験9: プロンプト対構築と語レベル行の組み立て).

GPU 不要。プロンプト構築は既存テンプレート (models/prompts.py) を再利用する。
"""

import torch

from typo_cot.repair.archive_access import RepairInputRecord
from typo_cot.repair.pipeline import (
    HF_MODEL_NAMES,
    build_prompt_pair,
    build_word_rows,
)
from typo_cot.repair.span_align import AlignedSpan


def _record(**kw: object) -> RepairInputRecord:
    defaults = dict(
        sample_id="gsm8k_00000",
        model="gemma-3-4b-it",
        benchmark="gsm8k",
        condition="lxt4",
        original_question="Janet has five ducks.",
        perturbed_question="Janet has five dicks.",
        perturbed_tokens=[
            {
                "token_index": 10,
                "original_token": " ducks",
                "perturbed_token": "dicks",
                "importance_score": 0.6,
                "perturbation_type": "proximity",
            }
        ],
        flip=True,
        clean_correct=True,
        span_extract_ok=True,
    )
    defaults.update(kw)
    return RepairInputRecord(**defaults)


class TestHFModelNames:
    def test_paper_models_are_mapped(self) -> None:
        # 論文の5モデル + Qwen2.5-7B
        for short in [
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
            "Mistral-7B-Instruct-v0.3",
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "Qwen2.5-7B-Instruct",
        ]:
            assert short in HF_MODEL_NAMES
            assert "/" in HF_MODEL_NAMES[short]


class TestBuildPromptPair:
    def test_gsm8k_prompts_share_context(self) -> None:
        rec = _record()
        clean_prompt, typo_prompt = build_prompt_pair(rec)
        assert "Janet has five ducks." in clean_prompt
        assert "Janet has five dicks." in typo_prompt
        # 質問以外の部分 (few-shot 文脈) は完全一致
        assert clean_prompt.replace("ducks", "") == typo_prompt.replace("dicks", "")

    def test_mmlu_typo_question_embeds_choices(self) -> None:
        # アーカイブの MMLU perturbed_question は選択肢行 "(A) ..." を
        # 埋め込み済みで perturbed_choices は None (include_choices=True の仕様)。
        # 生成時 (run_inference.py) と同様、typo 側は choices を渡さず
        # clean 側はテンプレートが同一形式で選択肢を付加する。
        rec = _record(
            benchmark="mmlu",
            original_question="Which animal quacks?",
            perturbed_question="Which animal qacks?\n(A) dukc (B) cat (C) dog (D) cow",
            choices=["duck", "cat", "dog", "cow"],
            perturbed_choices=None,
            subset="zoology",
            perturbed_tokens=[
                {
                    "token_index": 5,
                    "original_token": " quacks",
                    "perturbed_token": "qacks",
                    "importance_score": 1.2,
                    "perturbation_type": "omission",
                },
                {
                    "token_index": 9,
                    "original_token": " duck",
                    "perturbed_token": "dukc",
                    "importance_score": 0.4,
                    "perturbation_type": "proximity",
                },
            ],
        )
        clean_prompt, typo_prompt = build_prompt_pair(rec)
        assert "Which animal quacks?" in clean_prompt
        assert "Which animal qacks?" in typo_prompt
        # 選択肢行は両プロンプトに 1 回だけ現れ、同一形式
        assert clean_prompt.count("(A) duck (B) cat (C) dog (D) cow") == 1
        assert typo_prompt.count("(A) dukc (B) cat (C) dog (D) cow") == 1
        # clean 側の選択肢行が typo 側に重複して付加されない
        assert "(A) duck" not in typo_prompt


class TestBuildWordRows:
    def test_rows_carry_scores_and_metadata(self) -> None:
        rec = _record()
        spans = [
            AlignedSpan(
                clean_word="ducks",
                typo_word="dicks",
                clean_start=15,
                clean_end=20,
                typo_start=15,
                typo_end=20,
                importance_score=0.6,
                perturbation_type="proximity",
                token_index=10,
            )
        ]
        # 3層モデルの合成 cos 曲線 (層1 が最大)
        cos_curves = torch.tensor([[0.3], [0.9], [0.7]])  # [L+1, n_spans]
        rows = build_word_rows(
            rec,
            spans,
            cos_curves=cos_curves,
            typo_target_ranks=[[500, 3, 40]],  # span ごとの層別 rank
            clean_self_ranks=[[0, 0, 0]],
            split_increments=[1],
            zipf_freqs=[4.2],
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["sample_id"] == "gsm8k_00000"
        assert row["condition"] == "lxt4"
        assert row["clean_word"] == "ducks"
        assert row["typo_word"] == "dicks"
        assert abs(row["repair_score"] - 0.9) < 1e-6
        assert row["repair_layer"] == 1
        assert row["flip"] is True
        assert row["r_q"] == 0.6
        assert row["split_increment"] == 1
        assert row["zipf_freq"] == 4.2
        # logit lens: typo 側で clean 語先頭トークンが最良 rank 3 (層1)
        assert row["lens_min_rank"] == 3
        assert row["lens_first_hit_layer_top5"] == 1
        # clean 側サニティ: 自身の語を rank 0 で復号
        assert row["clean_self_min_rank"] == 0
        assert row["cos_curve"] == [0.3, 0.9, 0.7]

    def test_nan_curve_is_skipped(self) -> None:
        rec = _record()
        spans = [
            AlignedSpan(
                clean_word="ducks",
                typo_word="dicks",
                clean_start=15,
                clean_end=20,
                typo_start=15,
                typo_end=20,
            )
        ]
        cos_curves = torch.full((3, 1), float("nan"))
        rows = build_word_rows(
            rec,
            spans,
            cos_curves=cos_curves,
            typo_target_ranks=[None],
            clean_self_ranks=[None],
            split_increments=[0],
            zipf_freqs=[1.0],
        )
        assert rows == []
