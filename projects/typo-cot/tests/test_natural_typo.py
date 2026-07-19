"""実験10④: 自然typo分布サンプラー (natural_typo) のテスト.

GitHub Typo Corpus から推定した編集操作の経験分布に従って、
標的語固定のまま摂動の入れ方だけを差し替えるためのモジュールを検証する。
"""

import json

import pytest

from typo_cot.perturbation.natural_typo import (
    NaturalTypoDistribution,
    NaturalTypoGenerator,
    apply_natural_typos_to_targets,
    extract_single_edit,
    position_bucket,
)


@pytest.fixture
def toy_distribution() -> NaturalTypoDistribution:
    """テスト用の小さな分布."""
    return NaturalTypoDistribution(
        op_probs={
            "substitution": 0.4,
            "insertion": 0.25,
            "deletion": 0.25,
            "transposition": 0.1,
        },
        position_probs={"first": 0.1, "internal": 0.6, "last": 0.3},
        substitution_given_intended={"a": {"s": 0.7, "e": 0.3}},
        substitution_marginal={"e": 0.5, "s": 0.3, "t": 0.2},
        insertion_given_prev={"l": {"l": 0.9, "e": 0.1}},
        insertion_marginal={"e": 0.6, "s": 0.4},
        metadata={"source": "toy"},
    )


# ---------------------------------------------------------------------------
# 編集操作の抽出 (コーパス→経験分布の推定に使用)
# ---------------------------------------------------------------------------


class TestExtractSingleEdit:
    def test_substitution(self):
        op = extract_single_edit("the dog runs", "the dog tuns")
        assert op is not None
        assert op.operation == "substitution"
        assert op.intended_char == "r"
        assert op.typed_char == "t"
        assert op.word == "runs"
        assert op.bucket == "first"

    def test_deletion(self):
        # 意図: "language" -> 打鍵: "languge" (a が脱落)
        op = extract_single_edit("the language", "the languge")
        assert op is not None
        assert op.operation == "deletion"
        assert op.intended_char == "a"
        assert op.bucket == "internal"

    def test_insertion_doubling(self):
        op = extract_single_edit("apple pie", "applle pie")
        assert op is not None
        assert op.operation == "insertion"
        assert op.typed_char == "l"
        assert op.prev_char == "l"

    def test_transposition(self):
        op = extract_single_edit("receive it", "recieve it")
        assert op is not None
        assert op.operation == "transposition"

    def test_non_alpha_edit_rejected(self):
        # 空白の挿入はアルファベット編集ではない
        assert extract_single_edit("function() {", "function(){") is None

    def test_multi_edit_rejected(self):
        # 2箇所以上の編集は単一操作として扱わない
        assert extract_single_edit("the quick fox", "teh quikc fox") is None

    def test_identical_rejected(self):
        assert extract_single_edit("same", "same") is None

    def test_case_only_change_is_substitution(self):
        # 大文字小文字のみの置換も置換として扱われうる (None でも可)
        op = extract_single_edit("english text", "English text")
        assert op is None or op.operation == "substitution"


class TestPositionBucket:
    def test_first(self):
        assert position_bucket("word", 0) == "first"

    def test_last(self):
        assert position_bucket("word", 3) == "last"

    def test_internal(self):
        assert position_bucket("word", 2) == "internal"

    def test_single_char(self):
        assert position_bucket("a", 0) == "first"


# ---------------------------------------------------------------------------
# 分布のシリアライズ
# ---------------------------------------------------------------------------


class TestDistributionSerialization:
    def test_roundtrip(self, toy_distribution, tmp_path):
        path = tmp_path / "dist.json"
        toy_distribution.save(path)
        loaded = NaturalTypoDistribution.load(path)
        assert loaded.op_probs == toy_distribution.op_probs
        assert loaded.position_probs == toy_distribution.position_probs
        assert loaded.substitution_given_intended == (
            toy_distribution.substitution_given_intended
        )

    def test_json_is_plain(self, toy_distribution, tmp_path):
        path = tmp_path / "dist.json"
        toy_distribution.save(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["op_probs"]) == {
            "substitution",
            "insertion",
            "deletion",
            "transposition",
        }


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------


class TestNaturalTypoGenerator:
    def test_deterministic_with_seed(self, toy_distribution):
        r1 = NaturalTypoGenerator(toy_distribution, seed=123).perturb("importance")
        r2 = NaturalTypoGenerator(toy_distribution, seed=123).perturb("importance")
        assert r1 is not None and r2 is not None
        assert r1.perturbed == r2.perturbed
        assert r1.operation == r2.operation
        assert r1.position == r2.position

    def test_perturbed_differs_from_original(self, toy_distribution):
        for seed in range(20):
            result = NaturalTypoGenerator(toy_distribution, seed=seed).perturb("banana")
            assert result is not None
            assert result.perturbed != "banana"

    def test_operation_is_one_of_four(self, toy_distribution):
        seen = set()
        for seed in range(200):
            result = NaturalTypoGenerator(toy_distribution, seed=seed).perturb("linear")
            assert result is not None
            seen.add(result.operation)
        assert seen <= {"substitution", "insertion", "deletion", "transposition"}
        # 十分な試行で複数の操作が出現する
        assert len(seen) >= 3

    def test_non_alpha_token_returns_none(self, toy_distribution):
        gen = NaturalTypoGenerator(toy_distribution, seed=0)
        assert gen.perturb("1234") is None
        assert gen.perturb("(),.") is None
        assert gen.perturb("") is None

    def test_single_char_no_deletion_or_transposition(self, toy_distribution):
        for seed in range(50):
            result = NaturalTypoGenerator(toy_distribution, seed=seed).perturb("a")
            assert result is not None
            assert result.operation in {"substitution", "insertion"}

    def test_substitution_preserves_case(self, toy_distribution):
        # 大文字のみの語: 置換が起きた場合は大文字が保たれる
        for seed in range(100):
            result = NaturalTypoGenerator(toy_distribution, seed=seed).perturb("AAAA")
            assert result is not None
            if result.operation == "substitution":
                assert result.new_char is not None
                assert result.new_char.isupper()

    def test_transposition_swaps_adjacent(self):
        dist = NaturalTypoDistribution(
            op_probs={"transposition": 1.0},
            position_probs={"first": 1.0, "internal": 0.0, "last": 0.0},
            substitution_given_intended={},
            substitution_marginal={},
            insertion_given_prev={},
            insertion_marginal={},
        )
        result = NaturalTypoGenerator(dist, seed=0).perturb("ab")
        assert result is not None
        assert result.operation == "transposition"
        assert result.perturbed == "ba"

    def test_length_change_consistency(self, toy_distribution):
        for seed in range(100):
            result = NaturalTypoGenerator(toy_distribution, seed=seed).perturb("gradient")
            assert result is not None
            if result.operation == "insertion":
                assert len(result.perturbed) == len("gradient") + 1
            elif result.operation == "deletion":
                assert len(result.perturbed) == len("gradient") - 1
            else:
                assert len(result.perturbed) == len("gradient")


# ---------------------------------------------------------------------------
# 標的語固定の適用 (A/B 設計の B 側)
# ---------------------------------------------------------------------------


class TestApplyNaturalTyposToTargets:
    def test_fixed_targets_are_perturbed(self, toy_distribution):
        text = "Janet has sixteen ducks in the yard"
        # offset_mapping はプロンプト全体基準 (text_char_start=100 とする)
        offsets = {
            10: (100, 105),  # Janet
            11: (106, 109),  # has
            12: (110, 117),  # sixteen
            13: (118, 123),  # ducks
        }
        targets = [
            {"token_index": 10, "original_token": " Janet", "importance_score": 2.0},
            {"token_index": 13, "original_token": " ducks", "importance_score": 1.5},
        ]
        perturbed_text, entries, warnings = apply_natural_typos_to_targets(
            text=text,
            targets=targets,
            offset_mapping=offsets,
            text_char_start=100,
            distribution=toy_distribution,
            seed=42,
            sample_id="test_000",
        )
        assert len(entries) == 2
        assert perturbed_text != text
        # 標的でない語は変わらない
        assert "has" in perturbed_text
        assert "sixteen" in perturbed_text
        assert "in the yard" in perturbed_text
        # entries は token_index 昇順
        assert [e.token_index for e in entries] == [10, 13]
        assert not warnings

    def test_reproducible(self, toy_distribution):
        text = "Janet has sixteen ducks"
        offsets = {5: (50, 55), 8: (62, 67)}
        targets = [
            {"token_index": 5, "original_token": " Janet", "importance_score": 1.0},
            {"token_index": 8, "original_token": " ducks", "importance_score": 0.5},
        ]
        out1 = apply_natural_typos_to_targets(
            text, targets, offsets, 50, toy_distribution, 42, "s1"
        )
        out2 = apply_natural_typos_to_targets(
            text, targets, offsets, 50, toy_distribution, 42, "s1"
        )
        assert out1[0] == out2[0]

    def test_mismatch_warning(self, toy_distribution):
        text = "totally different words"
        offsets = {3: (10, 15)}
        targets = [
            {"token_index": 3, "original_token": " Janet", "importance_score": 1.0},
        ]
        _, _, warnings = apply_natural_typos_to_targets(
            text, targets, offsets, 10, toy_distribution, 42, "s2"
        )
        assert warnings
