"""実験14: no-CoT flip 集計ロジックのテスト (GPU 不要の合成テスト).

flip の定義 (実験1 の DE=C セルと整合):
  no-CoT flip = clean 質問の no-CoT 回答が正解 かつ typo 質問の no-CoT 回答が
  不正解 (= clean正解→摂動誤答)。clean 正解に条件付けたとき、答えが変われば
  必ず不正解になる (correct_answer は一意) ため、DE の C セル
  (answers[C] != answers[A]) の no-CoT 版と対応する。
"""

import math

from typo_cot.nocot.flip import flip_summary, join_records, odds_ratio


def _rec(answer: str, is_correct: bool) -> dict:
    return {"answer": answer, "is_correct": is_correct}


class TestJoinRecords:
    def test_join_only_common_ids(self) -> None:
        clean = {"a": _rec("1", True), "b": _rec("2", True)}
        typo = {"a": _rec("9", False), "c": _rec("3", True)}
        joined = join_records(clean, typo)
        assert [r["sample_id"] for r in joined] == ["a"]

    def test_answer_changed_and_flip_flags(self) -> None:
        clean = {"a": _rec("1", True), "b": _rec("2", True), "c": _rec("3", False)}
        typo = {"a": _rec("9", False), "b": _rec("2", True), "c": _rec("3", False)}
        joined = {r["sample_id"]: r for r in join_records(clean, typo)}
        # a: clean correct, typo wrong, answer changed -> flip
        assert joined["a"]["answer_changed"] is True
        assert joined["a"]["flip_correct_to_wrong"] is True
        # b: clean correct, typo same/correct -> no flip
        assert joined["b"]["answer_changed"] is False
        assert joined["b"]["flip_correct_to_wrong"] is False
        # c: clean already wrong -> not a correct_to_wrong flip
        assert joined["c"]["flip_correct_to_wrong"] is False


class TestFlipSummary:
    def test_rates(self) -> None:
        clean = {
            "a": _rec("1", True),
            "b": _rec("2", True),
            "c": _rec("3", True),
            "d": _rec("4", False),
        }
        typo = {
            "a": _rec("9", False),  # flip
            "b": _rec("2", True),  # stay
            "c": _rec("7", False),  # flip
            "d": _rec("4", False),  # clean wrong, ignored in flip rate
        }
        s = flip_summary(join_records(clean, typo))
        assert s["n_joined"] == 4
        assert s["n_clean_correct"] == 3
        assert s["n_flip"] == 2
        assert math.isclose(s["nocot_flip_rate"], 2 / 3)

    def test_empty_population(self) -> None:
        s = flip_summary([])
        assert s["n_joined"] == 0
        assert s["nocot_flip_rate"] is None


class TestOddsRatio:
    def test_basic_or(self) -> None:
        # a=both flip, b=cflip only, c=nflip only, d=neither
        res = odds_ratio(a=20, b=5, c=5, d=20, haldane=False)
        assert math.isclose(res["odds_ratio"], (20 * 20) / (5 * 5))

    def test_haldane_correction_avoids_div_zero(self) -> None:
        res = odds_ratio(a=10, b=0, c=0, d=10, haldane=True)
        assert res["odds_ratio"] is not None
        assert res["odds_ratio"] > 1

    def test_positive_association_gt_one(self) -> None:
        # samples that flip under DE also tend to flip under no-CoT
        res = odds_ratio(a=30, b=3, c=3, d=30, haldane=False)
        assert res["odds_ratio"] > 3
