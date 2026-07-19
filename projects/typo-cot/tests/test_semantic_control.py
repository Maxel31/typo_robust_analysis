"""実験8-fine A3 統制(c): 意味置換ペア生成 make_semantic_pair のテスト.

標的語 (typo の original_token) を「同義でない実語」にランダム置換した
semantic-perturbation ペアを作る純関数を検証する (GPU 不要):

- question_clean 中の標的語が実語に置換され question_typo に入る
- perturbed_token が置換後の実語になる (original_token は不変)
- 置換語は元語と異なる (自明な恒等でない)
- 決定論的 (同 seed で同一)
- 標的語が無い / 見つからない場合は None
"""

from dataclasses import replace

from typo_cot.intervention.records import PairRecord
from typo_cot.intervention.semantic_control import REAL_WORDS, make_semantic_pair


def _pair(question="How many cats are in the box", tokens=None):
    return PairRecord(
        sample_id="s0",
        model="m",
        benchmark="gsm8k",
        question_clean=question,
        question_typo=question.replace("cats", "cts"),
        choices_clean=None,
        choices_typo=None,
        subset=None,
        correct_answer="3",
        cot_clean="clean cot. The answer is 3",
        cot_typo="typo cot. The answer is 5",
        answer_clean="3",
        answer_typo="5",
        is_correct_clean=True,
        extra={
            "perturbed_tokens": tokens
            if tokens is not None
            else [{"original_token": "cats", "perturbed_token": "cts"}],
            "is_correct_typo": False,
        },
    )


class TestMakeSemanticPair:
    def test_replaces_target_with_real_word(self):
        sem = make_semantic_pair(_pair(), seed=1234)
        assert sem is not None
        rep = sem.extra["perturbed_tokens"][0]["perturbed_token"]
        assert rep.lower() in [w.lower() for w in REAL_WORDS]
        # 置換語が semantic 質問に入り、original は semantic 側から消える
        assert rep in sem.question_typo
        assert "cats" not in sem.question_typo.split()
        # original_token は保持
        assert sem.extra["perturbed_tokens"][0]["original_token"] == "cats"

    def test_replacement_differs_from_original(self):
        sem = make_semantic_pair(_pair(), seed=1234)
        rep = sem.extra["perturbed_tokens"][0]["perturbed_token"]
        assert rep.lower() != "cats"

    def test_deterministic(self):
        a = make_semantic_pair(_pair(), seed=7)
        b = make_semantic_pair(_pair(), seed=7)
        assert a.question_typo == b.question_typo

    def test_seed_changes_replacement(self):
        a = make_semantic_pair(_pair(), seed=1)
        b = make_semantic_pair(_pair(), seed=2)
        # 十分大きい実語プールなら seed 差で置換語が変わる
        assert a.question_typo != b.question_typo

    def test_no_perturbed_tokens_returns_none(self):
        p = replace(_pair(), extra={"perturbed_tokens": [], "is_correct_typo": False})
        assert make_semantic_pair(p, seed=1) is None

    def test_target_not_in_question_returns_none(self):
        p = _pair(question="No relevant words here")
        assert make_semantic_pair(p, seed=1) is None

    def test_preserves_capitalization(self):
        p = _pair(question="Cats are nice")
        p.extra["perturbed_tokens"] = [{"original_token": "Cats", "perturbed_token": "Cts"}]
        p.question_typo = "Cts are nice"
        sem = make_semantic_pair(p, seed=3)
        rep = sem.extra["perturbed_tokens"][0]["perturbed_token"]
        assert rep[0].isupper()
