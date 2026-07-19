"""実験6-(i)〜(iii) 帰属ファミリー代替手法のユニットテスト（GPU 不要・モックのみ）.

対象 (experiment_plan.md §4 実験6):
- (i)   Gradient×Input   — 答えトークン列 log-prob を目的関数、backward 1回
- (ii)  Integrated Gradients — 同目的関数、ベースライン=ゼロ埋め込み、m ステップ
- (iii) Attention rollout — attention 行列の層積（forward のみ）

いずれも LOO (exp6-iv) と同じ 3分割 (prompt + CoT + trigger + answer) と
語ランキング規約 (rc_word_ranking_from_token_scores 互換) を共有する。
"""

from types import SimpleNamespace

import pytest
import torch

from typo_cot.attribution_family.methods import (
    answer_logprob_from_logits,
    attention_rollout_token_scores,
    decode_tokens_for_alignment,
    gradient_x_input_token_scores,
    integrated_gradients_token_scores,
    prepare_sample,
    rollout_from_attentions,
    token_scores_to_word_ranking,
)

# ============================================================
# モック（GPU / ネットワーク不要）
# ============================================================


class MockTokenizer:
    """空白区切りのトークナイザ. 同一インスタンス内で語彙は安定."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<s>": 1}
        self.pad_token_id = 0
        self.all_special_ids = [0, 1]

    def _ids(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            ids.append(self.vocab[w])
        return ids

    def __call__(self, text: str, return_tensors=None, **kwargs):
        ids = self._ids(text)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return " ".join(rev.get(i, "?") for i in ids)


class MockEmbedModel(torch.nn.Module):
    """inputs_embeds を受け取れる決定的な微分可能モデル.

    logits[b, t, :] = head(embed[b, t])。位置間の混合はないが、
    目的関数の位置スライス・backward 配線の検証には十分。
    """

    def __init__(self, vocab_size: int = 64, dim: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(vocab_size, dim, generator=gen)
            )
            self.head.weight.copy_(torch.randn(vocab_size, dim, generator=gen))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return SimpleNamespace(logits=self.head(inputs_embeds))


class MockAttentionModel(torch.nn.Module):
    """output_attentions=True で固定 attention 行列を返すモック."""

    def __init__(self, attentions: list[torch.Tensor]) -> None:
        super().__init__()
        self._attentions = attentions

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        seq = input_ids.shape[1]
        return SimpleNamespace(
            logits=torch.zeros(1, seq, 4),
            attentions=tuple(a.unsqueeze(0) for a in self._attentions),
        )


@pytest.fixture()
def mock_lm():
    return MockEmbedModel(), MockTokenizer()


PROMPT = "Q: two plus three ?\nA:"
GENERATED = " two plus three equals five . The answer is 5"


# ============================================================
# prepare_sample: 3分割とトークン範囲
# ============================================================


class TestPrepareSample:
    def test_ranges(self, mock_lm):
        _, tok = mock_lm
        prep = prepare_sample(tok, PROMPT, GENERATED)
        assert prep is not None
        n_prompt = len(tok(PROMPT)["input_ids"])
        n_full = len(prep.input_ids)
        # CoT はプロンプト直後から trigger 直前まで
        assert prep.cot_token_start == n_prompt
        assert prep.cot_token_end >= prep.cot_token_start
        # 答えトークンは末尾（"5" 1トークン）
        assert prep.target_start == n_full - 1
        assert prep.answer_text == "5"
        # trigger ("The answer is") は CoT に含まれない
        cot_ids = prep.input_ids[prep.cot_token_start : prep.cot_token_end + 1]
        assert tok.decode(cot_ids).split() == "two plus three equals five .".split()

    def test_no_answer_pattern_returns_none(self, mock_lm):
        _, tok = mock_lm
        assert prepare_sample(tok, PROMPT, " no final answer here") is None

    def test_full_text_excludes_after_answer(self, mock_lm):
        _, tok = mock_lm
        prep = prepare_sample(tok, PROMPT, GENERATED + " . done")
        assert prep.full_text.endswith("5")


# ============================================================
# 目的関数: 答えトークン列 log-prob（teacher forcing、位置 t-1 が t を予測）
# ============================================================


def _naive_answer_logprob(logits: torch.Tensor, ids: list[int], target_start: int) -> float:
    """独立オラクル: 全位置 log_softmax → 答え位置を拾って合計."""
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    for t in range(target_start, len(ids)):
        total += float(logp[t - 1, ids[t]])
    return total


class TestAnswerLogprob:
    def test_matches_naive_oracle(self, mock_lm):
        model, tok = mock_lm
        ids = tok("a b c d e")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits
        for start in (1, 2, len(ids) - 1):
            got = answer_logprob_from_logits(logits, ids, start)
            assert got.shape == (1,)
            assert float(got[0]) == pytest.approx(
                _naive_answer_logprob(logits, ids, start), abs=1e-5
            )

    def test_batched(self, mock_lm):
        model, tok = mock_lm
        ids = tok("a b c d")["input_ids"]
        with torch.no_grad():
            single = model(input_ids=torch.tensor([ids])).logits
        batch = torch.cat([single, single], dim=0)
        got = answer_logprob_from_logits(batch, ids, 2)
        assert got.shape == (2,)
        assert float(got[0]) == pytest.approx(float(got[1]))


# ============================================================
# (i) Gradient×Input
# ============================================================


class TestGradientXInput:
    def test_matches_autograd_oracle(self, mock_lm):
        model, tok = mock_lm
        ids = tok("a b c d e f")["input_ids"]
        target_start = 4

        scores, obj = gradient_x_input_token_scores(model, ids, target_start)
        assert len(scores) == len(ids)

        # 独立オラクル: 素の autograd で同じ目的関数の grad×input を計算
        embeds = model.get_input_embeddings()(torch.tensor([ids])).detach()
        embeds.requires_grad_(True)
        logits = model(inputs_embeds=embeds).logits
        loss = torch.tensor(0.0)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        for t in range(target_start, len(ids)):
            loss = loss + logp[t - 1, ids[t]]
        loss.backward()
        oracle = (embeds.grad * embeds).sum(-1)[0]

        assert float(obj) == pytest.approx(float(loss), abs=1e-5)
        for got, want in zip(scores, oracle.tolist(), strict=True):
            assert got == pytest.approx(want, abs=1e-5)


# ============================================================
# (ii) Integrated Gradients
# ============================================================


class TestIntegratedGradients:
    def test_completeness(self, mock_lm):
        """sum(IG) ≈ F(x) - F(0)（ステップ数を上げれば収束する）."""
        model, tok = mock_lm
        ids = tok("a b c d e")["input_ids"]
        scores, info = integrated_gradients_token_scores(
            model, ids, target_start=3, steps=64, step_batch=8
        )
        assert len(scores) == len(ids)
        delta = info["f_x"] - info["f_baseline"]
        assert info["sum_attr"] == pytest.approx(delta, rel=0.05, abs=1e-3)
        assert info["completeness_ratio"] == pytest.approx(1.0, rel=0.05)

    def test_step_batching_is_equivalent(self, mock_lm):
        model, tok = mock_lm
        ids = tok("a b c d e")["input_ids"]
        s1, _ = integrated_gradients_token_scores(
            model, ids, target_start=3, steps=8, step_batch=1
        )
        s2, _ = integrated_gradients_token_scores(
            model, ids, target_start=3, steps=8, step_batch=8
        )
        for a, b in zip(s1, s2, strict=True):
            assert a == pytest.approx(b, abs=1e-5)


# ============================================================
# (iii) Attention rollout
# ============================================================


class TestAttentionRollout:
    def test_two_layer_hand_computed(self):
        # 1 head, S=2, 両層同一: A = [[1,0],[0.5,0.5]]
        a = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])  # (H=1, S, S)
        # 0.5I + 0.5A = [[1,0],[0.25,0.75]] (行和1で正規化不要)
        # R = A_hat @ A_hat = [[1,0],[0.4375,0.5625]]
        r = rollout_from_attentions([a, a], residual=0.5)
        want = torch.tensor([[1.0, 0.0], [0.4375, 0.5625]])
        assert torch.allclose(r, want, atol=1e-6)

    def test_rows_are_normalized(self):
        a = torch.rand(2, 5, 5)
        a = a / a.sum(-1, keepdim=True)
        r = rollout_from_attentions([a, a, a], residual=0.5)
        assert torch.allclose(r.sum(-1), torch.ones(5), atol=1e-5)

    def test_scores_from_model(self):
        # 答えトークン(位置3)を予測する行 = 位置2 の rollout 行
        a = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.2, 0.3, 0.5, 0.0],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ]
        )
        model = MockAttentionModel([a])
        ids = [3, 4, 5, 6]
        scores = attention_rollout_token_scores(model, ids, target_start=3)
        r = rollout_from_attentions([a], residual=0.5)
        assert scores == pytest.approx(r[2].tolist(), abs=1e-6)


# ============================================================
# トークンスコア → CoT 語ランキング（R_C 再構築ローダーと同一規約）
# ============================================================


class TestWordRanking:
    def test_cot_range_and_aggregation(self):
        full_text = "Q: x\nA: two plus three is five The answer is 5"
        tokens = ["<s>"] + [" " + w for w in full_text.split()]
        tokens[1] = "Q:"  # 先頭語には空白プレフィックスなし
        scores = [99.0] + [float(i) for i in range(len(tokens) - 1)]
        # tokens: 0=<s> 1=Q: 2=x 3=A: 4=two 5=plus 6=three 7=is 8=five 9=The ...
        # CoT = "two plus three is five" (トークン 4..8)
        ranking = token_scores_to_word_ranking(
            tokens, scores, full_text, cot_token_start=4, cot_token_end=8
        )
        words = [r["word"] for r in ranking]
        assert set(words) == {"two", "plus", "three", "is", "five"}
        # スコア降順
        got_scores = [r["score"] for r in ranking]
        assert got_scores == sorted(got_scores, reverse=True)
        # trigger/answer ("The answer is 5") と prompt は含まれない
        assert "answer" not in words
        assert "5" not in words

    def test_special_token_replacement(self):
        tok = MockTokenizer()
        ids = tok("hello world")["input_ids"]
        toks = decode_tokens_for_alignment(tok, [1] + ids)
        assert toks[0] == "<s>"  # special id は "<s>" に写像され整合対象外
