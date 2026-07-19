"""実験6-(i)〜(iii): 帰属ファミリー代替手法 (G×I / IG / attention rollout)."""

from typo_cot.attribution_family.methods import (
    PreparedSample,
    answer_logprob_from_logits,
    attention_rollout_token_scores,
    decode_tokens_for_alignment,
    gradient_x_input_token_scores,
    integrated_gradients_token_scores,
    prepare_sample,
    rollout_from_attentions,
    token_scores_to_word_ranking,
)

__all__ = [
    "PreparedSample",
    "answer_logprob_from_logits",
    "attention_rollout_token_scores",
    "decode_tokens_for_alignment",
    "gradient_x_input_token_scores",
    "integrated_gradients_token_scores",
    "prepare_sample",
    "rollout_from_attentions",
    "token_scores_to_word_ranking",
]
