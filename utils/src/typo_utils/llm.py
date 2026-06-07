"""Shared LLM wrapper and model-loading utility.

Faithful port of the ``LLM`` class and ``load_model`` function from
Tsuji et al.'s reference implementation (utils.py).

Heavy imports (torch, transformers) are deferred inside functions/class
bodies so that this module remains importable on CPU-only machines.
"""
from __future__ import annotations

__all__ = ["load_model", "LLM"]


def load_model(model_name: str, **kwargs):
    """Load a tokenizer and causal-LM from HuggingFace.

    Faithful port of ``load_model`` in Tsuji et al. utils.py:
    - ``AutoTokenizer.from_pretrained(model_name)``
    - ``AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
      torch_dtype=torch.bfloat16, **kwargs)``

    Returns
    -------
    tokenizer, model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    return tokenizer, model


class LLM:
    """Thin wrapper around a HuggingFace causal-LM.

    Faithful port of the ``LLM`` class in Tsuji et al. utils.py.

    Parameters
    ----------
    model:
        A loaded ``AutoModelForCausalLM`` (or compatible) instance.
    tokenizer:
        The corresponding tokenizer.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # generate_word
    # ------------------------------------------------------------------

    def generate_word(
        self,
        input_ids,
        max_new_tokens: int = 10,
        no_stop_token: bool = False,
        **kwargs,
    ) -> str:
        """Greedy-decode a single word, stopping at the ``'`` (apostrophe) token.

        Faithful port of ``LLM.generate_word`` in Tsuji et al. utils.py.
        """
        import torch

        if no_stop_token:
            eos_token_id = self.tokenizer.eos_token_id
        else:
            eos_token_id = [
                self.tokenizer.convert_tokens_to_ids("'"),
                self.tokenizer.eos_token_id,
            ]

        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            output = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                top_p=None,
                temperature=None,
                attention_mask=torch.ones_like(input_ids).to(self.model.device),
                **kwargs,
            )
        else:
            output = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                top_p=None,
                temperature=None,
                **kwargs,
            )

        output_word = self.tokenizer.decode(
            output[0, input_ids.size(1) : -1].to("cpu")
        )
        if output_word.find("'") != -1 and not no_stop_token:
            output_word = output_word[: output_word.find("'")]
        return output_word

    # ------------------------------------------------------------------
    # get_prob
    # ------------------------------------------------------------------

    def get_prob(self, input_ids, output_ids, **kwargs) -> float:
        """Compute the probability of ``output_ids`` given ``input_ids``.

        P = exp( sum_i log p(output_ids[i] | context_up_to_i) )

        Faithful port of ``LLM.get_prob`` in Tsuji et al. utils.py.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()

        all_input_ids = torch.cat([input_ids, output_ids], dim=-1)

        with torch.no_grad():
            outputs = self.model(
                all_input_ids.to(self.device),
                attention_mask=torch.ones_like(all_input_ids).to(self.model.device),
                use_cache=False,
                **kwargs,
            )
            logits = outputs.logits.to("cpu")

        log_probs = []
        for i in range(output_ids.size(1)):
            target_token_id = output_ids[0, i]
            prev_token_logits = logits[:, input_ids.size(1) + i - 1, :]
            probabilities = F.softmax(prev_token_logits, dim=-1)
            token_prob = probabilities[0, target_token_id].item()
            log_probs.append(torch.log(torch.tensor(token_prob)))

        log_prob_sum = torch.sum(torch.stack(log_probs))
        probability = torch.exp(log_prob_sum).item()
        return probability

    # ------------------------------------------------------------------
    # get_importance
    # ------------------------------------------------------------------

    def get_importance(self, input_ids, output_ids) -> list:
        """Gradient-based token-importance saliency over the input tokens.

        Returns a list of length ``input_ids.size(1)`` — one scalar per
        input token, computed as the L1 norm of the gradient w.r.t. the
        token embedding.

        Faithful port of ``LLM.get_importance`` in Tsuji et al. utils.py.
        """
        import torch

        self.model.eval()

        all_input_ids = torch.cat([input_ids, output_ids], dim=-1)
        embeddings = self.model.get_input_embeddings()(
            all_input_ids.to(self.device)
        )
        embeddings.retain_grad()

        outputs = self.model(
            inputs_embeds=embeddings.to(self.device),
            attention_mask=torch.ones_like(all_input_ids).to(self.model.device),
            use_cache=False,
        )
        last_token_logits = outputs.logits[:, input_ids.size(1) - 1 :, :]
        output_index = torch.tensor(
            [[0, i, output_id] for i, output_id in enumerate(output_ids[0])]
        )
        loss = last_token_logits[tuple(output_index.T)].sum()

        loss.backward()

        grads = embeddings.grad[: input_ids.size(1)]
        token_importance = grads.abs().sum(dim=-1).to("cpu")

        self.model.zero_grad()
        del loss
        torch.cuda.empty_cache()

        return token_importance[0].tolist()
