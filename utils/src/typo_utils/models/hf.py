"""HuggingFace モデルのロードラッパ（任意依存: extra ``llm``）。"""

from __future__ import annotations

from typing import Any


def load_causal_lm(model_name: str, device: str | None = None, **kwargs: Any):
    """因果言語モデルとトークナイザをロードして返す。

    ``transformers`` は optional extra。未インストール時は分かりやすいエラーを出す。

    Returns:
        (model, tokenizer)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "transformers/torch が必要です。`uv sync --extra llm` を実行してください。"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer
