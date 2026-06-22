"""較正データの準備。"""

from __future__ import annotations


def prepare_calibration_data(
    dataset_name: str = "wikitext",
    num_samples: int = 512,
    max_length: int = 2048,
    seed: int = 42,
) -> list[str]:
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-v1", split="train", trust_remote_code=False
        )
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=False)
    else:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=False)

    import random

    rng = random.Random(seed)
    texts: list[str] = []
    for row in ds:
        text = row.get("text", "").strip()
        if len(text) < 100:
            continue
        texts.append(text[:max_length])
        if len(texts) >= num_samples * 2:
            break

    rng.shuffle(texts)
    return texts[:num_samples]
