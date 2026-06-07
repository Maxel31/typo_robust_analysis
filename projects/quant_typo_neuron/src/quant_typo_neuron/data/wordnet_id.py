"""M0: WordNet word-identification dataset construction.

Faithful port of Tsuji et al.'s dataset construction pipeline:
- create_dataset.py  (create_dataset, rank_list)
- add_typo.py        (char-typo functions, keyboard_proximity)
- utils.py           (make_prompt, get_subword_variation, match_token_length,
                      create_variant_sequence, flatten, add_typo_to_data)

Primary path: loads the vendored data_src/original_data.json (no nltk required).
Optional fallback: from_wordnet() uses nltk if available (lazy import).
"""
from __future__ import annotations

import json
import random
import re
from collections import deque
from copy import deepcopy
from itertools import combinations
from pathlib import Path

__all__ = [
    # typo functions (add_typo.py verbatim)
    "substitute_char",
    "delete_char",
    "insert_char",
    "transpose_chars",
    "keyboard_proximity",
    "proximity_substitute",
    "insert_space",
    "delete_space",
    # utils.py verbatim
    "make_prompt",
    "get_subword_variation",
    "match_token_length",
    "create_variant_sequence",
    "flatten",
    "add_typo_to_data",
    # create_dataset.py
    "rank_list",
    "create_dataset",
    # I/O
    "load_original_data",
    "save_dataset",
    "load_dataset",
]

# ---------------------------------------------------------------------------
# Char-typo functions — verbatim from add_typo.py
# ---------------------------------------------------------------------------

def substitute_char(word: str) -> str:
    index = random.randint(0, len(word) - 1)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    chars = chars.replace(word[index], "")
    random_char = random.choice(chars)
    return word[:index] + random_char + word[index + 1:]


def delete_char(word: str) -> str:
    if len(word) == 1:
        return word
    index = random.randint(0, len(word) - 1)
    return word[:index] + word[index + 1:]


def insert_char(word: str) -> str:
    index_start = 1 if word[0] == " " else 0
    index = random.randint(index_start, len(word))
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    random_char = random.choice(chars)
    return word[:index] + random_char + word[index:]


def transpose_chars(word: str) -> str:
    if len(word) < 2:
        return word
    index = random.randint(0, len(word) - 2)
    return word[:index] + word[index + 1] + word[index] + word[index + 2:]


keyboard_proximity: dict[str, str] = {
    "q": "w",
    "w": "qe",
    "e": "wr",
    "r": "et",
    "t": "ry",
    "y": "tu",
    "u": "yi",
    "i": "uo",
    "o": "ip",
    "p": "o@",
    "a": "s",
    "s": "ad",
    "d": "sf",
    "f": "dg",
    "g": "fh",
    "h": "gj",
    "j": "hk",
    "k": "jl",
    "l": "k;",
    "z": "x",
    "x": "zc",
    "c": "xv",
    "v": "cb",
    "b": "vn",
    "n": "bm",
    "m": "n,",
}


def proximity_substitute(word: str) -> str:
    index = random.randint(0, len(word) - 1)
    char = word[index]
    if char in keyboard_proximity:
        replacement_char = random.choice(keyboard_proximity[char])
        return word[:index] + replacement_char + word[index + 1:]
    return word


def insert_space(word: str) -> str:
    index = random.randint(1, len(word) - 1)
    return word[:index] + " " + word[index + 1:]


def delete_space(word: str) -> str:
    space_index = [i for i, char in enumerate(word) if char == " "]
    delete_index = random.choice(space_index)
    return word[:delete_index] + word[delete_index + 1:]


# ---------------------------------------------------------------------------
# Prompt builder — verbatim from utils.py make_prompt
# ---------------------------------------------------------------------------

def make_prompt(meaning_ids, tokenizer, word=None):
    """Build the QA-prompt input_ids tensor.

    Faithful port of ``make_prompt`` from Tsuji et al. utils.py.

    Parameters
    ----------
    meaning_ids:
        List of token-ids for the meaning string (no special tokens).
    tokenizer:
        HuggingFace tokenizer (or stub with the same interface).
    word:
        If provided, append ``"{word}'"`` as the answer span.

    Returns
    -------
    torch.Tensor of shape (1, seq_len), dtype=torch.long
    """
    import torch

    prompt1 = tokenizer("Q. What is the word that means the following?\n")["input_ids"]
    prompt2 = tokenizer("\nA. That is '", add_special_tokens=False)["input_ids"]
    inputs = prompt1 + meaning_ids + prompt2
    if word is not None:
        output = tokenizer(f"{word}'", add_special_tokens=False)["input_ids"]
        inputs += output
    inputs = torch.tensor([inputs], dtype=torch.long)
    return inputs


# ---------------------------------------------------------------------------
# Subword utilities — verbatim from utils.py
# ---------------------------------------------------------------------------

def flatten(lst: list) -> list:
    """Flatten an arbitrarily nested list.

    Faithful port of ``flatten`` from Tsuji et al. utils.py.
    """
    result = []
    stack = deque([lst])
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
        else:
            result.append(current)
    return result


def get_subword_variation(tokenizer, word: str):
    """Enumerate all valid subword segmentations of ``word``.

    Faithful port of ``get_subword_variation`` from Tsuji et al. utils.py.

    Returns
    -------
    (subword_variation, normal_subword)
        subword_variation: list of lists-of-subword-strings
        normal_subword: tokenizer's default tokenization of ``word``
    """
    split_positions = range(1, len(word))
    subword_variation = []
    vocab = tokenizer.vocab

    normal_subword = tokenizer.tokenize(word)

    test_token = tokenizer.tokenize("aaaabaaaabaaabaaaaabaa")
    prefix = re.sub("a|b", "", test_token[1])
    if " " in word:
        test_token = tokenizer.tokenize(" aa")
        space = test_token[0][0]
        word = re.sub(" ", space, word)

    for i in range(1, len(word)):
        for combination in combinations(split_positions, i):
            subwords = []
            start = 0
            append_flag = True
            for index in combination:
                subword = word[start:index]
                start = index
                if len(subwords) != 0:
                    subword = prefix + subword
                if subword in vocab:
                    subwords.append(subword)
                else:
                    append_flag = False
                    break
            if len(subwords) == 0:
                subwords.append(word)
            else:
                subwords.append(prefix + word[start:])
            if append_flag and subwords[-1] in vocab:
                subword_variation.append(subwords)
    if normal_subword not in subword_variation:
        subword_variation.append(normal_subword)
    return subword_variation, normal_subword


def match_token_length(tokenizer, original_word: str, typo_word: str) -> list:
    """Find a segmentation of ``original_word`` with the same token count as ``typo_word``.

    Faithful port of ``match_token_length`` from Tsuji et al. utils.py.

    Returns a list of subword token strings.
    """
    typo_tokens = tokenizer.tokenize(typo_word)
    original_tokens = tokenizer.tokenize(original_word)

    if len(typo_tokens) == len(original_tokens):
        return original_tokens

    subword_variation, _ = get_subword_variation(tokenizer, original_word)
    candidates = []
    for subwords in subword_variation:
        if len(subwords) == len(typo_tokens):
            test_token = tokenizer.tokenize("aaaabaaaabaaabaaaaabaa")
            prefix = re.sub("a|b", "", test_token[1])
            candidates.append(
                [subwords, min([len(token.replace(prefix, "")) for token in subwords])]
            )
    if len(candidates) != 0:
        candidates = sorted(candidates, key=lambda x: x[1], reverse=False)
        return candidates[-1][0]
    return original_tokens


def create_variant_sequence(tokenizer, original_ids: list, variant_index: list, variant_tokens: list) -> list:
    """Replace tokens at ``variant_index`` with ``variant_tokens`` and flatten.

    Faithful port of ``create_variant_sequence`` from Tsuji et al. utils.py.
    """
    variant_ids = deepcopy(original_ids)
    for v_i, v_t in zip(variant_index, variant_tokens):
        variant_ids[v_i] = tokenizer.convert_tokens_to_ids(v_t)
    return flatten(variant_ids)


# ---------------------------------------------------------------------------
# add_typo_to_data — verbatim from utils.py
# ---------------------------------------------------------------------------

def add_typo_to_data(
    data: dict,
    tokenizer,
    prefix_ids=None,
    typo_type: str | None = None,
    typo_num: int = 1,
    index_range: int = 0,
    use_data_typo: bool = False,
    random_index: bool = False,
) -> dict:
    """Apply a character-level typo to the most important meaning token(s).

    Faithful port of ``add_typo_to_data`` from Tsuji et al. utils.py.

    Parameters
    ----------
    data:
        A dataset entry with keys ``importance_rank``, ``meaning_ids``, ``word``
        (and optionally ``typo_ids``, ``splited_ids``, ``variant_end_index`` if
        ``use_data_typo=True``).
    tokenizer:
        HuggingFace tokenizer (or stub).
    prefix_ids:
        If given, use ``len(prefix_ids)`` as the prompt prefix length.
        Otherwise computed from the standard QA prompt.
    typo_type:
        One of ``"insert_char"`` (currently the only type wired in reference).
    typo_num:
        Number of top-important tokens to corrupt.
    index_range:
        Extra offset added to end-indices (default 0).
    use_data_typo:
        If True, load pre-computed typo from ``data``.
    random_index:
        If True, select tokens randomly instead of by importance rank.

    Returns
    -------
    dict with keys:
        original_ids, typo_ids, splited_ids, word,
        start_index, original_end_index, variant_end_index, word_start_index
    """
    n = min(typo_num, len(data["importance_rank"]))
    important_index = [data["importance_rank"].index(i + 1) for i in range(n)]
    if random_index:
        important_index = random.sample(range(len(data["importance_rank"])), n)
    important_tokens = [
        tokenizer.decode(data["meaning_ids"][i_i]) for i_i in important_index
    ]

    if prefix_ids is not None:
        prefix_length = len(prefix_ids)
    else:
        pre_prompt = tokenizer("Q. What is the word that means the following?\n")[
            "input_ids"
        ]
        prefix_length = len(pre_prompt)

    original_ids = data["meaning_ids"]
    start_index = []
    original_end_index = []

    if typo_num == 0:
        typo_ids = deepcopy(original_ids)
        splited_ids = deepcopy(original_ids)
        variant_end_index = []

    elif use_data_typo:
        typo_ids = data["typo_ids"]
        splited_ids = data["splited_ids"]
        variant_end_index = data["variant_end_index"]

    else:
        if typo_type == "insert_char":
            typo_words = [insert_char(i_t) for i_t in important_tokens]
            typo_tokens = [tokenizer.tokenize(t_w) for t_w in typo_words]
            splited_tokens = [
                match_token_length(tokenizer, i_t, t_w)
                for i_t, t_w in zip(important_tokens, typo_words)
            ]

        typo_ids = create_variant_sequence(
            tokenizer, original_ids, important_index, typo_tokens
        )
        splited_ids = create_variant_sequence(
            tokenizer, original_ids, important_index, splited_tokens
        )

        variant_end_index = []

    for i, i_i in enumerate(important_index):
        start_index.append(prefix_length + i_i)
        original_end_index.append(prefix_length + i_i + 1 + index_range)
        if typo_num == 0:
            variant_end_index.append(prefix_length + i_i + 1 + index_range)
        elif not use_data_typo:
            variant_end_index.append(
                prefix_length + i_i + len(typo_tokens[i]) + index_range
            )

    outputs = {
        "original_ids": original_ids,
        "typo_ids": typo_ids,
        "splited_ids": splited_ids,
        "word": data["word"],
        "start_index": start_index,
        "original_end_index": original_end_index,
        "variant_end_index": variant_end_index,
        "word_start_index": -len(tokenizer(data["word"])) - 1,
    }
    return outputs


# ---------------------------------------------------------------------------
# rank_list — verbatim from create_dataset.py
# ---------------------------------------------------------------------------

def rank_list(lst: list) -> list:
    """Return rank positions (1-based) of each element by descending value.

    Faithful port of ``rank_list`` from Tsuji et al. create_dataset.py.
    """
    sorted_lst = sorted(lst, reverse=True)
    return [sorted_lst.index(x) + 1 for x in lst]


# ---------------------------------------------------------------------------
# create_dataset — faithful port of create_dataset.py
# ---------------------------------------------------------------------------

def create_dataset(llm, original_data: list, threshold: float = 0) -> list:
    """Build the word-identification dataset using an LLM.

    Faithful port of ``create_dataset`` from Tsuji et al. create_dataset.py.

    For each ``{word, meaning}`` entry:
    1. Build the QA prompt via standard tokenizer call.
    2. Call ``llm.generate_word`` — keep only if the generated word matches
       the gold word (case-insensitive).
    3. Call ``llm.get_prob`` — keep only if prob >= threshold.
    4. Call ``llm.get_importance`` over the prompt tokens; slice to the
       meaning-token range to obtain per-token importance scores.
    5. Emit ``{id, word, meaning, prob, meaning_ids, importance, importance_rank}``.

    Parameters
    ----------
    llm:
        An object with ``tokenizer``, ``generate_word(input_ids)``,
        ``get_prob(input_ids, output_ids)``, and
        ``get_importance(input_ids, output_ids)`` methods.
    original_data:
        List of ``{word, meaning}`` dicts (e.g. from ``load_original_data``).
    threshold:
        Minimum probability to include an entry (default 0, i.e. keep all
        correctly-generated words).

    Returns
    -------
    List of dataset entries.
    """
    import torch

    tokenizer = llm.tokenizer

    # Pre-compute prefix length (verbatim from create_dataset.py)
    prompt_ids = tokenizer(
        "Q. What is the word that means the following?\n", return_tensors="pt"
    )["input_ids"][0]

    dataset: list = []
    for original_entry in original_data:
        word = original_entry["word"]
        meaning = original_entry["meaning"]

        prompt = (
            f"Q. What is the word that means the following?\n{meaning}\nA. That is '"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output_word = llm.generate_word(input_ids)

        if word.lower() == output_word.lower():
            output_tokens = tokenizer(
                word, return_tensors="pt", add_special_tokens=False
            )
            output_ids = output_tokens["input_ids"]
            prob = llm.get_prob(input_ids, output_ids)
            if prob >= threshold:
                meaning_tokens = tokenizer(
                    meaning, return_tensors="pt", add_special_tokens=False
                )
                meaning_ids = meaning_tokens["input_ids"][0]
                meaning_range = [
                    prompt_ids.size(0),
                    prompt_ids.size(0) + meaning_ids.size(0),
                ]
                importance = llm.get_importance(input_ids, output_ids)
                importance = importance[meaning_range[0]: meaning_range[1]]
                importance_rank = rank_list(importance)

                data = {
                    "id": len(dataset),
                    "word": word,
                    "meaning": meaning,
                    "prob": prob,
                    "meaning_ids": meaning_ids.tolist(),
                    "importance": importance,
                    "importance_rank": importance_rank,
                }
                dataset.append(data)

    return dataset


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _default_original_data_path() -> Path:
    """Return the default path to the vendored original_data.json."""
    # This file lives at:
    #   projects/quant_typo_neuron/src/quant_typo_neuron/data/wordnet_id.py
    # data_src lives at:
    #   projects/quant_typo_neuron/data_src/original_data.json
    # Path(__file__).parent            → .../data/
    # .parent.parent                   → .../quant_typo_neuron/ (package)
    # .parent.parent.parent            → .../src/
    # .parent.parent.parent.parent     → .../quant_typo_neuron/ (project root)
    return Path(__file__).parent.parent.parent.parent / "data_src" / "original_data.json"


def load_original_data(path=None) -> list:
    """Load the vendored word-definition pairs.

    Parameters
    ----------
    path:
        Path to ``original_data.json``.  Defaults to
        ``projects/quant_typo_neuron/data_src/original_data.json``
        (relative to this file's package root).

    Returns
    -------
    List of ``{word, meaning}`` dicts.
    """
    if path is None:
        path = _default_original_data_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: list, path) -> None:
    """Persist a dataset list to JSON.

    Parameters
    ----------
    dataset:
        List of dataset-entry dicts.
    path:
        Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def load_dataset(path) -> list:
    """Load a previously-saved dataset from JSON.

    Parameters
    ----------
    path:
        Path to the JSON file written by ``save_dataset``.

    Returns
    -------
    List of dataset-entry dicts.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Optional WordNet fallback (no hard dependency on nltk)
# ---------------------------------------------------------------------------

def from_wordnet() -> list:
    """Build original_data entries via NLTK WordNet (lazy import).

    This is a *fallback* — prefer ``load_original_data`` which uses the
    vendored JSON.  Raises ``ImportError`` if nltk is not installed.

    Returns
    -------
    List of ``{word, meaning}`` dicts.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise ImportError(
            "nltk is not installed.  Install it or use load_original_data() "
            "with the vendored data_src/original_data.json instead."
        ) from exc

    # Ensure WordNet data is available
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")

    entries: list = []
    seen: set = set()
    for synset in wn.all_synsets():
        definition = synset.definition()
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ")
            key = (word, definition)
            if key not in seen:
                seen.add(key)
                entries.append({"word": word, "meaning": definition})
    return entries
