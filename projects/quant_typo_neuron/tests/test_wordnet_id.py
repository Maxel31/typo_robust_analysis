"""Tests for quant_typo_neuron.data.wordnet_id (CPU-only, no network).

Covers:
- Char-typo functions: determinism with seeded random, length deltas,
  proximity keyboard map.
- match_token_length: segmentation token-count equality.
- add_typo_to_data: correct span indices, typo on important token,
  split preserves characters.
- create_dataset: filtering by word-match and threshold; schema integrity.
- Vendored data: data_src/original_data.json exists and has 62643 entries.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from quant_typo_neuron.data.wordnet_id import (
    add_typo_to_data,
    delete_char,
    flatten,
    insert_char,
    keyboard_proximity,
    load_original_data,
    match_token_length,
    proximity_substitute,
    rank_list,
    substitute_char,
    transpose_chars,
)

# ---------------------------------------------------------------------------
# Stub tokenizer — minimal interface used by wordnet_id functions
# ---------------------------------------------------------------------------

class StubTokenizer:
    """Deterministic toy tokenizer for CPU/no-network tests.

    Tokenize splits a word into individual characters.
    Token IDs: ord(char) % 256.
    ``vocab`` contains single lowercase chars and a handful of digraphs so
    that get_subword_variation can find valid segmentations.

    The interface satisfied:
        tokenize(word)                 → list[str]
        __call__(text, ...)            → {"input_ids": list[int]}
        decode(token_id)               → str
        convert_tokens_to_ids(token)   → int (or list[int])
        vocab                          → dict[str, int]
    """

    _chars = list("abcdefghijklmnopqrstuvwxyz ")
    _digraphs = ["ca", "do", "at", "an", "he", "lo"]
    _all_tokens = _chars + _digraphs
    vocab: dict[str, int] = {t: i for i, t in enumerate(_all_tokens)}

    def tokenize(self, word: str) -> list[str]:
        """Split word into individual chars (char-level tokenization)."""
        return list(word)

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors=None):
        """Simulate tokenizer(text)["input_ids"].

        When return_tensors="pt" the value is a torch.Tensor so that
        callers using .size() and indexing work correctly.
        """
        ids = [ord(c) % 256 for c in text]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, token_id) -> str:
        """Map a single integer token-id back to a character."""
        return chr(int(token_id) % 256)

    def convert_tokens_to_ids(self, token):
        """Convert token string(s) to id(s)."""
        if isinstance(token, list):
            return [self.vocab.get(t, ord(t[0]) % 256 if t else 0) for t in token]
        return self.vocab.get(token, ord(token[0]) % 256 if token else 0)


# ---------------------------------------------------------------------------
# Patched create_dataset helper — injects stub context before generate_word
# ---------------------------------------------------------------------------

def _patched_create_dataset(stub_llm, original_data, threshold=0):
    """Run create_dataset logic with a StubLLM that needs its context set.

    Because create_dataset calls tokenizer(...) to build prompts and the stub
    LLM must know which (word, meaning) pair it is answering, we replicate
    the reference logic here while calling ``stub_llm._set_context`` before
    each ``generate_word`` call.
    """
    import torch
    from quant_typo_neuron.data.wordnet_id import rank_list

    tokenizer = stub_llm.tokenizer
    prompt_ids = tokenizer(
        "Q. What is the word that means the following?\n", return_tensors="pt"
    )["input_ids"][0]

    dataset: list = []
    for entry in original_data:
        word = entry["word"]
        meaning = entry["meaning"]
        stub_llm._set_context(word, meaning)

        prompt = (
            f"Q. What is the word that means the following?\n{meaning}\nA. That is '"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output_word = stub_llm.generate_word(input_ids)

        if word.lower() == output_word.lower():
            output_tokens = tokenizer(word, return_tensors="pt", add_special_tokens=False)
            output_ids = output_tokens["input_ids"]
            prob = stub_llm.get_prob(input_ids, output_ids)
            if prob >= threshold:
                meaning_tokens = tokenizer(
                    meaning, return_tensors="pt", add_special_tokens=False
                )
                meaning_ids = meaning_tokens["input_ids"][0]
                meaning_range = [
                    prompt_ids.size(0),
                    prompt_ids.size(0) + meaning_ids.size(0),
                ]
                importance = stub_llm.get_importance(input_ids, output_ids)
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
# Stub LLM
# ---------------------------------------------------------------------------

class StubLLM:
    """Stub LLM with canned per-(word,meaning) responses."""

    def __init__(self, tokenizer, word_map: dict, prob_map: dict):
        self.tokenizer = tokenizer
        self._word_map = word_map
        self._prob_map = prob_map
        self._last_key: tuple[str, str] = ("", "")

    def _set_context(self, word: str, meaning: str) -> None:
        self._last_key = (word, meaning)

    def generate_word(self, input_ids) -> str:
        return self._word_map.get(self._last_key, "")

    def get_prob(self, input_ids, output_ids) -> float:
        return self._prob_map.get(self._last_key, 0.0)

    def get_importance(self, input_ids, output_ids) -> list:
        """Return a list covering the full input; last positions have non-zero values."""
        try:
            n = input_ids.size(1)
        except AttributeError:
            n = len(input_ids[0])
        scores = [0.0] * n
        for i in range(min(5, n)):
            scores[n - 1 - i] = float(i + 1)
        return scores


# ===========================================================================
# Tests: substitute_char
# ===========================================================================

class TestSubstituteChar:
    def test_same_length(self):
        random.seed(0)
        word = "hello"
        assert len(substitute_char(word)) == len(word)

    def test_char_changed(self):
        random.seed(0)
        assert substitute_char("hello") != "hello"

    def test_determinism(self):
        random.seed(42)
        r1 = substitute_char("world")
        random.seed(42)
        r2 = substitute_char("world")
        assert r1 == r2

    def test_exactly_one_char_differs(self):
        random.seed(7)
        word = "abcdef"
        result = substitute_char(word)
        diff = sum(1 for a, b in zip(word, result) if a != b)
        assert diff == 1


# ===========================================================================
# Tests: delete_char
# ===========================================================================

class TestDeleteChar:
    def test_shorter_by_one(self):
        random.seed(0)
        word = "hello"
        assert len(delete_char(word)) == len(word) - 1

    def test_single_char_unchanged(self):
        assert delete_char("a") == "a"

    def test_determinism(self):
        random.seed(99)
        r1 = delete_char("python")
        random.seed(99)
        r2 = delete_char("python")
        assert r1 == r2


# ===========================================================================
# Tests: insert_char
# ===========================================================================

class TestInsertChar:
    def test_longer_by_one(self):
        random.seed(0)
        assert len(insert_char("hello")) == len("hello") + 1

    def test_space_prefix_not_at_position_zero(self):
        """When word starts with ' ', insertion must not go at index 0."""
        for seed in range(30):
            random.seed(seed)
            result = insert_char(" hello")
            assert result[0] == " ", (
                f"seed={seed}: leading space lost, got '{result}'"
            )

    def test_determinism(self):
        random.seed(13)
        r1 = insert_char("test")
        random.seed(13)
        r2 = insert_char("test")
        assert r1 == r2


# ===========================================================================
# Tests: transpose_chars
# ===========================================================================

class TestTransposeChars:
    def test_same_length(self):
        random.seed(0)
        assert len(transpose_chars("hello")) == len("hello")

    def test_same_multiset_of_chars(self):
        random.seed(0)
        word = "hello"
        result = transpose_chars(word)
        assert sorted(result) == sorted(word)

    def test_single_char_unchanged(self):
        assert transpose_chars("a") == "a"

    def test_determinism(self):
        random.seed(5)
        r1 = transpose_chars("abcde")
        random.seed(5)
        r2 = transpose_chars("abcde")
        assert r1 == r2


# ===========================================================================
# Tests: proximity_substitute
# ===========================================================================

class TestProximitySubstitute:
    def test_same_length(self):
        random.seed(0)
        assert len(proximity_substitute("hello")) == len("hello")

    def test_uses_keyboard_map(self):
        """When a changed char is found, replacement must be in proximity map."""
        for seed in range(30):
            random.seed(seed)
            word = "qwerty"
            result = proximity_substitute(word)
            changed = [i for i, (a, b) in enumerate(zip(word, result)) if a != b]
            if changed:
                idx = changed[0]
                orig = word[idx]
                new = result[idx]
                if orig in keyboard_proximity:
                    assert new in keyboard_proximity[orig], (
                        f"seed={seed}: '{new}' not in proximity['{orig}']={keyboard_proximity[orig]}"
                    )

    def test_no_map_char_unchanged(self):
        """Digit chars are not in keyboard_proximity — word returned unchanged."""
        word = "1234"
        result = proximity_substitute(word)
        assert result == word


# ===========================================================================
# Tests: flatten
# ===========================================================================

class TestFlatten:
    def test_flat_list(self):
        assert flatten([1, 2, 3]) == [1, 2, 3]

    def test_nested(self):
        assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]

    def test_deeply_nested(self):
        assert flatten([[[1]], [2, [3]]]) == [1, 2, 3]

    def test_empty(self):
        assert flatten([]) == []


# ===========================================================================
# Tests: rank_list
# ===========================================================================

class TestRankList:
    def test_descending_rank(self):
        # [3, 1, 2]: 3 is highest → rank 1; 1 is lowest → rank 3; 2 → rank 2
        assert rank_list([3, 1, 2]) == [1, 3, 2]

    def test_all_equal(self):
        result = rank_list([5, 5, 5])
        assert result == [1, 1, 1]

    def test_single_element(self):
        assert rank_list([7]) == [1]


# ===========================================================================
# Tests: match_token_length (with StubTokenizer)
# ===========================================================================

class TestMatchTokenLength:
    """Tests using StubTokenizer (char-level tokenization).

    NOTE: With the stub tokenizer both words tokenize to individual chars so
    lengths almost always match already (happy path). The primary assertion
    is therefore that the *returned* segmentation has a length within the
    expected range, i.e. either the typo length (ideal) or the original
    length (fallback). This is a clearly-marked lighter assertion because
    faithfully exercising the subword-variation search requires a real
    SentencePiece/BPE tokenizer with a multi-char vocab.
    """

    def setup_method(self):
        self.tok = StubTokenizer()

    def test_same_length_returns_original_tokens(self):
        # Both "cat" and "cxt" have 3 chars → same token count
        result = match_token_length(self.tok, "cat", "cxt")
        assert len(result) == len(self.tok.tokenize("cxt"))

    def test_result_length_in_expected_range(self):
        # LIGHTER ASSERTION (stub tokenizer): result length is either the
        # typo token count (best) or the original token count (fallback).
        original_word = "hello"
        typo_word = "helo"
        result = match_token_length(self.tok, original_word, typo_word)
        orig_len = len(self.tok.tokenize(original_word))
        typo_len = len(self.tok.tokenize(typo_word))
        assert len(result) in (orig_len, typo_len), (
            f"result len {len(result)} not in ({orig_len}, {typo_len})"
        )

    def test_returns_list_of_strings(self):
        result = match_token_length(self.tok, "cat", "cat")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)


# ===========================================================================
# Tests: add_typo_to_data (with StubTokenizer)
# ===========================================================================

class TestAddTypoToData:
    """Tests using StubTokenizer for CPU/no-network operation."""

    def setup_method(self):
        self.tok = StubTokenizer()

    def _make_data(self, word="cat", meaning="animal"):
        meaning_ids = [ord(c) % 256 for c in meaning]
        importance_rank = list(range(1, len(meaning_ids) + 1))
        return {
            "word": word,
            "meaning": meaning,
            "meaning_ids": meaning_ids,
            "importance": [float(i) for i in reversed(importance_rank)],
            "importance_rank": importance_rank,
        }

    def test_output_keys(self):
        random.seed(0)
        data = self._make_data()
        result = add_typo_to_data(data, self.tok, typo_type="insert_char", typo_num=1)
        expected_keys = {
            "original_ids", "typo_ids", "splited_ids", "word",
            "start_index", "original_end_index", "variant_end_index",
            "word_start_index",
        }
        assert set(result.keys()) == expected_keys

    def test_original_ids_preserved(self):
        random.seed(0)
        data = self._make_data(meaning="dog")
        result = add_typo_to_data(data, self.tok, typo_type="insert_char", typo_num=1)
        assert result["original_ids"] == data["meaning_ids"]

    def test_typo_ids_length_ge_original(self):
        """insert_char adds one char so typo_ids must be at least as long as original."""
        random.seed(0)
        data = self._make_data(meaning="dog")
        original_len = len(data["meaning_ids"])
        result = add_typo_to_data(data, self.tok, typo_type="insert_char", typo_num=1)
        assert len(result["typo_ids"]) >= original_len

    def test_start_index_points_to_important_token(self):
        """start_index[0] = prefix_length + position-of-rank-1-token."""
        random.seed(0)
        meaning = "animal"
        meaning_ids = [ord(c) % 256 for c in meaning]
        n = len(meaning_ids)
        # Make the last token most important (rank=1 at index n-1)
        importance_rank = list(range(2, n + 1)) + [1]
        data = {
            "word": "cat",
            "meaning": meaning,
            "meaning_ids": meaning_ids,
            "importance": [float(i) for i in reversed(range(n))],
            "importance_rank": importance_rank,
        }
        prefix_ids = list(range(10))  # fixed prefix of length 10
        result = add_typo_to_data(
            data, self.tok,
            prefix_ids=prefix_ids,
            typo_type="insert_char",
            typo_num=1,
        )
        expected_start = 10 + (n - 1)
        assert result["start_index"][0] == expected_start

    def test_typo_num_zero_is_identity(self):
        """typo_num=0 must return unchanged ids and empty variant_end_index."""
        data = self._make_data(meaning="cat")
        result = add_typo_to_data(data, self.tok, typo_type="insert_char", typo_num=0)
        assert result["typo_ids"] == result["original_ids"]
        assert result["splited_ids"] == result["original_ids"]
        assert result["variant_end_index"] == []

    def test_word_field_preserved(self):
        random.seed(0)
        data = self._make_data(word="lion", meaning="big cat")
        result = add_typo_to_data(data, self.tok, typo_type="insert_char", typo_num=1)
        assert result["word"] == "lion"


# ===========================================================================
# Tests: create_dataset (stub LLM + stub tokenizer)
# ===========================================================================

class TestCreateDataset:
    """Tests using StubLLM + StubTokenizer; no real model or network."""

    def setup_method(self):
        self.tok = StubTokenizer()

    def _llm(self, word_map, prob_map):
        return StubLLM(self.tok, word_map, prob_map)

    def test_only_correct_words_kept(self):
        """Mismatch between gold word and generated word must exclude entry."""
        original_data = [
            {"word": "cat", "meaning": "animal"},
            {"word": "dog", "meaning": "canine"},
        ]
        word_map = {("cat", "animal"): "cat", ("dog", "canine"): "bird"}
        prob_map = {("cat", "animal"): 0.9, ("dog", "canine"): 0.5}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        assert len(result) == 1
        assert result[0]["word"] == "cat"

    def test_threshold_filtering(self):
        """Entries with prob < threshold must be excluded."""
        original_data = [
            {"word": "cat", "meaning": "animal"},
            {"word": "ant", "meaning": "insect"},
        ]
        word_map = {("cat", "animal"): "cat", ("ant", "insect"): "ant"}
        prob_map = {("cat", "animal"): 0.9, ("ant", "insect"): 0.05}
        result = _patched_create_dataset(
            self._llm(word_map, prob_map), original_data, threshold=0.5
        )
        assert len(result) == 1
        assert result[0]["word"] == "cat"

    def test_output_schema(self):
        """Output entries must have the full reference schema."""
        original_data = [{"word": "cat", "meaning": "animal"}]
        word_map = {("cat", "animal"): "cat"}
        prob_map = {("cat", "animal"): 0.8}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        assert len(result) == 1
        required_keys = {"id", "word", "meaning", "prob", "meaning_ids", "importance", "importance_rank"}
        assert required_keys.issubset(set(result[0].keys()))

    def test_id_is_sequential(self):
        """id must be 0-based sequential across passing entries."""
        original_data = [
            {"word": "cat", "meaning": "animal"},
            {"word": "ant", "meaning": "insect"},
        ]
        word_map = {("cat", "animal"): "cat", ("ant", "insect"): "ant"}
        prob_map = {("cat", "animal"): 0.9, ("ant", "insect"): 0.8}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1

    def test_case_insensitive_match(self):
        """Word comparison must be case-insensitive."""
        original_data = [{"word": "Cat", "meaning": "animal"}]
        word_map = {("Cat", "animal"): "cat"}  # lowercase from model
        prob_map = {("Cat", "animal"): 0.7}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        assert len(result) == 1

    def test_prob_stored_correctly(self):
        """prob field must store the exact value returned by get_prob."""
        original_data = [{"word": "cat", "meaning": "animal"}]
        word_map = {("cat", "animal"): "cat"}
        prob_map = {("cat", "animal"): 0.123}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        assert abs(result[0]["prob"] - 0.123) < 1e-9

    def test_importance_rank_length_equals_meaning_ids(self):
        """importance_rank must have the same length as meaning_ids."""
        original_data = [{"word": "cat", "meaning": "animal"}]
        word_map = {("cat", "animal"): "cat"}
        prob_map = {("cat", "animal"): 0.9}
        result = _patched_create_dataset(self._llm(word_map, prob_map), original_data)
        entry = result[0]
        assert len(entry["importance_rank"]) == len(entry["meaning_ids"])

    def test_three_entries_two_match_one_below_threshold(self):
        """Smoke test: 3 entries, 2 match word, 1 of those below threshold."""
        original_data = [
            {"word": "cat", "meaning": "animal"},
            {"word": "dog", "meaning": "canine"},
            {"word": "ant", "meaning": "insect"},
        ]
        word_map = {
            ("cat", "animal"): "cat",    # match
            ("dog", "canine"): "bird",   # mismatch
            ("ant", "insect"): "ant",    # match but below threshold
        }
        prob_map = {
            ("cat", "animal"): 0.9,
            ("dog", "canine"): 0.5,
            ("ant", "insect"): 0.01,
        }
        result = _patched_create_dataset(
            self._llm(word_map, prob_map), original_data, threshold=0.5
        )
        assert len(result) == 1
        assert result[0]["word"] == "cat"


class _SequentialLLM:
    """Context-free stub: returns canned responses in original_data order.

    Unlike StubLLM (which needs ``_set_context``), this works with the *real*
    ``create_dataset`` because ``create_dataset`` iterates ``original_data``
    sequentially and calls ``generate_word`` exactly once per entry, in order.
    Used to exercise the real loop (and its tqdm progress path).
    """

    def __init__(self, tokenizer, words):
        self.tokenizer = tokenizer
        self._words = list(words)
        self._gi = 0

    def generate_word(self, input_ids) -> str:
        w = self._words[self._gi]
        self._gi += 1
        return w

    def get_prob(self, input_ids, output_ids) -> float:
        return 0.9

    def get_importance(self, input_ids, output_ids) -> list:
        try:
            n = input_ids.size(1)
        except AttributeError:
            n = len(input_ids[0])
        return [float(i) for i in range(n)]


class TestCreateDatasetProgress:
    """Exercise the real create_dataset loop and the tqdm progress option."""

    def setup_method(self):
        self.tok = StubTokenizer()
        self.original_data = [
            {"word": "cat", "meaning": "animal"},
            {"word": "dog", "meaning": "canine"},
        ]

    def test_progress_true_and_false_give_identical_results(self):
        """The progress flag must not affect the produced dataset."""
        from quant_typo_neuron.data.wordnet_id import create_dataset

        with_bar = create_dataset(
            _SequentialLLM(self.tok, ["cat", "dog"]), self.original_data, progress=True
        )
        without_bar = create_dataset(
            _SequentialLLM(self.tok, ["cat", "dog"]), self.original_data, progress=False
        )
        assert with_bar == without_bar
        assert [e["word"] for e in with_bar] == ["cat", "dog"]
        assert [e["id"] for e in with_bar] == [0, 1]

    def test_progress_default_is_on_and_runs(self):
        """Default call (progress omitted) must run and keep correct entries."""
        from quant_typo_neuron.data.wordnet_id import create_dataset

        result = create_dataset(
            _SequentialLLM(self.tok, ["cat", "dog"]), self.original_data
        )
        assert len(result) == 2


# ===========================================================================
# Tests: vendored data file
# ===========================================================================

class TestVendoredData:
    """Verify the vendored data_src/original_data.json is present and correct."""

    _data_path = Path(__file__).parent.parent / "data_src" / "original_data.json"

    def test_file_exists(self):
        assert self._data_path.exists(), (
            f"Vendored data not found at {self._data_path}"
        )

    def test_entry_count_is_62643(self):
        with open(self._data_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 62643, f"Expected 62643 entries, got {len(data)}"

    def test_schema_spot_check(self):
        """First 100 entries must each have string 'word' and 'meaning' fields."""
        with open(self._data_path, encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data[:100]):
            assert "word" in entry, f"Entry {i} missing 'word'"
            assert "meaning" in entry, f"Entry {i} missing 'meaning'"
            assert isinstance(entry["word"], str)
            assert isinstance(entry["meaning"], str)

    def test_load_original_data_default_path(self):
        """load_original_data() with no args must load 62643 entries."""
        data = load_original_data()
        assert len(data) == 62643
