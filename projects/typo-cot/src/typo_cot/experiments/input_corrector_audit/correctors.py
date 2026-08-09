"""The three input correctors used by the Appendix E audit."""

from __future__ import annotations

import gc
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CORRECTOR_IDS = (
    "pyspellchecker",
    "t5-large-spell",
    "qwen2.5-7b-instruct",
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CORRECTED_TAG_RE = re.compile(r"<corrected>(.*?)</corrected>", re.DOTALL)

QWEN_SYSTEM_PROMPT = "You are a careful and conservative proofreader."
QWEN_USER_TEMPLATE = (
    "The following text may contain typographical errors (misspelled words). "
    "Fix ONLY the typos. Do not rephrase, do not add or remove words, and do not "
    "change punctuation, numbers, formatting, line breaks, or any correctly "
    "spelled words. If there are no typos, reproduce the text exactly as given. "
    "Output ONLY the corrected text, wrapped between <corrected> and "
    "</corrected> tags.\n\n<text>\n{text}\n</text>"
)
QWEN_RETRY_REMINDER = (
    "\n\nReminder: your previous reply did not follow the required format. "
    "Reply with ONLY the corrected text wrapped between <corrected> and "
    "</corrected> tags, and nothing else."
)


def apply_case(template: str, corrected: str) -> str:
    """Apply the submitted pyspellchecker case-preservation rule."""
    if template.isupper():
        return corrected.upper()
    if template[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


class PySpellCheckerCorrector:
    """Deterministic wrapper around pyspellchecker 0.9.0."""

    corrector_id = "pyspellchecker"
    dependency_requirement = "pyspellchecker==0.9.0"

    def __init__(self, *, spellchecker: object | None = None, language: str = "en") -> None:
        if spellchecker is None:
            from spellchecker import SpellChecker

            spellchecker = SpellChecker(language=language)
        self._spellchecker = spellchecker

    def _correction(self, word: str) -> str | None:
        candidates = self._spellchecker.candidates(word)  # type: ignore[attr-defined]
        if not candidates:
            return None
        highest_frequency = max(self._spellchecker[candidate] for candidate in candidates)  # type: ignore[index]
        return min(
            candidate
            for candidate in candidates
            if self._spellchecker[candidate] == highest_frequency  # type: ignore[index]
        )

    def correct_with_changes(self, text: str) -> tuple[str, list[dict[str, object]]]:
        output: list[str] = []
        changes: list[dict[str, object]] = []
        previous_end = 0
        for match in WORD_RE.finditer(text):
            word = match.group(0)
            output.append(text[previous_end : match.start()])
            previous_end = match.end()
            lowercase = word.lower()
            if len(word) <= 1 or lowercase in self._spellchecker:  # type: ignore[operator]
                output.append(word)
                continue
            candidate = self._correction(lowercase)
            if candidate is None or candidate == lowercase:
                output.append(word)
                continue
            corrected = apply_case(word, candidate)
            changes.append(
                {
                    "original": word,
                    "corrected": corrected,
                    "start": match.start(),
                }
            )
            output.append(corrected)
        output.append(text[previous_end:])
        return "".join(output), changes

    def correct(self, text: str) -> str:
        return self.correct_with_changes(text)[0]

    def close(self) -> None:
        """Match the neural-corrector lifecycle interface."""


class T5LargeSpellCorrector:
    """Line-preserving wrapper around ``ai-forever/T5-large-spell``."""

    corrector_id = "t5-large-spell"
    model_id = "ai-forever/T5-large-spell"
    prefix = "grammar: "
    max_input_length = 512
    max_new_tokens = 256

    def __init__(
        self,
        *,
        generate_fn: Callable[..., str] | None = None,
        revision: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.revision = revision
        self.device = device
        self._generate_fn = generate_fn
        self._model: object | None = None
        self._tokenizer: object | None = None

    def _load(self) -> Callable[..., str]:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        revision = {"revision": self.revision} if self.revision is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **revision).to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

        def generate(prompt: str, **generation: object) -> str:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generate

    @property
    def generate_fn(self) -> Callable[..., str]:
        if self._generate_fn is None:
            self._generate_fn = self._load()
        return self._generate_fn

    def correct(self, text: str) -> str:
        output: list[str] = []
        for line in text.split("\n"):
            if not line.strip():
                output.append(line)
                continue
            generated = self.generate_fn(
                self.prefix + line,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            if not isinstance(generated, str):
                raise TypeError("T5 corrector generation must return text")
            output.append(generated.strip())
        return "\n".join(output)

    def resolved_revision(self) -> str | None:
        config = getattr(self._model, "config", None)
        value = getattr(config, "_commit_hash", None)
        return value if isinstance(value, str) else None

    def close(self) -> None:
        self._generate_fn = None
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class QwenCorrector:
    """Conservative tagged-output Qwen2.5 corrector with one retry."""

    corrector_id = "qwen2.5-7b-instruct"
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens = 1024

    def __init__(
        self,
        *,
        generate_fn: Callable[..., str] | None = None,
        revision: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.revision = revision
        self.device = device
        self._generate_fn = generate_fn
        self._model: object | None = None
        self._tokenizer: object | None = None

    def _load(self) -> Callable[..., str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        revision = {"revision": self.revision} if self.revision is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            **revision,
        ).to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

        def generate(
            messages: Sequence[Mapping[str, str]],
            **generation: object,
        ) -> str:
            inputs = tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    pad_token_id=pad_token_id,
                    **generation,
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            return tokenizer.decode(
                output_ids[0][prompt_length:],
                skip_special_tokens=True,
            )

        return generate

    @property
    def generate_fn(self) -> Callable[..., str]:
        if self._generate_fn is None:
            self._generate_fn = self._load()
        return self._generate_fn

    @staticmethod
    def _parse(response: str) -> str | None:
        match = _CORRECTED_TAG_RE.search(response)
        return None if match is None else match.group(1).strip("\n")

    def _messages(self, content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def correct_with_meta(self, text: str) -> tuple[str, dict[str, object]]:
        prompt = QWEN_USER_TEMPLATE.format(text=text)
        response = self.generate_fn(
            self._messages(prompt),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        if not isinstance(response, str):
            raise TypeError("Qwen corrector generation must return text")
        parsed = self._parse(response)
        calls = 1
        if parsed is None:
            response = self.generate_fn(
                self._messages(prompt + QWEN_RETRY_REMINDER),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            if not isinstance(response, str):
                raise TypeError("Qwen corrector generation must return text")
            parsed = self._parse(response)
            calls = 2
        if parsed is None:
            return text, {
                "parse_failed": True,
                "n_calls": calls,
                "raw_response": response,
            }
        return parsed, {
            "parse_failed": False,
            "n_calls": calls,
            "raw_response": response,
        }

    def correct(self, text: str) -> str:
        return self.correct_with_meta(text)[0]

    def resolved_revision(self) -> str | None:
        config = getattr(self._model, "config", None)
        value = getattr(config, "_commit_hash", None)
        return value if isinstance(value, str) else None

    def close(self) -> None:
        self._generate_fn = None
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_corrector(corrector_id: str, **kwargs: Any) -> object:
    """Construct one public corrector without importing neural dependencies eagerly."""
    if corrector_id == "pyspellchecker":
        return PySpellCheckerCorrector(**kwargs)
    if corrector_id == "t5-large-spell":
        return T5LargeSpellCorrector(**kwargs)
    if corrector_id == "qwen2.5-7b-instruct":
        return QwenCorrector(**kwargs)
    raise ValueError(f"unknown-corrector: {corrector_id!r}; choose from {CORRECTOR_IDS}")
