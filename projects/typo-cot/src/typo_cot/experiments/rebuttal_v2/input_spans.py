"""Outcome-independent, exact-codepoint edit spans and diagnostic risk flags.

The representative Levenshtein path uses diagonal, deletion, then insertion
priority while tracing backwards. Multiple optimal paths are never certified as
an independent alignment, even if their representative happens to match archive
coordinates. All bounds and segmentation choices are part of the audit identity.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import re
import unicodedata


DIFF_VERSION = "codepoint-levenshtein-unit-backtrace-diagonal-delete-insert/v1"
WORD_VERSION = "unicode-lnm-internal-apostrophe/v1"
RISK_VERSION = "explicit-layout-and-english-diagnostic-lexicons/v1"
MAX_DIFF_CELLS = 4_000_000
MAX_TEXT_CODEPOINTS = 10_000

RISK_NAMES = (
    "stem",
    "option_content",
    "option_label",
    "gold_option",
    "number",
    "quantity",
    "unit",
    "negation",
    "entity",
)
QUANTITY_WORDS = frozenset(
    "all any both each either enough every few fewer half least less many more most much "
    "neither no none one only several some twice zero two three four five six seven eight "
    "nine ten hundred thousand million billion dozen total remaining remainder".split()
)
UNIT_WORDS = frozenset(
    "m cm mm km g kg mg l ml meter meters metre metres centimeter centimeters "
    "centimetre centimetres millimeter millimeters kilometer kilometers kilometre "
    "kilometres gram grams kilogram kilograms liter liters litre litres second seconds "
    "minute minutes hour hours day days week weeks month months year years dollar dollars "
    "cent cents percent percentage inch inches foot feet yard yards mile miles pound "
    "pounds ounce ounces gallon gallons degree degrees celsius fahrenheit".split()
)
NEGATION_WORDS = frozenset(
    "no not never none neither nor without cannot can't couldn't don't doesn't didn't "
    "isn't aren't wasn't weren't won't wouldn't shouldn't mustn't hasn't haven't hadn't".split()
)
UNIT_SYMBOLS = frozenset("%$€£¥°")
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:[eE][+-]?\d+)?")
REGION_KINDS = frozenset(
    {"stem", "option-content", "option-label", "instructions", "context", "entity"}
)


@dataclass(frozen=True)
class SpanAudit:
    """Exact query-relative half-open spans; path count two means two or more."""

    diff_status: str
    edit_distance: int | None
    optimal_path_count: int
    edit_spans: list[dict[str, int]]
    changed_words: dict[str, list[list[int]]]
    aligned_words: list[dict[str, object]]
    alignment_status: str
    reasons: list[str]
    algorithm_versions: dict[str, str | int]


def _versions() -> dict[str, str | int]:
    return {
        "diff": DIFF_VERSION,
        "word_segmentation": WORD_VERSION,
        "unicode_database": unicodedata.unidata_version,
        "max_diff_cells": MAX_DIFF_CELLS,
        "max_text_codepoints": MAX_TEXT_CODEPOINTS,
    }


def _word_character(character: str) -> bool:
    return unicodedata.category(character)[0] in "LNM"


def word_spans(text: str) -> list[list[int]]:
    """Maximal L/N/M runs with internal ' or ’; a run needs a letter/number.

    No NFC/NFKC or case conversion is applied to the text or its offsets.
    Leading/trailing combining marks are retained when attached to a run;
    combining-mark-only runs do not constitute a word.
    """

    words: list[list[int]] = []
    start: int | None = None
    has_base = False
    for index, character in enumerate(text):
        category = unicodedata.category(character)[0]
        internal_apostrophe = (
            character in {"'", "’"}
            and index > 0
            and index + 1 < len(text)
            and _word_character(text[index - 1])
            and _word_character(text[index + 1])
        )
        if category in "LNM" or internal_apostrophe:
            if start is None:
                start = index
            has_base = has_base or category in "LN"
        else:
            if start is not None and has_base:
                words.append([start, index])
            start, has_base = None, False
    if start is not None and has_base:
        words.append([start, len(text)])
    return words


def _affected_words(words: list[list[int]], start: int, end: int) -> list[tuple[int, int]]:
    if start < end:
        return [(left, right) for left, right in words if left < end and start < right]
    interior = [(left, right) for left, right in words if left < start < right]
    if interior:
        return interior
    return [(left, right) for left, right in words if start in {left, right}]


def _align_words(
    clean: str, typo: str, edits: list[dict[str, int]]
) -> tuple[dict[str, list[list[int]]], list[dict[str, object]], list[str]]:
    words = {"clean": word_spans(clean), "typo": word_spans(typo)}
    changed: dict[str, set[tuple[int, int]]] = {"clean": set(), "typo": set()}
    pairs: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    reasons: set[str] = set()
    for index, edit in enumerate(edits):
        candidates = {
            side: _affected_words(words[side], edit[f"{side}_start"], edit[f"{side}_end"])
            for side in ("clean", "typo")
        }
        for side in candidates:
            changed[side].update(candidates[side])
        if any(len(candidates[side]) != 1 for side in candidates):
            reasons.add("edited_word_pair_not_unique")
            if any(edit[f"{side}_start"] == edit[f"{side}_end"] for side in candidates):
                reasons.add("zero_width_edit_word_boundary_unresolved")
            continue
        key = (candidates["clean"][0], candidates["typo"][0])
        pairs.setdefault(key, []).append(index)
    clean_mapping: dict[tuple[int, int], set[tuple[int, int]]] = {}
    typo_mapping: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for left, right in pairs:
        clean_mapping.setdefault(left, set()).add(right)
        typo_mapping.setdefault(right, set()).add(left)
    if any(len(values) != 1 for values in (*clean_mapping.values(), *typo_mapping.values())):
        reasons.add("edited_word_pair_not_one_to_one")
    aligned = [
        {"clean_span": list(left), "typo_span": list(right), "edit_indices": indices}
        for (left, right), indices in sorted(pairs.items())
    ]
    return (
        {side: [list(span) for span in sorted(values)] for side, values in changed.items()},
        aligned,
        sorted(reasons),
    )


def diff_actual_spans(clean: str, typo: str) -> SpanAudit:
    """Compute every disjoint representative edit without archive/outcome inputs."""

    if not isinstance(clean, str) or not isinstance(typo, str):
        raise ValueError("diff_actual_spans requires exact clean and typo strings")
    versions = _versions()
    empty_words: dict[str, list[list[int]]] = {"clean": [], "typo": []}
    if max(len(clean), len(typo)) > MAX_TEXT_CODEPOINTS:
        return SpanAudit(
            "unavailable",
            None,
            0,
            [],
            empty_words,
            [],
            "unavailable",
            ["diff_text_resource_limit"],
            versions,
        )
    # Equality is exact and does not require a quadratic allocation.
    if clean == typo:
        return SpanAudit(
            "no_change",
            0,
            1,
            [],
            empty_words,
            [],
            "no_change",
            ["no_net_change", "cancellation_not_identifiable_from_texts"],
            versions,
        )
    width = len(typo) + 1
    if (len(clean) + 1) * width > MAX_DIFF_CELLS:
        return SpanAudit(
            "unavailable",
            None,
            0,
            [],
            empty_words,
            [],
            "unavailable",
            ["diff_matrix_resource_limit"],
            versions,
        )
    costs = [array("I", range(width))]
    previous_counts = bytearray([1]) * width
    for row, clean_character in enumerate(clean, start=1):
        previous = costs[-1]
        current = array("I", [row])
        counts = bytearray(width)
        counts[0] = 1
        for column, typo_character in enumerate(typo, start=1):
            diagonal = previous[column - 1] + (clean_character != typo_character)
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            best = min(diagonal, deletion, insertion)
            current.append(best)
            counts[column] = min(
                2,
                (previous_counts[column - 1] if diagonal == best else 0)
                + (previous_counts[column] if deletion == best else 0)
                + (counts[column - 1] if insertion == best else 0),
            )
        costs.append(current)
        previous_counts = counts
    row, column = len(clean), len(typo)
    backwards: list[tuple[int, int, int, int, bool]] = []
    while row or column:
        if (
            row
            and column
            and costs[row][column]
            == (costs[row - 1][column - 1] + (clean[row - 1] != typo[column - 1]))
        ):
            backwards.append((row - 1, row, column - 1, column, clean[row - 1] == typo[column - 1]))
            row, column = row - 1, column - 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            backwards.append((row - 1, row, column, column, False))
            row -= 1
        else:
            backwards.append((row, row, column - 1, column, False))
            column -= 1
    edits: list[dict[str, int]] = []
    active: dict[str, int] | None = None
    for clean_start, clean_end, typo_start, typo_end, equal in reversed(backwards):
        if equal:
            active = None
        elif active is None:
            active = {
                "clean_start": clean_start,
                "clean_end": clean_end,
                "typo_start": typo_start,
                "typo_end": typo_end,
            }
            edits.append(active)
        else:
            active["clean_end"], active["typo_end"] = clean_end, typo_end
    changed, aligned, reasons = _align_words(clean, typo, edits)
    ambiguous = previous_counts[-1] > 1
    if ambiguous:
        reasons.append("multiple_optimal_diff_paths")
    return SpanAudit(
        "ambiguous" if ambiguous else "changed",
        costs[-1][-1],
        previous_counts[-1],
        edits,
        changed,
        aligned,
        "ambiguous" if ambiguous else "unsupported" if reasons else "aligned",
        sorted(reasons),
        versions,
    )


def _positive_edit_spans(audit: SpanAudit, side: str) -> list[tuple[int, int]]:
    return [
        (edit[f"{side}_start"], edit[f"{side}_end"])
        for edit in audit.edit_spans
        if edit[f"{side}_start"] < edit[f"{side}_end"]
    ]


def _lexical_flags(clean: str, typo: str, audit: SpanAudit) -> dict[str, bool]:
    words: list[str] = []
    changed_text: list[str] = []
    numeric_span_hit = False
    for side, text in (("clean", clean), ("typo", typo)):
        edits = _positive_edit_spans(audit, side)
        changed_text.extend(text[start:end] for start, end in edits)
        words.extend(
            text[start:end].casefold().replace("’", "'") for start, end in audit.changed_words[side]
        )
        numeric_span_hit = numeric_span_hit or any(
            match.start() < right and left < match.end()
            for match in NUMBER_PATTERN.finditer(text)
            for left, right in edits
        )
    return {
        "number": numeric_span_hit
        or any(unicodedata.category(char).startswith("N") for word in words for char in word),
        "quantity": any(word in QUANTITY_WORDS for word in words),
        "unit": any(word in UNIT_WORDS for word in words)
        or any(char in UNIT_SYMBOLS for text in changed_text for char in text),
        "negation": any(word in NEGATION_WORDS for word in words),
    }


def _valid_span(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(part) is int for part in value)
        and 0 <= value[0] <= value[1] <= length
    )


def _regions_for_side(record: dict, side: str, text: str) -> tuple[list[dict] | None, int, str]:
    prompt = record.get(f"{side}_prompt")
    query_span = record.get(f"{side}_query_span")
    if not isinstance(prompt, str) or not _valid_span(query_span, len(prompt)):
        return None, 0, "exact_prompt_or_query_span_unavailable"
    if prompt[query_span[0] : query_span[1]] != text:
        return None, 0, "query_span_text_mismatch"
    layout = record.get("prompt_regions")
    regions = layout.get(side) if isinstance(layout, dict) else None
    if not isinstance(regions, list):
        return None, query_span[0], "prompt_regions_unavailable"
    for region in regions:
        if (
            not isinstance(region, dict)
            or not isinstance(region.get("kind"), str)
            or region.get("kind") not in REGION_KINDS
            or not _valid_span([region.get("start"), region.get("end")], len(prompt))
            or region["start"] == region["end"]
            or (
                region.get("option_label") is not None
                and not (
                    isinstance(region["option_label"], str)
                    and len(region["option_label"]) == 1
                    and "A" <= region["option_label"] <= "Z"
                )
            )
        ):
            return None, query_span[0], "prompt_regions_unsupported"
    return regions, query_span[0], ""


def _covered(start: int, end: int, regions: list[dict]) -> bool:
    cursor = start
    for region in sorted(regions, key=lambda item: (item["start"], item["end"])):
        if region["kind"] == "entity" or region["end"] <= cursor:
            continue
        if region["start"] > cursor:
            break
        cursor = max(cursor, region["end"])
        if cursor >= end:
            return True
    return False


def classify_edit_regions(record: dict, span_audit: SpanAudit | None = None) -> dict:
    """Return diagnostics, never a semantic label or outcome-based eligibility.

    False lexical flags mean no match in the frozen diagnostic lexicon, not proof
    that the concept is absent. Entity absence cannot be established from partial
    regions, so an entity flag is true for an explicit hit and otherwise unknown.
    """

    flags: dict[str, bool | None] = dict.fromkeys(RISK_NAMES)
    reasons: dict[str, list[str]] = {name: [] for name in RISK_NAMES}
    result = {"flags": flags, "reasons": reasons, "algorithm_versions": {"risk": RISK_VERSION}}
    clean, typo = record.get("clean_text"), record.get("typo_text")
    if not isinstance(clean, str) or not isinstance(typo, str):
        for values in reasons.values():
            values.append("exact_query_text_unavailable")
        return result
    audit = span_audit if span_audit is not None else diff_actual_spans(clean, typo)
    result["algorithm_versions"].update(audit.algorithm_versions)
    if audit.diff_status in {"ambiguous", "unavailable"}:
        for values in reasons.values():
            values.append(f"diff_{audit.diff_status}")
        return result
    if audit.diff_status == "no_change":
        for name in flags:
            flags[name] = False
            reasons[name].append("no_net_change")
        return result
    for name, value in _lexical_flags(clean, typo, audit).items():
        flags[name] = value
        reasons[name].append("diagnostic_lexicon_hit" if value else "no_diagnostic_lexicon_hit")
    hits: list[dict] = []
    complete = True
    layout_reasons: set[str] = set()
    for side, text in (("clean", clean), ("typo", typo)):
        edits = _positive_edit_spans(audit, side)
        if not edits:
            continue
        regions, offset, reason = _regions_for_side(record, side, text)
        if regions is None:
            complete = False
            layout_reasons.add(f"{side}_{reason}")
            continue
        for left, right in edits:
            start, end = offset + left, offset + right
            hits.extend(
                region for region in regions if region["start"] < end and start < region["end"]
            )
            if not _covered(start, end, regions):
                complete = False
                layout_reasons.add(f"{side}_edited_region_uncovered")
    for name, kind in (
        ("stem", "stem"),
        ("option_content", "option-content"),
        ("option_label", "option-label"),
    ):
        hit = any(region["kind"] == kind for region in hits)
        flags[name] = True if hit else False if complete else None
        reasons[name] = (
            ["explicit_region_hit"]
            if hit
            else ["covered_by_other_regions"]
            if complete
            else sorted(layout_reasons)
        )
    entity_hit = any(region["kind"] == "entity" for region in hits)
    flags["entity"] = True if entity_hit else None
    reasons["entity"] = [
        "explicit_entity_region_hit" if entity_hit else "entity_annotations_not_exhaustive"
    ]
    gold = record.get("canonical_gold")
    labels = record.get("valid_labels")
    if not isinstance(gold, str) or not isinstance(labels, list) or gold not in labels:
        reasons["gold_option"] = ["item_specific_gold_label_unavailable"]
    else:
        option_hits = [
            region for region in hits if region["kind"] in {"option-content", "option-label"}
        ]
        if any(region.get("option_label") == gold for region in option_hits):
            flags["gold_option"] = True
            reasons["gold_option"] = ["explicit_gold_option_region_hit"]
        elif complete and all(region.get("option_label") in labels for region in option_hits):
            flags["gold_option"] = False
            reasons["gold_option"] = ["covered_non_gold_regions"]
        else:
            reasons["gold_option"] = sorted(layout_reasons | {"option_membership_unresolved"})
    return result
