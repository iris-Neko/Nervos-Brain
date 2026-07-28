"""Fuzzy matching for names, abbreviations, and minor spelling errors.

RapidFuzz performs a compiled full-corpus prefilter. The shortlisted records
are then reranked with stdlib ``difflib.SequenceMatcher`` so existing score
semantics are retained while most Python comparisons are avoided.
"""
from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, List

from rapidfuzz import fuzz, process


# Indel/LCS similarity upper-bounds SequenceMatcher's matching-block ratio.
# This epsilon only protects threshold comparisons from float rounding.
_PREFILTER_EPSILON = 1e-6
_FIELD_TITLE = 0
_FIELD_KEYWORD = 1
_FIELD_ANCHOR = 2


@dataclass
class FuzzyResult:
    anchor: str
    title: str
    source: str
    score: float
    matched_field: str


@dataclass(frozen=True)
class _IndexedCandidate:
    anchor: str
    title: str
    source: str
    choice_start: int
    choice_end: int


class FuzzyIndex:
    """Immutable-at-query-time fuzzy index reusable across searches."""

    def __init__(self, candidates: Iterable[Any] = ()) -> None:
        self._candidates: list[_IndexedCandidate] = []
        self._choices: list[str] = []
        self._owners: array[int] = array("I")
        self._field_codes = bytearray()
        self.build(candidates)

    def build(self, candidates: Iterable[Any]) -> int:
        self._candidates = []
        self._choices = []
        self._owners = array("I")
        self._field_codes = bytearray()

        for owner, candidate in enumerate(candidates):
            anchor = str(_candidate_value(candidate, "anchor") or "")
            title = str(_candidate_value(candidate, "title") or "")
            source = str(_candidate_value(candidate, "source") or "")
            keywords = str(_candidate_value(candidate, "keywords") or "")
            choice_start = len(self._choices)

            self._add_choice(owner, title, _FIELD_TITLE)
            for keyword in _split_keywords(keywords):
                self._add_choice(owner, keyword, _FIELD_KEYWORD)
            anchor_stem = anchor.split("#")[0].replace("doc:", "").replace("-", " ")
            self._add_choice(owner, anchor_stem, _FIELD_ANCHOR)

            self._candidates.append(
                _IndexedCandidate(
                    anchor=anchor,
                    title=title,
                    source=source,
                    choice_start=choice_start,
                    choice_end=len(self._choices),
                )
            )
        return len(self._candidates)

    def search(
        self,
        query: str,
        *,
        threshold: float = 0.55,
        top_k: int = 10,
        allowed_owners: Set[int] | None = None,
    ) -> List[FuzzyResult]:
        normalized_query = str(query or "").strip().lower()
        if (
            not normalized_query
            or not self._choices
            or top_k <= 0
            or (allowed_owners is not None and not allowed_owners)
        ):
            return []

        normalized_threshold = max(0.0, min(float(threshold), 1.0))
        prefilter_cutoff = max(0.0, normalized_threshold - _PREFILTER_EPSILON) * 100.0
        owner_ids: set[int] = set()
        for _choice, _score, choice_index in process.extract_iter(
            normalized_query,
            self._choices,
            scorer=fuzz.ratio,
            processor=None,
            score_cutoff=prefilter_cutoff,
            score_hint=normalized_threshold * 100.0,
        ):
            owner = int(self._owners[choice_index])
            if allowed_owners is None or owner in allowed_owners:
                owner_ids.add(owner)

        results: list[FuzzyResult] = []
        for owner in sorted(owner_ids):
            candidate = self._candidates[owner]
            best_score = 0.0
            best_choice_index = -1
            for choice_index in range(candidate.choice_start, candidate.choice_end):
                score = _ratio(
                    normalized_query,
                    self._choices[choice_index],
                    minimum=max(normalized_threshold, best_score),
                )
                if score > best_score:
                    best_score = score
                    best_choice_index = choice_index
            if best_score < normalized_threshold or best_choice_index < 0:
                continue
            results.append(
                FuzzyResult(
                    anchor=candidate.anchor,
                    title=candidate.title,
                    source=candidate.source,
                    score=best_score,
                    matched_field=self._matched_field(best_choice_index),
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _add_choice(self, owner: int, value: str, field_code: int) -> None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return
        self._choices.append(normalized)
        self._owners.append(owner)
        self._field_codes.append(field_code)

    def _matched_field(self, choice_index: int) -> str:
        field_code = self._field_codes[choice_index]
        if field_code == _FIELD_TITLE:
            return "title"
        if field_code == _FIELD_KEYWORD:
            return f"keyword:{self._choices[choice_index]}"
        return "anchor"

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def choice_count(self) -> int:
        return len(self._choices)


def _ratio(a: str, b: str, *, minimum: float = 0.0) -> float:
    """Return the legacy similarity ratio, skipping impossible pairs."""
    left = a.lower()
    right = b.lower()
    total_length = len(left) + len(right)
    if total_length == 0:
        return 0.0

    upper_bound = (2.0 * min(len(left), len(right))) / total_length
    if upper_bound < minimum:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def fuzzy_search(
    query: str,
    candidates: List[dict],
    threshold: float = 0.55,
    top_k: int = 10,
) -> List[FuzzyResult]:
    """Build a temporary index and return the best fuzzy matches.

    Long-lived retrievers should build one ``FuzzyIndex`` and reuse it.
    """
    return FuzzyIndex(candidates).search(
        query,
        threshold=threshold,
        top_k=top_k,
    )


def _split_keywords(keywords: str) -> List[str]:
    """Split comma-, semicolon-, or whitespace-separated keywords."""
    tokens = re.split(r"[,;\s]+", keywords.strip())
    return [token.strip() for token in tokens if token.strip()]


def _candidate_value(candidate: Any, key: str) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, "")
    return getattr(candidate, key, "")
