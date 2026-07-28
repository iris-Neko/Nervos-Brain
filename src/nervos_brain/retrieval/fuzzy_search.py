"""Fuzzy-match layer for the multi-path retrieval pipeline.

Purpose: catch naming variants, abbreviations, and minor spelling differences
that neither semantic vectors nor exact BM25 terms cover.

Examples:
  query "open_channel"  →  hits record titled "OpenChannel"
  query "fibre"         →  hits record titled "Fiber"
  query "htlc timeout"  →  hits record with keywords "HTLC, on-chain timeout"

Implementation uses stdlib ``difflib.SequenceMatcher`` (no extra deps).
For large corpora (>50 k candidates) ``rapidfuzz`` would be faster, but
difflib is sufficient here and avoids an additional dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List


@dataclass
class FuzzyResult:
    anchor: str
    title: str
    source: str
    score: float          # SequenceMatcher ratio, in [0, 1]
    matched_field: str    # which field produced the best match


def _ratio(a: str, b: str, *, minimum: float = 0.0) -> float:
    """Return the similarity ratio, skipping pairs that cannot reach minimum."""
    left = a.lower()
    right = b.lower()
    total_length = len(left) + len(right)
    if total_length == 0:
        return 0.0

    # Even a perfect alignment cannot match more than twice the shorter
    # string's length. This cheap bound avoids expensive edit comparisons
    # between a full natural-language question and short keyword tokens.
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
    """Score each candidate against *query* and return the best matches.

    Args:
        query:      The user's natural-language query or a specific term.
        candidates: List of dicts with at least ``anchor``, ``title``,
                    ``source``, and optionally ``keywords``.
        threshold:  Minimum ``SequenceMatcher.ratio()`` to include a hit.
        top_k:      Maximum number of results returned.

    Scoring strategy
    ----------------
    We check three fields (``title``, ``keywords`` token-by-token, ``anchor``)
    and keep the highest ratio as the record's score.  This way a record
    is surfaced even when only one of its identifiers matches the query.
    """
    results: List[FuzzyResult] = []

    for c in candidates:
        anchor: str = c.get("anchor", "")
        title: str = c.get("title", "")
        source: str = c.get("source", "")
        keywords: str = c.get("keywords", "")

        best_score = 0.0
        best_field = ""

        # title
        s = _ratio(query, title, minimum=max(threshold, best_score))
        if s > best_score:
            best_score, best_field = s, "title"

        # keyword tokens (individual terms are often more informative than
        # the full keyword string)
        for kw in _split_keywords(keywords):
            s = _ratio(query, kw, minimum=max(threshold, best_score))
            if s > best_score:
                best_score, best_field = s, f"keyword:{kw}"

        # anchor (e.g. "doc:open-channel-rfc#chunk:0")
        # strip the structural prefix before comparing
        anchor_stem = anchor.split("#")[0].replace("doc:", "").replace("-", " ")
        s = _ratio(query, anchor_stem, minimum=max(threshold, best_score))
        if s > best_score:
            best_score, best_field = s, "anchor"

        if best_score >= threshold:
            results.append(
                FuzzyResult(
                    anchor=anchor,
                    title=title,
                    source=source,
                    score=best_score,
                    matched_field=best_field,
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def _split_keywords(keywords: str) -> List[str]:
    """Split comma / semicolon / space separated keyword string into tokens."""
    import re
    tokens = re.split(r"[,;\s]+", keywords.strip())
    return [t.strip() for t in tokens if t.strip()]
