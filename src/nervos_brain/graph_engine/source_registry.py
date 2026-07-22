"""Retrieval source registry shared by planning and runtime validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalSource:
    id: str
    label: str
    tool_hint: str
    description: str


SOURCES: dict[str, RetrievalSource] = {
    "github_docs": RetrievalSource(
        id="github_docs",
        label="repository documentation and specifications",
        tool_hint="qdrant_search",
        description=(
            "Repository documentation, tutorials, protocol specifications, READMEs, standards, "
            "and conceptual material. Prefer this source for authoritative onboarding paths, "
            "concept explanations, specifications, and documentation links."
        ),
    ),
    "github_code": RetrievalSource(
        id="github_code",
        label="repository source code, configuration, and examples",
        tool_hint="github_search",
        description=(
            "Repository source code, configuration, scripts, functions, types, modules, tests, "
            "SDK examples, and executable call patterns. Prefer this source for source locations, "
            "function or class implementations, configuration options, call chains, code related "
            "to an error, and concrete code snippets."
        ),
    ),
    "nervos_talk": RetrievalSource(
        id="nervos_talk",
        label="forum posts and replies",
        tool_hint="discourse_query",
        description=(
            "Forum posts, replies, community discussions, proposals, project introductions, real "
            "cases, community opinions, and roadmap discussions. Prefer this source for community "
            "availability, project cases, discussion links, and related examples."
        ),
    ),
}

SOURCE_ALIASES: dict[str, str] = {
    "official": "github_docs",
    "official_docs": "github_docs",
    "official-docs": "github_docs",
    "docs": "github_docs",
    "documentation": "github_docs",
    "github": "github_docs",
    "github_doc": "github_docs",
    "github_docs": "github_docs",
    "code": "github_code",
    "source_code": "github_code",
    "github_code": "github_code",
    "github-source": "github_code",
    "github_source": "github_code",
    "repo_code": "github_code",
    "rfcs": "github_docs",
    "rfc": "github_docs",
    "talk": "nervos_talk",
    "forum": "nervos_talk",
    "community": "nervos_talk",
    "discourse": "nervos_talk",
    "nervos_talk": "nervos_talk",
}

QDRANT_FILTER_KEYS = {
    "source",
    "type",
    "doc_type",
    "version",
    "lang",
    "url",
    "anchor",
    "topic",
    "title",
    "keywords",
}


def format_source_registry_for_prompt() -> str:
    lines = [
        "The source field may use only these exact values. Do not invent aliases such as "
        "official_docs, docs, or documentation:"
    ]
    for source in SOURCES.values():
        lines.append(
            f"- source={source.id}: {source.description} "
            f"Recommended tool: {source.tool_hint}."
        )
    lines.append(
        "qdrant_search.filters supports: source, topic, type/doc_type, version, lang, url, "
        "anchor, title, and keywords."
    )
    lines.append(
        "Use source=github_docs for official tutorials, official documentation, and onboarding "
        "material; source=github_code or github_search for source code, functions, configuration, "
        "and code snippets; and discourse_query for community discussions and project cases."
    )
    return "\n".join(lines)


def normalize_tool_filters(tool: str, filters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize LLM-produced filters before tool execution."""
    if not isinstance(filters, dict):
        return {}, ["filters_not_dict"]

    normalized: dict[str, Any] = {}
    notes: list[str] = []
    for raw_key, raw_value in filters.items():
        key = str(raw_key).strip()
        if not key or raw_value in (None, ""):
            continue
        if isinstance(raw_value, (list, dict, tuple, set)):
            notes.append(f"dropped_complex_filter:{key}")
            continue

        value = str(raw_value).strip()
        if not value:
            continue

        if tool == "qdrant_search" and key not in QDRANT_FILTER_KEYS:
            notes.append(f"dropped_unknown_filter:{key}")
            continue

        if key == "source":
            canonical = SOURCE_ALIASES.get(value.lower())
            if canonical:
                if canonical != value:
                    notes.append(f"mapped_source:{value}->{canonical}")
                normalized[key] = canonical
            else:
                notes.append(f"dropped_unknown_source:{value}")
            continue

        if key == "type":
            normalized[key] = "github_doc" if value == "official_docs" else value
            continue

        normalized[key] = value

    return normalized, notes


def should_retry_qdrant_without_filters(filters: dict[str, Any], evidence_count: int) -> bool:
    return evidence_count == 0 and bool(filters)
