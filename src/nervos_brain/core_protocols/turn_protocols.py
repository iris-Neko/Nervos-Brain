"""Typed protocol for one semantically interpreted user turn.

The protocol deliberately keeps natural-language decisions in data returned by
the Turn Interpreter.  Downstream nodes consume this contract instead of
re-running platform-specific keyword heuristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypedDict


TurnRelation = Literal[
    "new_task",
    "follow_up",
    "correction",
    "clarification_answer",
]
ContextRequirement = Literal["none", "direct_reply", "recent_history"]
TurnRoute = Literal["direct", "retrieve", "clarify", "policy_response"]
RetrievalPolicy = Literal["none", "single", "deep"]
PolicyAction = Literal["allow", "constrain", "refuse"]
LanguageClarity = Literal["clear", "ambiguous"]
LocaleSource = Literal[
    "explicit_request",
    "current_message",
    "confirmed_preference",
    "context_reference",
    "default_policy",
]
InfoAvailability = Literal["public", "user_owned"]


class TurnLanguage(TypedDict):
    input_locales: List[str]
    communication_locale: str | None
    clarity: LanguageClarity
    requested_output_locale: str | None
    locale_source: LocaleSource


class TurnContext(TypedDict):
    requirement: ContextRequirement
    purpose: str
    selected_message_ids: List[str]


class TurnPolicy(TypedDict):
    action: PolicyAction
    categories: List[str]
    reason: str


class TurnConfidence(TypedDict):
    intent: float
    language: float
    context: float


class TurnInfoNeed(TypedDict):
    kind: str
    question: str
    required: bool
    availability: InfoAvailability
    purpose: str
    hints: NotRequired[Dict[str, str]]


class TurnContract(TypedDict):
    schema_version: str
    core_deliverable: str
    resolved_request: str
    turn_relation: TurnRelation
    language: TurnLanguage
    context: TurnContext
    route: TurnRoute
    retrieval_policy: RetrievalPolicy
    info_needs: List[TurnInfoNeed]
    policy: TurnPolicy
    constraints: List[str]
    confidence: TurnConfidence


def is_turn_contract(value: Any) -> bool:
    """Return whether ``value`` has the minimum contract shape.

    Detailed normalization belongs to ``graph_engine.turn_contract``.  This
    small predicate is useful at protocol boundaries without importing policy
    or LLM code.
    """
    if not isinstance(value, dict):
        return False
    required = (
        "schema_version",
        "core_deliverable",
        "resolved_request",
        "turn_relation",
        "language",
        "context",
        "route",
        "retrieval_policy",
        "info_needs",
        "policy",
        "constraints",
        "confidence",
    )
    return all(key in value for key in required)
