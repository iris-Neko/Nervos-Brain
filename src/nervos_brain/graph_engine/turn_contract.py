"""Normalization and deterministic derivation for TurnContract values."""

from __future__ import annotations

from typing import Any, Mapping

from nervos_brain.core_protocols.turn_protocols import TurnContract

from .product_policy import ProductPolicy, policy_from_state


_RELATIONS = {"new_task", "follow_up", "correction", "clarification_answer"}
_CONTEXT_REQUIREMENTS = {"none", "direct_reply", "recent_history"}
_ROUTES = {"direct", "retrieve", "clarify", "policy_response"}
_RETRIEVAL_POLICIES = {"none", "single", "deep"}
_POLICY_ACTIONS = {"allow", "constrain", "refuse"}
_CLARITIES = {"clear", "ambiguous"}
_LOCALE_SOURCES = {
    "explicit_request",
    "current_message",
    "confirmed_preference",
    "context_reference",
    "default_policy",
}
_AVAILABILITY = {"public", "user_owned"}
def _text(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _normalize_locale(value: Any, policy: ProductPolicy) -> str | None:
    candidate = _text(value).replace("_", "-")
    if not candidate:
        return None
    if candidate in policy.language.supported_locales:
        return candidate
    prefix = candidate.split("-", 1)[0]
    for supported in policy.language.supported_locales:
        if supported.split("-", 1)[0] == prefix:
            return supported
    return None


def resolve_response_locale(
    language: Mapping[str, Any],
    policy: ProductPolicy,
    *,
    confirmed_preference: str | None = None,
) -> tuple[str, str]:
    """Resolve locale by precedence without inspecting natural-language text."""
    requested = _normalize_locale(language.get("requested_output_locale"), policy)
    if requested:
        return requested, "explicit_request"

    communication = _normalize_locale(language.get("communication_locale"), policy)
    if communication and _text(language.get("clarity")) == "clear":
        return communication, "current_message"

    if policy.language.use_confirmed_user_preference:
        preference = _normalize_locale(confirmed_preference, policy)
        if preference:
            return preference, "confirmed_preference"
    return policy.language.ambiguous_locale, "default_policy"


def _normalize_info_needs(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        kind = _text(item.get("kind"))
        question = _text(item.get("question"))
        purpose = _text(item.get("purpose"))
        if not kind or not question:
            continue
        if not purpose:
            purpose = f"Support the current deliverable: {question}"
        availability = _text(item.get("availability"))
        if availability not in _AVAILABILITY:
            availability = "public"
        hints_raw = item.get("hints", {})
        hints = {
            _text(key): _text(value)
            for key, value in hints_raw.items()
            if _text(key) and _text(value)
        } if isinstance(hints_raw, Mapping) else {}
        normalized: dict[str, Any] = {
            "kind": kind,
            "question": question,
            "required": bool(item.get("required", False)),
            "availability": availability,
            "purpose": purpose,
        }
        if hints:
            normalized["hints"] = hints
        result.append(normalized)
    return result


def normalize_turn_contract(
    raw: Any,
    policy: ProductPolicy | None = None,
    *,
    current_message: str = "",
    confirmed_preference: str | None = None,
    available_context_ids: set[str] | None = None,
) -> TurnContract:
    """Validate model output and apply only protocol-level defaults."""
    active_policy = policy or ProductPolicy()
    source = raw if isinstance(raw, Mapping) else {}
    language_raw = source.get("language", {})
    language_raw = language_raw if isinstance(language_raw, Mapping) else {}
    requested = _normalize_locale(language_raw.get("requested_output_locale"), active_policy)
    communication = _normalize_locale(language_raw.get("communication_locale"), active_policy)
    clarity = _text(language_raw.get("clarity"))
    if clarity not in _CLARITIES:
        clarity = "ambiguous"
    language_source = _text(language_raw.get("locale_source"))
    if language_source not in _LOCALE_SOURCES:
        language_source = "default_policy"
    input_locales = _list_of_strings(language_raw.get("input_locales"))
    if communication and communication not in input_locales:
        input_locales.append(communication)
    response_locale, resolved_source = resolve_response_locale(
        {
            "requested_output_locale": requested,
            "communication_locale": communication,
            "clarity": clarity,
        },
        active_policy,
        confirmed_preference=confirmed_preference,
    )
    if language_source == "default_policy" or resolved_source == "default_policy":
        language_source = resolved_source

    context_raw = source.get("context", {})
    context_raw = context_raw if isinstance(context_raw, Mapping) else {}
    requirement = _text(context_raw.get("requirement"))
    if requirement not in _CONTEXT_REQUIREMENTS:
        requirement = "none"
    selected_ids = _list_of_strings(context_raw.get("selected_message_ids"))
    if available_context_ids is not None:
        selected_ids = [item for item in selected_ids if item in available_context_ids]
    if requirement == "none":
        selected_ids = []
    context = {
        "requirement": requirement,
        "purpose": _text(context_raw.get("purpose")),
        "selected_message_ids": selected_ids,
    }

    relation = _text(source.get("turn_relation"))
    if relation not in _RELATIONS:
        relation = "new_task"
    route = _text(source.get("route"))
    if not route:
        legacy_decision = _text(source.get("decision"))
        route = {
            "answer_direct": "direct",
            "direct": "direct",
            "has_needs": "retrieve",
            "retrieve": "retrieve",
            "ask_user": "clarify",
            "clarify": "clarify",
        }.get(legacy_decision, "")
    if route not in _ROUTES:
        route = "direct"
    retrieval_policy = _text(source.get("retrieval_policy"))
    if retrieval_policy not in _RETRIEVAL_POLICIES:
        retrieval_policy = "none" if route != "retrieve" else "single"
    if route in {"direct", "clarify", "policy_response"}:
        retrieval_policy = "none"

    info_needs = _normalize_info_needs(source.get("info_needs"))
    required_user_owned = any(
        bool(item.get("required")) and item.get("availability") == "user_owned"
        for item in info_needs
    )
    if route == "clarify" and not required_user_owned:
        route = "retrieve" if info_needs else "direct"
        retrieval_policy = "single" if route == "retrieve" else "none"
    if route == "retrieve" and not info_needs:
        info_needs = [{
            "kind": "concept_gap",
            "question": _text(source.get("resolved_request")) or _text(current_message),
            "required": False,
            "availability": "public",
            "purpose": "Provide evidence needed for the current deliverable.",
        }]
    if route != "retrieve":
        retrieval_policy = "none"

    policy_raw = source.get("policy", {})
    policy_raw = policy_raw if isinstance(policy_raw, Mapping) else {}
    policy_action = _text(policy_raw.get("action"))
    if policy_action not in _POLICY_ACTIONS:
        policy_action = "allow"
    turn_policy = {
        "action": policy_action,
        "categories": _list_of_strings(policy_raw.get("categories")),
        "reason": _text(policy_raw.get("reason")),
    }

    confidence_raw = source.get("confidence", {})
    confidence_raw = confidence_raw if isinstance(confidence_raw, Mapping) else {}
    constraints = _list_of_strings(source.get("constraints"))
    resolved_request = _text(source.get("resolved_request")) or _text(current_message)
    core_deliverable = _text(source.get("core_deliverable")) or resolved_request
    return {
        "schema_version": "turn_contract.v1",
        "core_deliverable": core_deliverable,
        "resolved_request": resolved_request,
        "turn_relation": relation,
        "language": {
            "input_locales": input_locales,
            "communication_locale": communication,
            "clarity": clarity,
            "requested_output_locale": requested,
            "locale_source": language_source,
        },
        "context": context,
        "route": route,
        "retrieval_policy": retrieval_policy,
        "info_needs": info_needs,
        "policy": turn_policy,
        "constraints": constraints,
        "confidence": {
            "intent": _float(confidence_raw.get("intent")),
            "language": _float(confidence_raw.get("language")),
            "context": _float(confidence_raw.get("context")),
        },
    }


def derive_legacy_state(
    contract: TurnContract,
    policy: ProductPolicy | None = None,
    *,
    confirmed_preference: str | None = None,
) -> dict[str, Any]:
    """Map the new contract to fields consumed by existing retrieval nodes."""
    active_policy = policy or ProductPolicy()
    route_map = {
        "direct": "answer_direct",
        "retrieve": "has_needs",
        "clarify": "ask_user",
        "policy_response": "answer_direct",
    }
    response_locale = resolve_response_locale(
        contract["language"],
        active_policy,
        confirmed_preference=confirmed_preference,
    )[0]
    return {
        "turn_contract": contract,
        "response_locale": response_locale,
        "locale": response_locale,
        "resolved_question": contract["resolved_request"],
        "retrieval_policy": contract["retrieval_policy"],
        "info_needs": list(contract["info_needs"]),
        "_route_decision": route_map[contract["route"]],
        "_financial_guidance_refusal": contract["policy"]["action"] == "refuse",
    }


def merge_contextual_contract(
    first_pass: TurnContract,
    second_pass: Any,
    policy: ProductPolicy | None = None,
    *,
    current_message: str = "",
    confirmed_preference: str | None = None,
    available_context_ids: set[str] | None = None,
) -> TurnContract:
    """Merge a context-aware pass while preserving current-turn constraints."""
    active_policy = policy or ProductPolicy()
    normalized = normalize_turn_contract(
        second_pass,
        active_policy,
        current_message=current_message,
        confirmed_preference=confirmed_preference,
        available_context_ids=available_context_ids,
    )
    first_language = first_pass["language"]
    explicit = _normalize_locale(first_language.get("requested_output_locale"), active_policy)
    merged = dict(first_pass)
    merged["language"] = dict(first_language)
    if explicit:
        merged["language"]["requested_output_locale"] = explicit
        merged["language"]["locale_source"] = "explicit_request"

    # The second pass may refine the request after seeing selected history. It
    # may change route and retrieval breadth, but the first pass remains the
    # authority for explicit output language and the current deliverable.
    merged_context = dict(normalized["context"])
    if first_pass["context"]["requirement"] != "none":
        merged_context["requirement"] = first_pass["context"]["requirement"]
    if not merged_context.get("selected_message_ids"):
        merged_context["selected_message_ids"] = list(
            first_pass["context"].get("selected_message_ids", [])
        )
    merged["context"] = merged_context
    if normalized["resolved_request"]:
        merged["resolved_request"] = normalized["resolved_request"]
    merged["turn_relation"] = normalized["turn_relation"]
    merged["route"] = normalized["route"]
    merged["retrieval_policy"] = normalized["retrieval_policy"]
    merged["info_needs"] = normalized["info_needs"]
    merged["constraints"] = list(dict.fromkeys(first_pass["constraints"] + normalized["constraints"]))
    first_action = first_pass["policy"]["action"]
    second_action = normalized["policy"]["action"]
    action_rank = {"allow": 0, "constrain": 1, "refuse": 2}
    if action_rank.get(second_action, 0) >= action_rank.get(first_action, 0):
        merged["policy"] = normalized["policy"]
    else:
        merged["policy"] = first_pass["policy"]
    merged["confidence"] = normalized["confidence"]
    merged["schema_version"] = "turn_contract.v1"
    return merged


def product_policy_for_state(state: Mapping[str, Any] | None) -> ProductPolicy:
    return policy_from_state(state)
