"""M7-T1~T8: 8 个真实 GraphEngine 节点函数。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from nervos_brain.response_normalizer.normalizer import (
    chunk_for_platform,
    normalize_citations,
    sanitize_markdown,
    validate_response_shape,
)
from nervos_brain.response_normalizer.platform_formatter import format_response_to_outbound

from . import prompts
from .llm import call_llm, call_llm_json, get_last_call_meta
from .product_policy import policy_from_state
from .provider_registry import ProviderCapabilityRegistry
from .source_registry import (
    format_source_registry_for_prompt,
    normalize_tool_filters,
    should_retry_qdrant_without_filters,
)
def _normalize_info_needs_schema(info_needs: Any) -> list[dict]:
    """Normalize an already interpreted info-need list without inference."""
    if not isinstance(info_needs, list):
        return []

    sanitized: list[dict] = []
    for raw_need in info_needs:
        if not isinstance(raw_need, dict):
            continue
        need = dict(raw_need)
        need.setdefault("kind", "concept_gap")
        need.setdefault("question", "")
        need["required"] = bool(need.get("required", False))
        need.setdefault("availability", "public")
        need.setdefault("purpose", str(need.get("question", "") or "Support the current deliverable."))
        sanitized.append(need)
    return sanitized


def _force_policy_response(state: dict, *, question: str | None = None) -> dict[str, Any]:
    policy = policy_from_state(state)
    request_id = str(state.get("request_id", "unknown"))
    locale = str(state.get("response_locale") or state.get("locale") or policy.language.default_locale)
    response = {
        "request_id": request_id,
        "text": policy.message("policy_refusal", locale),
        "citations": [],
        "answer_mode": "policy_refusal",
    }
    return {
        "_route_decision": "answer_direct",
        "retrieval_policy": "none",
        "info_needs": [],
        "resolved_question": str(question if question is not None else _effective_question(state)),
        "budget": _merge_policy_budget(state, "none"),
        "_final_response": response,
        "_direct_answer": True,
        "_financial_guidance_refusal": True,
    }


def _enforce_policy_response(state: dict, response: dict[str, Any]) -> bool:
    """Apply an already classified policy action without text classification."""
    contract = state.get("turn_contract", {})
    policy_action = contract.get("policy", {}).get("action") if isinstance(contract, dict) else None
    if policy_action != "refuse":
        return False
    product_policy = policy_from_state(state)
    locale = str(state.get("response_locale") or state.get("locale") or product_policy.language.default_locale)
    response["text"] = product_policy.message("policy_refusal", locale)
    response["citations"] = []
    response["answer_mode"] = "policy_refusal"
    return True


logger = logging.getLogger(__name__)

_MODEL_TIERS = {"low", "medium", "high"}
_NODE_FALLBACK_TIERS = {
    "turn_interpreter": "low",
    "retriever_planner": "low",
    "reflection_pre": "low",
    "reflection_post": "medium",
    "direct_answer": "low",
    "answer_composer": "medium",
    "response_compliance": "medium",
}
_LLM_ROUTER_SYSTEM = """Prompt ID: model_router
You are the model-tier router.
Your only task is to select one of low, medium, or high for the current graph
node. Judge only task complexity, risk, and the node goal. Do not change graph
routing, retrieval policy, or answer content.

Tier meanings:
- low: low-risk, local, short-path work such as a short direct answer, format
  conversion, structured classification, or a light judgment with sufficient
  evidence.
- medium: the default general tier for ordinary technical answers, final answer
  composition, and stable integration of a small amount of evidence.
- high: deep reasoning for complex implementation, cross-source conflict,
  high-risk multi-evidence synthesis, severe off-topic review, complex
  debugging, or safety-sensitive decisions.

Selection constraints:
- Do not underspend or overspend model capacity. Judge only what this node
  must decide; do not upgrade based on a domain name, entity name, or request
  for detail.
- turn_interpreter and retriever_planner default to low. Upgrade only when
  the task structure has conflict, complex dependencies, or high risk.
- reflection_pre defaults to low. Do not use a higher tier for more background
  when existing evidence already completes the core deliverable.
- reflection_post is usually medium. Use high only for severe citation mismatch,
  incorrect evidence generalization, unsupported claims, or obvious scope drift.
- answer_composer defaults to medium. Use high only for complex implementation,
  multi-source conflict, or safety-sensitive plans. Many evidence items do not
  require a higher tier when the core answer is simple.
- direct_answer defaults to low. Upgrade only when the request itself needs
  complex reasoning or has a high-risk boundary.
- Near or beyond the time target, choose a faster tier and complete the core
  deliverable instead of using deep reasoning for marginal completeness.
- high is exceptional; do not use it for format repair, short follow-ups, lists
  of resources, or simple synthesis with sufficient evidence.

Return JSON only:
{"tier":"low|medium|high","reasoning":"brief reason","confidence":0.0}
"""


def _budget_int(state: dict, key: str, default: int) -> int:
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        return default
    value = budget.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _merge_policy_budget(state: dict, retrieval_policy: str) -> dict[str, Any]:
    raw_budget = state.get("budget", {})
    budget = dict(raw_budget) if isinstance(raw_budget, dict) else {}

    def current_int(key: str, default: int) -> int:
        try:
            return int(budget.get(key, default))
        except (TypeError, ValueError):
            return default

    if retrieval_policy == "none":
        budget["max_hops"] = 0
        budget["max_tool_calls"] = 0
        budget["max_reflection_rounds_pre"] = 0
        budget["max_reflection_rounds_post"] = min(current_int("max_reflection_rounds_post", 1), 1)
    elif retrieval_policy == "single":
        budget["max_hops"] = max(1, min(current_int("max_hops", 1), 1))
        budget["max_tool_calls"] = max(1, min(current_int("max_tool_calls", 2), 2))
        budget["max_reflection_rounds_pre"] = max(1, min(current_int("max_reflection_rounds_pre", 1), 1))
        budget["max_reflection_rounds_post"] = max(1, min(current_int("max_reflection_rounds_post", 1), 1))
    else:
        budget.setdefault("max_hops", 3)
        budget.setdefault("max_tool_calls", 3)
        budget.setdefault("max_reflection_rounds_pre", 2)
        budget.setdefault("max_reflection_rounds_post", 2)

    return budget


def _int_like(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_message_context(state: dict) -> dict[str, str]:
    user_msg = state.get("user_message", {})
    if not isinstance(user_msg, dict):
        return {}
    context = user_msg.get("context", {})
    if not isinstance(context, dict):
        return {}
    return {k: str(v) for k, v in context.items() if v is not None}


def _image_paths_from_state(state: dict) -> list[str]:
    user_msg = state.get("user_message", {})
    if not isinstance(user_msg, dict):
        return []
    attachments = user_msg.get("attachments", [])
    if not isinstance(attachments, list):
        return []
    paths: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        if str(attachment.get("kind", "") or "") != "image":
            continue
        local_path = str(attachment.get("local_path", "") or "").strip()
        if local_path:
            paths.append(local_path)
    return paths[:4]


def _merge_trace_summary(existing: str, extra: str) -> str:
    left = existing.strip()
    right = extra.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} | {right}"


def _time_budget_snapshot(state: dict) -> dict[str, Any]:
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        budget = {}
    started_ms = _int_like(state.get("_request_started_ts_ms", 0))
    now_ms = int(time.time() * 1000)
    elapsed_ms = max(0, now_ms - started_ms) if started_ms > 0 else 0
    target_ms = _int_like(budget.get("target_elapsed_ms", 0))
    max_ms = _int_like(budget.get("max_elapsed_ms", 0))
    llm_trace = state.get("_llm_trace", [])
    node_timings = state.get("_node_timings", [])
    return {
        "elapsed_ms": elapsed_ms,
        "target_elapsed_ms": target_ms,
        "max_elapsed_ms": max_ms,
        "remaining_target_ms": max(0, target_ms - elapsed_ms) if target_ms > 0 else 0,
        "remaining_max_ms": max(0, max_ms - elapsed_ms) if max_ms > 0 else 0,
        "node_timings": list(node_timings)[-8:] if isinstance(node_timings, list) else [],
        "llm_calls": len(llm_trace) if isinstance(llm_trace, list) else 0,
    }


def _time_budget_prompt(state: dict) -> str:
    snap = _time_budget_snapshot(state)
    target = snap["target_elapsed_ms"]
    max_ms = snap["max_elapsed_ms"]
    if target <= 0 and max_ms <= 0:
        return "No hard time budget is set; still avoid unnecessary retrieval, reflection, and rewriting."
    return (
        f"Elapsed {snap['elapsed_ms']}ms; "
        f"target {target or 'unset'}ms, target remaining {snap['remaining_target_ms']}ms; "
        f"maximum {max_ms or 'unset'}ms, maximum remaining {snap['remaining_max_ms']}ms. "
        "When little time remains, prioritize an accurate concise answer with its evidence boundary "
        "and avoid additional retrieval, reflection, or rewriting."
    )


def _merge_llm_trace(state: dict, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = state.get("_llm_trace", [])
    merged = list(existing) if isinstance(existing, list) else []
    merged.extend(rows)
    return merged[-80:]


def _summarize_llm_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "calls": len(rows),
        "elapsed_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "by_node": {},
    }
    by_node: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        elapsed = _int_like(row.get("elapsed_ms", 0))
        usage = row.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = _int_like(usage.get("input_tokens", 0))
        output_tokens = _int_like(usage.get("output_tokens", 0))
        total_tokens = _int_like(usage.get("total_tokens", 0))
        summary["elapsed_ms"] += elapsed
        summary["input_tokens"] += input_tokens
        summary["output_tokens"] += output_tokens
        summary["total_tokens"] += total_tokens
        node = str(row.get("node", "unknown") or "unknown")
        bucket = by_node.setdefault(
            node,
            {"calls": 0, "elapsed_ms": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        bucket["elapsed_ms"] += elapsed
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += total_tokens
    summary["by_node"] = by_node
    return summary


def _llm_meta_row(
    state: dict,
    *,
    node_name: str,
    call_kind: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = get_last_call_meta() or {}
    usage = meta.get("usage", {})
    return {
        "node": node_name,
        "kind": call_kind,
        "model": str(meta.get("model") or (profile or {}).get("model") or ""),
        "tier": str((profile or {}).get("tier", "")),
        "reasoning_effort": str(meta.get("reasoning_effort") or (profile or {}).get("reasoning_effort") or ""),
        "service_tier": str(meta.get("service_tier") or state.get("_llm_service_tier") or ""),
        "json_mode": bool(meta.get("json_mode", call_kind.endswith("_json"))),
        "max_tokens": _int_like(meta.get("max_tokens", (profile or {}).get("max_tokens", 0))),
        "elapsed_ms": _int_like(meta.get("elapsed_ms", 0)),
        "usage": usage if isinstance(usage, dict) else {},
        "time_budget": _time_budget_snapshot(state),
    }


def _llm_update(state: dict, row: dict[str, Any]) -> dict[str, Any]:
    rows = _merge_llm_trace(state, [row])
    return {
        "_llm_trace": rows,
        "_llm_usage_summary": _summarize_llm_usage(rows),
    }


def _llm_update_for_call(
    state: dict,
    *,
    node_name: str,
    call_kind: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    router_row = profile.get("_router_llm_row")
    if isinstance(router_row, dict):
        rows.append(router_row)
    rows.append(_llm_meta_row(state, node_name=node_name, call_kind=call_kind, profile=profile))
    merged = _merge_llm_trace(state, rows)
    return {
        "_llm_trace": merged,
        "_llm_usage_summary": _summarize_llm_usage(merged),
    }


def _run_async_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(awaitable)).result()


def _archive_store_from_state(state: dict) -> Any | None:
    archive = state.get("_archive_store")
    if archive is not None:
        return archive
    retriever = state.get("_multi_retriever")
    if retriever is None:
        return None
    return getattr(retriever, "_archive", None)


def _tool_args_for_step(
    *,
    state: dict,
    step: dict[str, Any],
    tool: str,
    query: str,
    filters: dict[str, Any],
    top_k: int,
    context: dict[str, str],
) -> dict[str, Any] | None:
    retriever = state.get("_multi_retriever")
    memory_service = state.get("_memory_service")
    archive_store = _archive_store_from_state(state)
    transport = state.get("_tool_transport") or state.get("_mcp_transport")

    if tool == "qdrant_search":
        args: dict[str, Any] = {"query": query, "filters": filters, "top_k": top_k}
        regex_queries = step.get("regex_queries")
        if isinstance(regex_queries, list):
            args["regex_queries"] = regex_queries
        if retriever is not None:
            args["_multi_retriever"] = retriever
        if state.get("_qdrant_store") is not None:
            args["_store"] = state["_qdrant_store"]
        return args

    if tool == "discourse_query":
        args = {"query": query, "top_k": top_k}
        category = str(filters.get("category") or filters.get("topic") or "").strip()
        time_range = str(step.get("time_range", filters.get("time_range", "")) or "").strip()
        if category:
            args["category"] = category
        if time_range:
            args["time_range"] = time_range
        if archive_store is not None:
            args["_archive_store"] = archive_store
        if transport is not None:
            args["_transport"] = transport
        return args

    if tool == "github_search":
        args = {"query": query, "top_k": top_k}
        repo = str(filters.get("repo") or filters.get("topic") or "").strip()
        path = str(filters.get("path") or "").strip()
        if repo:
            args["repo"] = repo
        if path:
            args["path"] = path
        if archive_store is not None:
            args["_archive_store"] = archive_store
        if transport is not None:
            args["_transport"] = transport
        return args

    if tool == "memory_fetch":
        platform = context.get("platform", "")
        if platform and context.get("guild_id") and context.get("channel_id"):
            args = {
                "namespace": "channel",
                "platform": platform,
                "guild_id": context["guild_id"],
                "channel_id": context["channel_id"],
            }
        elif platform and context.get("user_id"):
            args = {
                "namespace": "user",
                "platform": platform,
                "user_id": context["user_id"],
            }
        else:
            return None
        if memory_service is not None:
            args["_memory_service"] = memory_service
        return args

    return None


def _execute_retrieval_tool_call(
    *,
    request_id: str,
    step_id: str,
    tool: str,
    args: dict[str, Any],
    build_tool_call_request: Any,
    check_idempotency: Any,
    execute_tool: Any,
    handlers: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    try:
        req = build_tool_call_request(
            request_id=request_id,
            step_id=step_id,
            tool=tool,
            args=args,
        )
    except ValueError as exc:
        return None, {
            "step_id": step_id,
            "tool": str(tool),
            "status": "error",
            "error_code": "ERR_TOOL_SCHEMA_INVALID",
            "error_message": str(exc)[:200],
            "evidence_count": 0,
        }, False

    if check_idempotency(req["idempotency_key"]):
        return None, {
            "step_id": step_id,
            "tool": str(tool),
            "status": "duplicate",
            "evidence_count": 0,
        }, False

    handler = handlers.get(tool)
    if handler is None:
        return None, {
            "step_id": step_id,
            "tool": str(tool),
            "status": "error",
            "error_code": "ERR_MCP_TRANSPORT_UNAVAILABLE",
            "evidence_count": 0,
        }, False

    try:
        result = _run_async_sync(execute_tool(req, handler))
    except Exception as exc:
        return None, {
            "step_id": step_id,
            "tool": str(tool),
            "status": "error",
            "error_code": "ERR_TOOL_EXECUTION_FAILED",
            "error_message": str(exc)[:200],
            "evidence_count": 0,
        }, False

    evidence_items = result.get("evidence", []) if isinstance(result, dict) else []
    evidence_count = len(evidence_items) if isinstance(evidence_items, list) else 0
    trace_row = {
        "step_id": step_id,
        "tool": str(tool),
        "status": "empty" if result.get("ok") and evidence_count == 0 else str(result.get("status", "error")),
        "evidence_count": evidence_count,
        "latency_ms": max(
            0,
            _int_like(result.get("finished_ts_ms", 0)) - _int_like(result.get("started_ts_ms", 0)),
        ),
    }
    if isinstance(result.get("error"), dict):
        trace_row["error_code"] = str(result["error"].get("code", ""))
        trace_row["error_message"] = str(result["error"].get("message", ""))
    data = result.get("data", {}) if isinstance(result, dict) else {}
    if isinstance(data, dict):
        for key in (
            "regex_queries_count",
            "regex_valid_count",
            "regex_dropped_count",
            "regex_dropped_reasons",
        ):
            if key in data:
                trace_row[key] = data[key]
    return result, trace_row, True


def _tool_trace_summary(traces: list[dict[str, Any]]) -> str:
    if not traces:
        return ""
    parts: list[str] = []
    for row in traces[:6]:
        step_id = str(row.get("step_id", "?"))
        tool = str(row.get("tool", "?"))
        status = str(row.get("status", "unknown"))
        code = str(row.get("error_code", "")).strip()
        count = _int_like(row.get("evidence_count", 0))
        if code:
            status = f"{status}/{code}"
        parts.append(f"{step_id}:{tool}={status}:{count}")
    return "tools=" + ",".join(parts)


def _select_model(
    state: dict,
    task_type: str,
    *,
    require_json: bool,
) -> str | None:
    registry = state.get("_provider_registry")
    if registry is None:
        registry = ProviderCapabilityRegistry()
    if not hasattr(registry, "get_model_for"):
        return None

    try:
        max_cost = str(state.get("_provider_max_cost", "high"))
        return registry.get_model_for(  # type: ignore[attr-defined]
            task_type, require_json=require_json, max_cost=max_cost
        )
    except Exception:
        return None


def _coerce_profile(raw: Any, *, fallback_tier: str = "low") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    tier = str(raw.get("tier", fallback_tier) or fallback_tier).strip().lower()
    allowed_tiers = {"router", "low", "medium", "high"}
    if tier not in allowed_tiers:
        tier = fallback_tier if fallback_tier in allowed_tiers else "low"
    return {
        "tier": tier,
        "model": str(raw.get("model", "") or ""),
        "reasoning_effort": str(raw.get("reasoning_effort", "") or ""),
        "verbosity": str(raw.get("verbosity", "") or ""),
        "max_tokens": _int_like(raw.get("max_tokens", 0)),
    }


def _profile_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": profile.get("model") or None}
    if profile.get("reasoning_effort"):
        kwargs["reasoning_effort"] = str(profile["reasoning_effort"])
    if profile.get("verbosity"):
        kwargs["verbosity"] = str(profile["verbosity"])
    if _int_like(profile.get("max_tokens", 0)) > 0:
        kwargs["max_tokens"] = _int_like(profile.get("max_tokens", 0))
    if profile.get("service_tier"):
        kwargs["service_tier"] = str(profile["service_tier"])
    return kwargs


def _call_llm_json_with_profile(
    system_prompt: str,
    user_prompt: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    kwargs = _profile_kwargs(profile)
    try:
        return call_llm_json(system_prompt, user_prompt, **kwargs)
    except TypeError:
        # Backward compatibility for tests or custom monkeypatches that still
        # expose the old call_llm_json(..., model=...) signature.
        return call_llm_json(system_prompt, user_prompt, model=kwargs.get("model"))


def _supported_llm_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter optional LLM kwargs for monkeypatched legacy call signatures."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _call_llm_with_profile_kwargs(
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> str:
    return call_llm(
        system_prompt,
        user_prompt,
        **_supported_llm_kwargs(call_llm, kwargs),
    )


def _get_provider_registry(state: dict) -> Any:
    registry = state.get("_provider_registry")
    if registry is None:
        registry = ProviderCapabilityRegistry()
    return registry


def _get_profile(
    state: dict,
    tier: str,
    *,
    task_type: str,
    require_json: bool,
) -> dict[str, Any]:
    registry = _get_provider_registry(state)
    if hasattr(registry, "get_profile_for"):
        try:
            max_cost = str(state.get("_provider_max_cost", "high"))
            profile = _coerce_profile(
                registry.get_profile_for(  # type: ignore[attr-defined]
                    task_type,
                    tier=tier,
                    require_json=require_json,
                    max_cost=max_cost,
                ),
                fallback_tier=tier,
            )
            if state.get("_llm_service_tier"):
                profile["service_tier"] = str(state.get("_llm_service_tier"))
            return profile
        except Exception:
            pass
    model = _select_model(state, task_type, require_json=require_json)
    profile = {
        "tier": tier,
        "model": model or "",
        "reasoning_effort": "",
        "verbosity": "",
        "max_tokens": 0,
    }
    if state.get("_llm_service_tier"):
        profile["service_tier"] = str(state.get("_llm_service_tier"))
    return profile


def _router_context_flags(state: dict, *, node_name: str) -> dict[str, Any]:
    response = state.get("_final_response", {})
    draft = response.get("text", "") if isinstance(response, dict) else ""
    evidence = state.get("evidence", [])
    conflicts = state.get("conflicts", [])
    info_needs = state.get("info_needs", [])
    contract = state.get("turn_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    return {
        "turn_route": str(contract.get("route", "")),
        "turn_relation": str(contract.get("turn_relation", "")),
        "policy_action": str((contract.get("policy") or {}).get("action", "")),
        "needs_citation_check": node_name.startswith("reflection") and bool(evidence),
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "conflict_count": len(conflicts) if isinstance(conflicts, list) else 0,
        "info_need_count": len(info_needs) if isinstance(info_needs, list) else 0,
        "draft_answer_chars": len(str(draft or "")),
    }


def _select_model_profile(
    state: dict,
    task_type: str,
    *,
    node_name: str,
    require_json: bool,
    fallback_tier: str | None = None,
) -> dict[str, Any]:
    fallback = fallback_tier or _NODE_FALLBACK_TIERS.get(node_name, "low")
    router_profile = _get_profile(state, "router", task_type="general", require_json=True)
    question = _effective_question(state)
    router_payload = {
        "node_name": node_name,
        "task_type": task_type,
        "node_goal": _node_goal(node_name),
        "question": question[:1200],
        "retrieval_policy": str(state.get("retrieval_policy", "")),
        "route_decision": str(state.get("_route_decision", "")),
        "hop_count": _int_like(state.get("hop_count", 0)),
        "retry_count": _int_like(state.get("retry_count", 0)),
        "tool_trace_summary": _tool_trace_summary(
            state.get("_tool_execution_trace", [])
            if isinstance(state.get("_tool_execution_trace", []), list)
            else []
        ),
        "flags": _router_context_flags(state, node_name=node_name),
        "time_budget": _time_budget_snapshot(state),
        "fallback_tier": fallback,
        "allowed_tiers": ["low", "medium", "high"],
    }

    tier = fallback
    router_result: dict[str, Any] = {}
    try:
        router_result = _call_llm_json_with_profile(
            _LLM_ROUTER_SYSTEM,
            json.dumps(router_payload, ensure_ascii=False, default=str),
            router_profile,
        )
        candidate = str(router_result.get("tier", "")).strip().lower()
        if candidate in _MODEL_TIERS:
            tier = candidate
    except Exception as exc:
        logger.debug(
            "model_router fallback node=%s fallback_tier=%s err=%s",
            node_name,
            fallback,
            type(exc).__name__,
        )

    profile = _get_profile(state, tier, task_type=task_type, require_json=require_json)
    profile["_router_llm_row"] = _router_llm_row(
        state,
        node_name=node_name,
        router_profile=router_profile,
        selected_tier=tier,
    )
    profile["router"] = {
        "tier": tier,
        "fallback_tier": fallback,
        "reasoning": str(router_result.get("reasoning", ""))[:240]
        if isinstance(router_result, dict)
        else "",
        "confidence": router_result.get("confidence")
        if isinstance(router_result, dict)
        else None,
    }
    logger.debug(
        "model_router node=%s task=%s tier=%s model=%s reasoning_effort=%s",
        node_name,
        task_type,
        profile.get("tier"),
        profile.get("model"),
        profile.get("reasoning_effort"),
    )
    return profile


def _router_llm_row(
    state: dict,
    *,
    node_name: str,
    router_profile: dict[str, Any],
    selected_tier: str,
) -> dict[str, Any]:
    row = _llm_meta_row(state, node_name=node_name, call_kind="router_json", profile=router_profile)
    row["selected_tier"] = selected_tier
    return row


def _node_goal(node_name: str) -> str:
    return {
        "turn_interpreter": "Interpret the current turn and create the semantic contract for routing.",
        "retriever_planner": "Convert information needs into the smallest necessary retrieval plan.",
        "reflection_pre": "Decide whether current evidence is sufficient for the core question.",
        "reflection_post": "Check whether the draft is correct, citations match, and rewriting is needed.",
        "direct_answer": "Answer a low-risk question directly without citations.",
        "answer_composer": "Generate the final answer from evidence and context.",
        "response_compliance": "Check language, identity, scope, evidence, and policy compliance.",
    }.get(node_name, "Execute the current full graph node task.")


def _thread_key_from_context(context: dict[str, str]) -> dict[str, str] | None:
    required = ("platform", "guild_id", "channel_id", "user_id")
    if not all(context.get(k) for k in required):
        return None
    base_thread = context.get("thread_id") or "__default__"
    return {
        "platform": context["platform"],
        "guild_id": context["guild_id"],
        "channel_id": context["channel_id"],
        "thread_id": f"{base_thread}:user:{context['user_id']}",
    }


def _load_recent_messages(state: dict) -> list[dict[str, Any]]:
    raw = state.get("recent_messages", [])
    if isinstance(raw, list) and raw:
        return [row for row in raw if isinstance(row, dict)]

    svc = state.get("_memory_service")
    if svc is None or not hasattr(svc, "list_recent_message_events"):
        return []

    context = _get_message_context(state)
    platform = context.get("platform", "")
    user_id = context.get("user_id", "")
    if not platform or not user_id:
        return []

    try:
        rows = svc.list_recent_message_events(
            platform=platform,
            user_id=user_id,
            guild_id=context.get("guild_id") or None,
            channel_id=context.get("channel_id") or None,
            thread_id=context.get("thread_id") or None,
            limit=_budget_int(state, "memory_context_limit", 20),
        )
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def _format_conversation_context(messages: list[dict[str, Any]], *, limit_chars: int = 1800) -> str:
    if not messages:
        return "(none)"
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "message") or "message")
        content = re.sub(r"\s+", " ", str(msg.get("content", "") or "")).strip()
        if not content:
            continue
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        lines.append(f"{role}: {content}")
    text = "\n".join(lines).strip() or "(none)"
    if len(text) > limit_chars:
        text = text[-limit_chars:].lstrip()
    return text


def _conversation_context_from_state(state: dict) -> str:
    selected = state.get("selected_context", [])
    if isinstance(selected, list) and selected:
        return _format_conversation_context(
            [item for item in selected if isinstance(item, dict)],
            limit_chars=_policy_context_limit(state),
        )
    # A context string is trusted only after the interpreter has written a
    # contract. This prevents legacy gateways from injecting ordinary history.
    if isinstance(state.get("turn_contract"), dict):
        existing = str(state.get("conversation_context", "") or "").strip()
        if existing:
            return existing
    return "(none)"


def _policy_context_limit(state: dict) -> int:
    return policy_from_state(state).context.max_context_chars


def _load_memory_facts(state: dict) -> list[dict]:
    svc = state.get("_memory_service")
    if svc is None:
        return list(state.get("memory_facts", []))

    context = _get_message_context(state)
    platform = context.get("platform", "")
    user_id = context.get("user_id", "")

    facts: list[dict] = []
    try:
        if platform and context.get("guild_id") and context.get("channel_id"):
            channel_key = {
                "platform": platform,
                "guild_id": context["guild_id"],
                "channel_id": context["channel_id"],
            }
            facts.extend(svc.list_channel_facts(key=channel_key))
    except Exception:
        pass

    try:
        if platform and user_id:
            user_key = {"platform": platform, "user_id": user_id}
            facts.extend(svc.list_user_facts(key=user_key))
    except Exception:
        pass

    # 同 id 去重，并按置信度/更新时间排序后裁剪。
    by_id: dict[str, dict] = {}
    for fact in facts:
        fid = str(fact.get("id", ""))
        if fid:
            by_id[fid] = fact

    ordered = sorted(
        by_id.values(),
        key=lambda f: (
            float(f.get("confidence", 0.0)),
            int(f.get("updated_ts_ms", 0)),
        ),
        reverse=True,
    )
    max_facts = _budget_int(state, "max_memory_facts", 8)
    return ordered[:max_facts]


def _resume_thread_checkpoint(state: dict) -> dict[str, object] | None:
    svc = state.get("_memory_service")
    if svc is None:
        return None
    context = _get_message_context(state)
    key = _thread_key_from_context(context)
    if key is None:
        return None
    try:
        return svc.resume_thread(key=key)
    except Exception:
        return None


def _complete_thread_checkpoint(state: dict) -> None:
    svc = state.get("_memory_service")
    if svc is None:
        return
    context = _get_message_context(state)
    key = _thread_key_from_context(context)
    if key is None:
        return
    complete = getattr(svc, "complete_thread", None)
    if callable(complete):
        try:
            complete(key=key)
        except Exception:
            pass


def _question_from_user_message(state: dict) -> str:
    user_msg = state.get("user_message", {})
    return user_msg.get("content", "") if isinstance(user_msg, dict) else str(user_msg)


def _effective_question(state: dict) -> str:
    resolved = state.get("resolved_question", "")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return _question_from_user_message(state).strip()


def _memory_facts_to_evidence(facts: list[dict], namespace: str) -> list[dict]:
    evidence: list[dict] = []
    for fact in facts:
        fid = str(fact.get("id", ""))
        evidence.append(
            {
                "id": fid,
                "source": "memory",
                "title": f"fact:{fact.get('key', '')}",
                "url": f"memory://{namespace}/{fid}",
                "anchor": "kind:fact",
                "snippet": f"{fact.get('key', '')}={fact.get('value', '')}",
                "score": float(fact.get("confidence", 0.0)),
                "payload": {
                    "source": "memory",
                    "type": "fact",
                    "namespace": namespace,
                },
                "hash": fid,
                "retrieved_ts_ms": int(fact.get("updated_ts_ms", 0)),
            }
        )
    return evidence


def _collect_missing_params(info_needs: list[dict]) -> list[str]:
    fields: set[str] = set()
    for need in info_needs:
        if not isinstance(need, dict):
            continue
        if not bool(need.get("required", False)):
            continue
        if str(need.get("availability", "public")) != "user_owned":
            continue

        hints = need.get("hints", {})
        if isinstance(hints, dict):
            for key in hints.keys():
                if str(key).strip():
                    fields.add(str(key).strip())

        if not hints:
            kind = str(need.get("kind", "")).strip()
            if kind:
                fields.add(kind)

    if not fields:
        return ["missing_param"]
    return sorted(fields)


def _is_transient_llm_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    markers = (
        "timeout",
        "timed out",
        "serviceunavailable",
        "unavailable",
        "temporarily",
        "try again later",
        "rate limit",
        "429",
        "503",
        "handshake",
        "connection",
    )
    return any(m in text for m in markers)


def _call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    service_tier: str | None = None,
    max_tokens: int | None = None,
    max_attempts: int = 2,
    image_paths: list[str] | None = None,
) -> str:
    last_exc: Exception | None = None
    attempts = max(1, int(max_attempts))
    for idx in range(attempts):
        try:
            return _call_llm_with_profile_kwargs(
                system_prompt,
                user_prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                service_tier=service_tier,
                max_tokens=max_tokens,
                image_paths=image_paths,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if idx >= attempts - 1 or not _is_transient_llm_error(exc):
                raise
            time.sleep(min(2.0, 0.6 * (2**idx)))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unexpected llm retry state")


def _normalize_ask_user_question(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return ""

    # Keep text concise and avoid overlong noisy prompts.
    if len(text) > 220:
        text = text[:220].rstrip() + "..."
    return text


# ---------------------------------------------------------------------------
# TurnInterpreter — current-turn semantic contract
# ---------------------------------------------------------------------------

def _direct_reply_from_state(state: dict) -> dict[str, str]:
    message = state.get("user_message", {})
    if not isinstance(message, dict):
        return {}
    content = str(message.get("reply_to_content", "") or "").strip()
    message_id = str(message.get("reply_to_message_id", "") or "").strip()
    if not content or not message_id:
        return {}
    return {
        "message_id": message_id,
        "role": str(message.get("reply_to_role", "message") or "message"),
        "content": content,
    }


def _checkpoint_for_prompt(checkpoint: dict[str, object] | None) -> str:
    if not checkpoint:
        return "(none)"
    payload = checkpoint.get("context_payload", {})
    payload = payload if isinstance(payload, dict) else {}
    return json.dumps(
        {
            "missing_params": checkpoint.get("missing_params", []),
            "resume_node": checkpoint.get("resume_node", ""),
            "origin_question": payload.get("origin_question", ""),
            "ask_user_question": payload.get("ask_user_question", ""),
        },
        ensure_ascii=False,
        default=str,
    )


def _turn_interpreter_prompt(
    state: dict,
    *,
    question: str,
    facts: list[dict],
    checkpoint: dict[str, object] | None,
    direct_reply: dict[str, str],
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    message = state.get("user_message", {})
    platform_hint = message.get("platform_locale_hint", "") if isinstance(message, dict) else ""
    prompt = prompts.TURN_INTERPRETER_USER.format(
        question=question,
        product_policy=policy_from_state(state).render_prompt_block(),
        platform_locale_hint=str(platform_hint or "(none)"),
        direct_reply=json.dumps(direct_reply, ensure_ascii=False, default=str) if direct_reply else "(none)",
        checkpoint=_checkpoint_for_prompt(checkpoint),
        memory_facts=json.dumps(facts[:8], ensure_ascii=False, default=str)[:1200] or "(none)",
        evidence_count=len(state.get("evidence", [])) if isinstance(state.get("evidence", []), list) else 0,
        time_budget=_time_budget_prompt(state),
    )
    if recent_history:
        prompt += "\nQuoted recent history selected for this second interpretation pass:\n"
        prompt += _format_conversation_context(
            recent_history,
            limit_chars=policy_from_state(state).context.max_context_chars,
        )
    return prompt


def _run_turn_interpreter(
    state: dict,
    user_prompt: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
    policy = policy_from_state(state)
    tiers = [policy.model_profiles.turn_interpreter, policy.model_profiles.turn_interpreter_fallback]
    seen: set[str] = set()
    trace: dict[str, Any] = {}
    for index, tier in enumerate(tiers):
        tier = str(tier or "low")
        if tier in seen:
            continue
        seen.add(tier)
        profile = _get_profile(state, tier, task_type="planning", require_json=True)
        try:
            result = _call_llm_json_with_profile(
                prompts.TURN_INTERPRETER_SYSTEM,
                user_prompt,
                profile,
            )
            trace = _llm_update_for_call(
                state,
                node_name="turn_interpreter",
                call_kind="business_json",
                profile=profile,
            )
            if isinstance(result, dict):
                return result, trace, index + 1
        except Exception as exc:
            logger.warning(
                "turn_interpreter failed tier=%s attempt=%d type=%s",
                tier,
                index + 1,
                type(exc).__name__,
            )
    return None, trace, len(seen)


def turn_interpreter(state: dict) -> dict:
    """Interpret the current turn once, then load history only when requested."""
    policy = policy_from_state(state)
    raw_question = _question_from_user_message(state)
    facts = _load_memory_facts(state)
    checkpoint = _resume_thread_checkpoint(state)
    direct_reply = _direct_reply_from_state(state)
    first_prompt = _turn_interpreter_prompt(
        state,
        question=raw_question,
        facts=facts,
        checkpoint=checkpoint,
        direct_reply=direct_reply,
    )
    result, llm_trace_update, passes = _run_turn_interpreter(state, first_prompt)
    if result is None:
        locale = policy.language.default_locale
        return {
            "_turn_interpreter_error": True,
            "_terminal_interpretation_error": True,
            "turn_interpretation_passes": passes,
            "contract_repair_count": int(state.get("contract_repair_count", 0) or 0),
            "memory_facts": facts,
            "response_locale": locale,
            "locale": locale,
            "_final_response": {
                "request_id": str(state.get("request_id", "unknown")),
                "text": policy.message("generation_failed", locale),
                "citations": [],
            },
            **llm_trace_update,
        }

    from .turn_contract import derive_legacy_state, normalize_turn_contract

    contract = normalize_turn_contract(
        result,
        policy,
        current_message=raw_question,
        confirmed_preference=state.get("confirmed_locale_preference"),
        available_context_ids={direct_reply["message_id"]} if direct_reply else set(),
    )
    recent_messages: list[dict[str, Any]] = []
    if contract["context"]["requirement"] == "recent_history":
        recent_messages = _load_recent_messages(state)
        available_ids = {
            str(item.get("id") or item.get("message_id"))
            for item in recent_messages
            if isinstance(item, dict) and (item.get("id") or item.get("message_id"))
        }
        if direct_reply:
            available_ids.add(direct_reply["message_id"])
        second_prompt = _turn_interpreter_prompt(
            state,
            question=raw_question,
            facts=facts,
            checkpoint=checkpoint,
            direct_reply=direct_reply,
            recent_history=recent_messages,
        )
        second_result, second_trace, second_passes = _run_turn_interpreter(state, second_prompt)
        passes += second_passes
        if second_result is not None:
            from .turn_contract import merge_contextual_contract

            contract = merge_contextual_contract(
                contract,
                second_result,
                policy,
                current_message=raw_question,
                confirmed_preference=state.get("confirmed_locale_preference"),
                available_context_ids=available_ids,
            )
            llm_trace_update = {**llm_trace_update, **second_trace}

    if contract["context"]["requirement"] == "direct_reply" and direct_reply:
        selected_ids = set(contract["context"]["selected_message_ids"])
        selected_context = [direct_reply] if not selected_ids or direct_reply["message_id"] in selected_ids else []
    elif contract["context"]["requirement"] == "recent_history":
        selected_ids = set(contract["context"]["selected_message_ids"])
        selected_context = ([direct_reply] if direct_reply and direct_reply["message_id"] in selected_ids else []) + [
            item for item in recent_messages
            if str(item.get("id") or item.get("message_id")) in selected_ids
        ]
    else:
        selected_context = []

    if checkpoint and contract["turn_relation"] == "clarification_answer":
        _complete_thread_checkpoint(state)
    elif checkpoint and contract["turn_relation"] == "new_task":
        _complete_thread_checkpoint(state)

    legacy = derive_legacy_state(
        contract,
        policy,
        confirmed_preference=state.get("confirmed_locale_preference"),
    )
    if state.get("force_retrieval") and contract["route"] == "direct" and raw_question.strip():
        legacy["_route_decision"] = "has_needs"
        legacy["retrieval_policy"] = "single"
        legacy["info_needs"] = [{
            "kind": "concept_gap",
            "question": raw_question,
            "required": False,
            "availability": "public",
            "purpose": "Retrieve public material before answering the explicit retrieval request.",
        }]
    update: dict[str, Any] = {
        **legacy,
        "memory_facts": facts,
        "recent_messages": recent_messages,
        "selected_context": selected_context,
        "conversation_context": _format_conversation_context(selected_context, limit_chars=policy.context.max_context_chars),
        "budget": _merge_policy_budget(state, legacy["retrieval_policy"]),
        "turn_interpretation_passes": passes,
        "contract_repair_count": int(state.get("contract_repair_count", 0) or 0),
        **llm_trace_update,
    }
    if contract["policy"]["action"] == "refuse":
        update.update(_force_policy_response({**state, **update}, question=contract["resolved_request"]))
    logger.debug(
        "turn_interpreter route=%s retrieval_policy=%s info_needs=%d context=%s",
        contract["route"],
        contract["retrieval_policy"],
        len(contract["info_needs"]),
        contract["context"]["requirement"],
    )
    return update


# ---------------------------------------------------------------------------
# M7-T2: RetrieverPlanner — 检索规划
# ---------------------------------------------------------------------------

def retriever_planner(state: dict) -> dict:
    """调 LLM 根据 info_needs 生成 RetrievalPlan (JSON)。"""
    question = _effective_question(state)
    reflection_hints = state.get("reflection_hints", {})
    if isinstance(reflection_hints, dict):
        next_query = str(reflection_hints.get("next_query", "")).strip()
        if next_query:
            question = next_query
    info_needs = state.get("info_needs", [])
    retry_count = state.get("retry_count", 0)
    facts = state.get("memory_facts", [])

    user_prompt = prompts.RETRIEVER_PLANNER_USER.format(
        turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
        info_needs=json.dumps(info_needs, ensure_ascii=False, default=str),
        question=question,
        retrieval_policy=str(state.get("retrieval_policy", "single")),
        retry_count=retry_count,
        conversation_context=_conversation_context_from_state(state),
        time_budget=_time_budget_prompt(state),
    )
    if facts:
        user_prompt += "\nAvailable memory facts:\n" + json.dumps(facts, ensure_ascii=False, default=str)[:800]

    profile = _select_model_profile(
        state,
        "planning",
        node_name="retriever_planner",
        require_json=True,
    )
    planner_system_prompt = (
        prompts.RETRIEVER_PLANNER_SYSTEM
        + "\n\nAvailable source registry:\n"
        + format_source_registry_for_prompt()
    )
    try:
        plan = _call_llm_json_with_profile(
            planner_system_prompt,
            user_prompt,
            profile,
        )
        llm_trace_update = _llm_update_for_call(
            state,
            node_name="retriever_planner",
            call_kind="business_json",
            profile=profile,
        )
    except Exception:
        plan = {
            "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
            "rationale": "fallback plan",
            "steps": [
                {
                    "step_id": "step_1",
                    "tool": "qdrant_search",
                    "query": question,
                    "filters": {},
                    "top_k": 5,
                }
            ],
            "parallel_groups": [["step_1"]],
            "budget": {},
        }
        llm_trace_update = {}

    plan.setdefault("plan_id", f"plan_{uuid.uuid4().hex[:8]}")
    plan.setdefault("rationale", "")
    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    max_tool_calls = _budget_int(state, "max_tool_calls", 3)
    max_evidence_chunks = _budget_int(state, "max_evidence_chunks", 8)
    supported_tools = {"qdrant_search", "discourse_query", "github_search", "memory_fetch"}
    normalized_steps: list[dict] = []
    for raw_step in raw_steps[:max_tool_calls]:
        if not isinstance(raw_step, dict):
            continue
        step_top_k = raw_step.get("top_k", min(5, max_evidence_chunks))
        try:
            step_top_k = int(step_top_k)
        except (TypeError, ValueError):
            step_top_k = min(5, max_evidence_chunks)
        step_top_k = max(1, min(step_top_k, 20))
        raw_tool = str(raw_step.get("tool", "qdrant_search")).strip() or "qdrant_search"
        tool = raw_tool if raw_tool in supported_tools else "qdrant_search"
        raw_filters = raw_step.get("filters", {}) if isinstance(raw_step.get("filters", {}), dict) else {}
        filters, filter_notes = normalize_tool_filters(tool, raw_filters)
        step = {
            "step_id": str(raw_step.get("step_id", f"step_{uuid.uuid4().hex[:4]}")),
            "tool": tool,
            "query": str(raw_step.get("query", question)).strip() or question,
            "filters": filters,
            "top_k": step_top_k,
        }
        regex_queries = _normalize_regex_queries_for_step(raw_step.get("regex_queries"))
        if tool == "qdrant_search" and regex_queries:
            step["regex_queries"] = regex_queries
        if filter_notes:
            step["filter_notes"] = filter_notes
        time_range = str(raw_step.get("time_range", "") or "").strip()
        if time_range:
            step["time_range"] = time_range
        normalized_steps.append(step)

    if not normalized_steps:
        normalized_steps = [
            {
                "step_id": "step_1",
                "tool": "qdrant_search",
                "query": question,
                "filters": {},
                "top_k": min(5, max_evidence_chunks),
            }
        ]

    step_ids = {step["step_id"] for step in normalized_steps}
    raw_groups = plan.get("parallel_groups", [])
    groups: list[list[str]] = []
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if not isinstance(group, list):
                continue
            valid = [str(step_id) for step_id in group if str(step_id) in step_ids]
            if valid:
                groups.append(valid)
    if not groups:
        groups = [[step["step_id"] for step in normalized_steps]]

    return {
        "retrieval_plan": {
            "plan_id": str(plan["plan_id"]),
            "rationale": str(plan["rationale"]),
            "steps": normalized_steps,
            "parallel_groups": groups,
            "budget": {
                "max_tool_calls": max_tool_calls,
                "max_evidence_chunks": max_evidence_chunks,
            },
        },
        **llm_trace_update,
    }


def _normalize_regex_queries_for_step(raw_queries: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_queries, list):
        return []
    allowed_fields = {"title", "keywords", "anchor", "url", "summary", "raw_text"}
    normalized: list[dict[str, Any]] = []
    for raw_query in raw_queries[:3]:
        if not isinstance(raw_query, dict):
            continue
        pattern = str(raw_query.get("pattern", "") or "").strip()
        if not pattern:
            continue
        label = str(raw_query.get("label", "") or "").strip()[:80]
        reason = str(raw_query.get("reason", "") or "").strip()[:160]
        raw_fields = raw_query.get("fields", [])
        fields: list[str] = []
        if isinstance(raw_fields, list):
            for raw_field in raw_fields:
                field = str(raw_field or "").strip()
                if field in allowed_fields and field not in fields:
                    fields.append(field)
        entry: dict[str, Any] = {"label": label, "pattern": pattern[:160]}
        if fields:
            entry["fields"] = fields
        if reason:
            entry["reason"] = reason
        normalized.append(entry)
    return normalized


# ---------------------------------------------------------------------------
# M7-T3: RetrievalExecutor — 执行检索计划
# ---------------------------------------------------------------------------

def retrieval_executor(state: dict) -> dict:
    """从 retrieval_plan 取 steps，逐步构造 ToolCallRequest 并调 ToolRuntime。"""
    from nervos_brain.tool_runtime import (
        build_tool_call_request,
        check_idempotency,
        execute_tool,
    )
    from nervos_brain.tool_runtime.handlers import TOOL_HANDLERS

    plan = state.get("retrieval_plan", {})
    steps = plan.get("steps", [])
    request_id = state.get("request_id", "unknown")
    max_tool_calls = _budget_int(state, "max_tool_calls", 3)
    max_evidence_chunks = _budget_int(state, "max_evidence_chunks", 8)
    existing_evidence = list(state.get("evidence", []))
    context = _get_message_context(state)

    tool_calls = 0
    tool_traces: list[dict[str, Any]] = []

    for step in steps[:max_tool_calls]:
        if len(existing_evidence) >= max_evidence_chunks:
            tool_traces.append(
                {
                    "step_id": str(step.get("step_id", "step_0")),
                    "tool": str(step.get("tool", "qdrant_search")),
                    "status": "skipped",
                    "error_code": "ERR_BUDGET_EXCEEDED",
                    "evidence_count": 0,
                }
            )
            break

        tool = step.get("tool", "qdrant_search")
        query = str(step.get("query", ""))
        raw_filters = step.get("filters", {}) if isinstance(step.get("filters", {}), dict) else {}
        filters, filter_notes = normalize_tool_filters(str(tool), raw_filters)
        step_filter_notes = step.get("filter_notes", [])
        if isinstance(step_filter_notes, list):
            for note in step_filter_notes:
                note_text = str(note).strip()
                if note_text and note_text not in filter_notes:
                    filter_notes.append(note_text)
        elif step_filter_notes:
            note_text = str(step_filter_notes).strip()
            if note_text and note_text not in filter_notes:
                filter_notes.append(note_text)
        top_k = step.get("top_k", 5)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))
        step_id = str(step.get("step_id", "step_0"))
        args = _tool_args_for_step(
            state=state,
            step=step,
            tool=str(tool),
            query=query,
            filters=filters,
            top_k=top_k,
            context=context,
        )
        if not isinstance(args, dict):
            tool_traces.append(
                {
                    "step_id": step_id,
                    "tool": str(tool),
                    "status": "skipped",
                    "error_code": "ERR_TOOL_SCHEMA_INVALID",
                    "evidence_count": 0,
                }
            )
            continue

        result, trace_row, counted_call = _execute_retrieval_tool_call(
            request_id=str(request_id),
            step_id=step_id,
            tool=str(tool),
            args=args,
            build_tool_call_request=build_tool_call_request,
            check_idempotency=check_idempotency,
            execute_tool=execute_tool,
            handlers=TOOL_HANDLERS,
        )
        if filter_notes:
            trace_row["filter_notes"] = filter_notes
        if counted_call:
            tool_calls += 1
        tool_traces.append(trace_row)
        if result is None:
            continue

        evidence_items = result.get("evidence", []) if isinstance(result, dict) else []
        evidence_count = len(evidence_items) if isinstance(evidence_items, list) else 0
        if (
            result.get("ok")
            and str(tool) == "qdrant_search"
            and should_retry_qdrant_without_filters(filters, evidence_count)
            and tool_calls < max_tool_calls
        ):
            fallback_args = dict(args)
            fallback_args["filters"] = {}
            fallback_result, fallback_trace, fallback_counted = _execute_retrieval_tool_call(
                request_id=str(request_id),
                step_id=f"{step_id}_unfiltered",
                tool=str(tool),
                args=fallback_args,
                build_tool_call_request=build_tool_call_request,
                check_idempotency=check_idempotency,
                execute_tool=execute_tool,
                handlers=TOOL_HANDLERS,
            )
            fallback_trace["fallback_reason"] = "empty_filtered_qdrant_search"
            tool_traces.append(fallback_trace)
            if fallback_counted:
                tool_calls += 1
            if fallback_result is not None:
                fallback_items = fallback_result.get("evidence", []) if isinstance(fallback_result, dict) else []
                if isinstance(fallback_items, list) and fallback_items:
                    result = fallback_result
                    evidence_items = fallback_items

        if result.get("ok") and isinstance(evidence_items, list):
            existing_evidence.extend(evidence_items)
            existing_evidence = existing_evidence[:max_evidence_chunks]

    tool_summary = _tool_trace_summary(tool_traces)
    logger.debug(
        "retrieval_executor request_id=%s tool_calls=%d evidence=%d %s",
        state.get("request_id", "unknown"),
        tool_calls,
        len(existing_evidence),
        tool_summary,
    )
    return {
        "evidence": existing_evidence,
        "_tool_calls_executed": tool_calls,
        "_tool_execution_trace": tool_traces,
        "_tool_execution_summary": tool_summary,
        "hop_count": _int_like(state.get("hop_count", 0)) + 1,
    }


# ---------------------------------------------------------------------------
# M7-T4: EvidenceMerger — 证据去重与冲突检测
# ---------------------------------------------------------------------------

def evidence_merger(state: dict) -> dict:
    """去重 (by hash) + 简单冲突检测 (source/version 比对)。"""
    evidence_list = state.get("evidence", [])

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for ev in evidence_list:
        h = ev.get("hash", "")
        if not h:
            h = hashlib.sha256(json.dumps(ev, sort_keys=True, default=str).encode()).hexdigest()
            ev["hash"] = h
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(ev)

    conflicts: list[dict] = []
    for i, a in enumerate(deduped):
        for b in deduped[i + 1:]:
            a_payload = a.get("payload", {})
            b_payload = b.get("payload", {})
            if (
                a_payload.get("source") == b_payload.get("source")
                and a_payload.get("version", "") != b_payload.get("version", "")
                and a_payload.get("version") and b_payload.get("version")
            ):
                conflicts.append({
                    "a_id": a.get("id", ""),
                    "b_id": b.get("id", ""),
                    "reason": "version_mismatch",
                })

    return {"evidence": deduped, "conflicts": conflicts}


# ---------------------------------------------------------------------------
# M7-T5: Reflection (通用反思，覆盖证据与回答草稿)
# ---------------------------------------------------------------------------

def _stage_label(stage: str) -> str:
    if stage == "post_answer":
        return "post-answer review"
    return "pre-answer evidence review"


def _summarize_evidence(evidence: list[dict]) -> str:
    return "\n".join(
        f"- [{e.get('source', '?')}] {e.get('title', '?')}: {e.get('snippet', '')[:200]}"
        for e in evidence[:10]
    ) or "(no evidence)"


def _summarize_conflicts(conflicts: list[dict]) -> str:
    return "\n".join(
        f"- {c.get('a_id', '?')} vs {c.get('b_id', '?')}: {c.get('reason', '?')}"
        for c in conflicts
    ) or "(no conflicts)"


def _summarize_citations(citations: list[dict]) -> str:
    return "\n".join(
        f"{c.get('label', '?')}: {c.get('title', '?')} ({c.get('url', '')})"
        for c in citations[:10]
        if isinstance(c, dict)
    ) or "(no citations)"


def _has_reference_section(text: str, headings: list[str]) -> bool:
    patterns = [re.escape(item.strip()) for item in headings if str(item).strip()]
    if not patterns:
        return False
    return bool(
        re.search(
            rf"(?im)^\s{{0,3}}(?:#{{1,6}}\s*)?(?:{'|'.join(patterns)})\b",
            text,
        )
    )


def _append_reference_section(
    text: str,
    citations: list[dict[str, Any]],
    *,
    locale: str | None = None,
    policy: Any | None = None,
) -> str:
    active_policy = policy or policy_from_state({})
    selected_locale = active_policy.resolve_locale(locale)
    headings = [
        str(messages.get("references_heading", ""))
        for messages in active_policy.messages.values()
        if isinstance(messages, dict)
    ]
    if not citations or _has_reference_section(text, headings):
        return text.strip()

    heading = active_policy.message("references_heading", selected_locale, default="References")
    fallback_title = active_policy.message("source_label", selected_locale, default="Source")

    rows: list[str] = []
    for idx, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            continue
        label = str(citation.get("label") or f"[{idx}]").strip()
        title = str(citation.get("title") or "").strip()
        url = str(citation.get("url") or "").strip()
        anchor = str(citation.get("anchor") or "").strip()
        if not title:
            title = url or anchor or fallback_title
        if url:
            rows.append(f"{label} **{title}**\n{url}")
        else:
            rows.append(f"{label} **{title}**")

    if not rows:
        return text.strip()
    return text.strip() + f"\n\n## {heading}\n\n" + "\n\n".join(rows)


_INLINE_CITE_RE = re.compile(r"\{\{\s*cite\s*:\s*E(\d+)\s*\}\}", re.IGNORECASE)
_LEGACY_CITE_RE = re.compile(r"\[\d+\]")
_FENCED_CODE_RE = re.compile(r"(```[\s\S]*?```)")


def _has_inline_citation_tags(text: str) -> bool:
    return bool(_INLINE_CITE_RE.search(str(text or "")))


def _compile_inline_citations(
    text: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    evidence_by_id: dict[str, dict[str, Any]] = {
        f"E{idx}": ev
        for idx, ev in enumerate(evidence[:10], start=1)
        if isinstance(ev, dict)
    }
    label_by_evidence_id: dict[str, str] = {}
    citations: list[dict[str, str]] = []

    def citation_for(evidence_id: str) -> str:
        ev = evidence_by_id.get(evidence_id)
        if not ev:
            return ""
        label = label_by_evidence_id.get(evidence_id)
        if label:
            return label
        label = f"[{len(citations) + 1}]"
        label_by_evidence_id[evidence_id] = label
        citations.append(
            {
                "label": label,
                "url": str(ev.get("url", "") or ""),
                "anchor": str(ev.get("anchor", "") or ""),
                "title": str(ev.get("title", "") or ""),
            }
        )
        return label

    def replace_tags(chunk: str) -> str:
        def repl(match: re.Match[str]) -> str:
            return citation_for(f"E{match.group(1)}")

        return _INLINE_CITE_RE.sub(repl, _LEGACY_CITE_RE.sub("", chunk))

    parts = _FENCED_CODE_RE.split(str(text or ""))
    rendered: list[str] = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            rendered.append(part)
        else:
            rendered.append(replace_tags(part))
    cleaned = "".join(rendered)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), citations


def _required_questions(info_needs: list[dict]) -> list[str]:
    questions: list[str] = []
    for need in info_needs:
        if not isinstance(need, dict):
            continue
        if not bool(need.get("required", False)):
            continue
        if str(need.get("availability", "public")) != "user_owned":
            continue
        question = str(need.get("question", "")).strip()
        if question:
            questions.append(question)
    return questions


def _reflection_has_user_required_clarification(
    info_needs: list[dict],
    hints: dict[str, Any],
) -> bool:
    """Allow AskUser only for a typed user-owned required info need."""
    _ = hints
    return any(
        isinstance(need, dict)
        and bool(need.get("required", False))
        and str(need.get("availability", "public")) == "user_owned"
        and bool(str(need.get("question", "") or "").strip())
        for need in info_needs
    )


def _pre_answer_budget_exhausted(state: dict, *, reflection_round: int) -> bool:
    hop_count = max(_int_like(state.get("hop_count", 0)), _int_like(state.get("retry_count", 0)))
    max_hops = _budget_int(state, "max_hops", 3)
    max_pre_rounds = _budget_int(state, "max_reflection_rounds_pre", 2)
    return hop_count >= max_hops or reflection_round >= max_pre_rounds


def _normalize_reflection_decision(raw: str, *, stage: str) -> str:
    decision = str(raw).strip()
    allowed = {"continue_retrieval", "ask_user", "revise_answer", "accept_answer"}
    if decision in allowed:
        return decision
    if stage == "post_answer":
        return "revise_answer"
    return "continue_retrieval"


def _heuristic_reflection_pre(state: dict) -> dict[str, Any]:
    info_needs = state.get("info_needs", [])
    evidence = state.get("evidence", [])
    conflicts = state.get("conflicts", [])
    question = _effective_question(state)

    req_questions = [
        question
        for need in (info_needs if isinstance(info_needs, list) else [])
        if isinstance(need, dict)
        and bool(need.get("required", False))
        and str(need.get("availability", "public")) == "user_owned"
        for question in [str(need.get("question", "") or "").strip()]
        if question
    ]
    if req_questions:
        return {
            "decision": "ask_user",
            "reasoning": "A required parameter is missing; ask the user before continuing.",
            "uncertainty_score": 0.92,
            "clarify_question": req_questions[0],
            "missing_params": _collect_missing_params(info_needs),
        }

    if conflicts:
        return {
            "decision": "continue_retrieval",
            "reasoning": "Public evidence has a version conflict; continue retrieval or state the boundary instead of asking the user to judge it.",
            "uncertainty_score": 0.58,
            "next_query": question,
        }

    if not evidence:
        return {
            "decision": "continue_retrieval",
            "reasoning": "There is no evidence yet; continue retrieval.",
            "uncertainty_score": 0.35,
            "next_query": question,
        }

    if len(evidence) < 2:
        return {
            "decision": "continue_retrieval",
            "reasoning": "Evidence coverage is thin; one additional retrieval hop is appropriate.",
            "uncertainty_score": 0.40,
            "next_query": question,
        }

    return {
        "decision": "accept_answer",
        "reasoning": "The evidence supports an answer; proceed to answer composition.",
        "uncertainty_score": 0.20,
    }


def _heuristic_reflection_post(state: dict) -> dict[str, Any]:
    response = state.get("_final_response", {})
    if not isinstance(response, dict):
        response = {}
    text = str(response.get("text", "") or "")
    citations = response.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    evidence = state.get("evidence", [])
    if isinstance(evidence, list) and _has_inline_citation_tags(text):
        text, citations = _compile_inline_citations(text, evidence)
    conflicts = state.get("conflicts", [])
    compose_error = state.get("_compose_error")
    question = _effective_question(state)

    if state.get("_direct_answer") and not isinstance(compose_error, dict):
        if text.strip():
            return {
                "decision": "accept_answer",
                "reasoning": "The low-risk direct answer needs no citation and can be published.",
                "uncertainty_score": 0.18,
            }
        return {
            "decision": "revise_answer",
            "reasoning": "The direct answer is empty and must be rewritten.",
            "uncertainty_score": 0.85,
            "revise_instructions": "Give a concise direct answer and do not invent details that require retrieval.",
        }

    if isinstance(compose_error, dict):
        return {
            "decision": "revise_answer",
            "reasoning": "Answer generation failed; retry generation first.",
            "uncertainty_score": 0.93,
            "revise_instructions": "Regenerate the complete answer from the existing evidence and ensure citation labels are correct.",
        }

    if conflicts:
        return {
            "decision": "ask_user",
            "reasoning": "The evidence conflict is unresolved, so the answer may not be reliable.",
            "uncertainty_score": 0.82,
            "clarify_question": policy_from_state(state).message(
                "clarify_conflict",
                state.get("response_locale") or state.get("locale"),
            ),
        }

    if not text.strip():
        return {
            "decision": "revise_answer",
            "reasoning": "The answer draft is empty and must be rewritten.",
            "uncertainty_score": 0.90,
            "revise_instructions": "Give a concise conclusion with verifiable citations.",
        }

    labels_in_text = set(re.findall(r"\[\d+\]", text))
    labels_in_citations = {
        str(c.get("label", "")) for c in citations if isinstance(c, dict)
    }
    if labels_in_text and not labels_in_text.issubset(labels_in_citations):
        return {
            "decision": "revise_answer",
            "reasoning": "Citation labels in the body do not match the citation list.",
            "uncertainty_score": 0.72,
            "revise_instructions": "Fix the mapping between citation labels in the body and the citations list.",
        }

    if not evidence:
        return {
            "decision": "continue_retrieval",
            "reasoning": "The draft lacks evidence support; return to retrieval.",
            "uncertainty_score": 0.75,
            "next_query": question,
        }

    if not citations:
        return {
            "decision": "revise_answer",
            "reasoning": "The answer has no citations.",
            "uncertainty_score": 0.60,
            "revise_instructions": "Add citation labels for the important factual claims.",
        }

    max_considered = max(1, min(len(evidence), 10))
    coverage = len(citations[:10]) / max_considered
    if coverage < 0.4:
        return {
            "decision": "revise_answer",
            "reasoning": "Citation coverage is too low.",
            "uncertainty_score": 0.55,
            "revise_instructions": "Increase citation coverage for the important claims.",
        }

    return {
        "decision": "accept_answer",
        "reasoning": "The answer and citations are broadly consistent and can be published.",
        "uncertainty_score": 0.18,
    }


def _reflection_fallback(state: dict, *, stage: str) -> dict[str, Any]:
    if stage == "post_answer":
        return _heuristic_reflection_post(state)
    return _heuristic_reflection_pre(state)


def _reflect(state: dict, *, stage: str) -> dict[str, Any]:
    question = _effective_question(state)
    info_needs = state.get("info_needs", [])
    if not isinstance(info_needs, list):
        info_needs = []
    memory_facts = state.get("memory_facts", [])
    if not isinstance(memory_facts, list):
        memory_facts = []
    evidence = state.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    conflicts = state.get("conflicts", [])
    if not isinstance(conflicts, list):
        conflicts = []
    response = state.get("_final_response", {})
    if not isinstance(response, dict):
        response = {}
    draft_answer = str(response.get("text", "") or "")
    compose_error = state.get("_compose_error")
    citations = response.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    review_answer = draft_answer
    review_citations = citations
    if stage == "post_answer" and _has_inline_citation_tags(draft_answer):
        review_answer, review_citations = _compile_inline_citations(draft_answer, evidence)

    round_key = "_reflection_rounds_post" if stage == "post_answer" else "_reflection_rounds_pre"
    reflection_round = _int_like(state.get(round_key, 0)) + 1
    stage_label = _stage_label(stage)

    user_prompt = prompts.REFLECTION_USER.format(
        stage=stage,
        locale=str(state.get("response_locale") or state.get("locale") or policy_from_state(state).language.default_locale),
        turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
        question=question,
        info_needs=json.dumps(info_needs, ensure_ascii=False, default=str),
        memory_facts=json.dumps(memory_facts, ensure_ascii=False, default=str)[:800],
        evidence_count=len(evidence),
        evidence_summary=_summarize_evidence(evidence),
        conflict_count=len(conflicts),
        conflicts_summary=_summarize_conflicts(conflicts),
        draft_answer=review_answer[:2400] or "(no draft)",
        citations_summary=_summarize_citations(review_citations),
        hop_count=_int_like(state.get("hop_count", 0)),
        reflection_round=reflection_round,
        time_budget=_time_budget_prompt(state),
    )

    node_name = "reflection_post" if stage == "post_answer" else "reflection_pre"
    profile = _select_model_profile(
        state,
        "reflection",
        node_name=node_name,
        require_json=True,
    )
    try:
        raw_result = _call_llm_json_with_profile(
            prompts.REFLECTION_SYSTEM.format(stage_label=stage_label),
            user_prompt,
            profile,
        )
        llm_trace_update = _llm_update_for_call(
            state,
            node_name=node_name,
            call_kind="business_json",
            profile=profile,
        )
    except Exception:
        raw_result = _reflection_fallback(state, stage=stage)
        llm_trace_update = {}

    if not isinstance(raw_result, dict):
        raw_result = _reflection_fallback(state, stage=stage)

    # 兼容旧格式：DocGrader / SelfCheck 输出。
    if "decision" not in raw_result and "grade" in raw_result:
        grade = str(raw_result.get("grade", "need_more"))
        raw_result["decision"] = (
            "continue_retrieval"
            if grade == "need_more"
            else "accept_answer"
        )
        raw_result.setdefault("uncertainty_score", 0.4 if grade == "need_more" else 0.2)
    if "decision" not in raw_result and "pass" in raw_result:
        passed = bool(raw_result.get("pass", False))
        raw_result["decision"] = "accept_answer" if passed else "revise_answer"
        raw_result.setdefault("uncertainty_score", 0.2 if passed else 0.6)

    decision = _normalize_reflection_decision(raw_result.get("decision", ""), stage=stage)
    reasoning = str(raw_result.get("reasoning", "")).strip()
    try:
        uncertainty = float(raw_result.get("uncertainty_score", 0.5))
    except (TypeError, ValueError):
        uncertainty = 0.5
    uncertainty = max(0.0, min(1.0, uncertainty))

    hints: dict[str, Any] = {}
    for key in ("missing_params", "clarify_question", "next_query", "revise_instructions"):
        if key in raw_result and raw_result[key] not in (None, ""):
            hints[key] = raw_result[key]

    # 防止 pre-answer 阶段过早追问：
    # ask_user 只允许用于用户私有/现场必填信息；公开资料缺口或版本冲突应继续检索，
    # 若预算已耗尽且已有证据，则基于现有证据回答并说明边界。
    req_questions = [
        str(need.get("question", "") or "").strip()
        for need in info_needs
        if isinstance(need, dict)
        and bool(need.get("required", False))
        and str(need.get("availability", "public")) == "user_owned"
        and str(need.get("question", "") or "").strip()
    ]
    has_user_required_clarification = _reflection_has_user_required_clarification(info_needs, hints)
    pre_budget_exhausted = (
        stage == "pre_answer"
        and _pre_answer_budget_exhausted(state, reflection_round=reflection_round)
    )
    if stage == "pre_answer" and decision == "ask_user":
        if not has_user_required_clarification:
            decision = "accept_answer" if evidence and pre_budget_exhausted else "continue_retrieval"
            hints.pop("clarify_question", None)
            hints.pop("missing_params", None)
            if decision == "continue_retrieval" and not hints.get("next_query"):
                hints["next_query"] = question
            if not reasoning:
                reasoning = "No missing required private user information was identified; do not ask the user."

    if stage == "post_answer" and decision == "ask_user" and not has_user_required_clarification:
        hints.pop("clarify_question", None)
        hints.pop("missing_params", None)
        if evidence:
            decision = "revise_answer"
            hints.setdefault(
                "revise_instructions",
                "The previous answer did not correctly cover the current question. Rewrite it from the current question and evidence; do not ask the user.",
            )
        else:
            decision = "continue_retrieval"
            hints.setdefault("next_query", question)
        if not reasoning:
            reasoning = "Post-answer ask_user has no required private user information; rewrite or continue retrieval instead."

    # post-answer 若生成器已报错，不能 accept_answer。
    if stage == "post_answer" and (
        isinstance(compose_error, dict)
    ):
        decision = "revise_answer"
        uncertainty = max(uncertainty, 0.90)
        if not reasoning:
            reasoning = "Answer generation failed and must be retried."
        hints.setdefault("revise_instructions", "Retry answer generation from the existing evidence.")

    if (
        stage == "pre_answer"
        and decision == "continue_retrieval"
        and evidence
        and pre_budget_exhausted
        and not has_user_required_clarification
    ):
        decision = "accept_answer"
        hints.pop("clarify_question", None)
        hints.pop("missing_params", None)

    update: dict[str, Any] = {
        "reflection_stage": stage,
        "reflection_decision": decision,
        "reflection_reasoning": reasoning,
        "uncertainty_score": uncertainty,
        "reflection_hints": hints,
        "reflection_round": reflection_round,
        round_key: reflection_round,
        **llm_trace_update,
    }

    # 兼容旧字段，避免历史逻辑/测试直接断裂。
    if stage == "post_answer":
        update["_self_check_pass"] = decision == "accept_answer"
    else:
        update["_grade"] = (
            "enough" if decision in {"accept_answer", "revise_answer"} else "need_more"
        )
    logger.debug(
        "reflection stage=%s decision=%s uncertainty=%.2f round=%d evidence=%d conflicts=%d",
        stage,
        decision,
        uncertainty,
        reflection_round,
        len(evidence),
        len(conflicts),
    )
    return update


def reflection_pre(state: dict) -> dict:
    """通用反思（pre_answer）：先评估证据状态。"""
    return _reflect(state, stage="pre_answer")


def reflection_post(state: dict) -> dict:
    """通用反思（post_answer）：评估回答草稿质量。"""
    return _reflect(state, stage="post_answer")


def response_compliance(state: dict) -> dict:
    """Review every answer path for locale, identity, scope, evidence and policy."""
    response = state.get("_final_response", {})
    if not isinstance(response, dict):
        response = {}
    policy = policy_from_state(state)
    locale = str(state.get("response_locale") or state.get("locale") or policy.language.default_locale)
    if _enforce_policy_response(state, response):
        return {
            "_final_response": response,
            "_compliance_decision": "accept",
            "compliance_replacement_count": int(state.get("compliance_replacement_count", 0) or 0),
            "_self_check_pass": True,
        }

    evidence = state.get("evidence", [])
    evidence = evidence if isinstance(evidence, list) else []
    citations = response.get("citations", [])
    citations = citations if isinstance(citations, list) else []
    user_prompt = prompts.RESPONSE_COMPLIANCE_USER.format(
        locale=locale,
        product_policy=policy.render_prompt_block(),
        turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
        conversation_context=_conversation_context_from_state(state),
        draft_answer=str(response.get("text", "") or "")[:5000] or "(empty)",
        evidence_summary=_summarize_evidence(evidence),
        citations_summary=_summarize_citations(citations),
    )
    raw_result: dict[str, Any] = {}
    llm_trace_update: dict[str, Any] = {}
    try:
        profile = _select_model_profile(
            state,
            "reflection",
            node_name="response_compliance",
            require_json=True,
            fallback_tier=policy.model_profiles.response_compliance,
        )
        raw_result = _call_llm_json_with_profile(
            prompts.RESPONSE_COMPLIANCE_SYSTEM,
            user_prompt,
            profile,
        )
        llm_trace_update = _llm_update_for_call(
            state,
            node_name="response_compliance",
            call_kind="business_json",
            profile=profile,
        )
    except Exception as exc:
        logger.warning("response_compliance failed type=%s", type(exc).__name__)

    decision = str(raw_result.get("decision", "accept") if isinstance(raw_result, dict) else "accept").strip()
    replacement = str(raw_result.get("replacement_answer", "") or "").strip() if isinstance(raw_result, dict) else ""
    current_replacements = int(state.get("compliance_replacement_count", 0) or 0)
    if decision == "replace" and replacement and current_replacements < policy.compliance.max_replacements:
        response["text"] = replacement
        current_replacements += 1
        response["citations"] = list(citations)
        update: dict[str, Any] = {
            "_final_response": response,
            "compliance_replacement_count": current_replacements,
            "_compliance_decision": "replace",
            "_compliance_issue_codes": list(raw_result.get("issue_codes", [])) if isinstance(raw_result, dict) else [],
            "_self_check_pass": True,
            **llm_trace_update,
        }
        if _has_inline_citation_tags(replacement):
            compiled_text, compiled_citations = _compile_inline_citations(replacement, evidence)
            response["text"] = compiled_text
            response["citations"] = compiled_citations
        return update

    return {
        "_final_response": response,
        "compliance_replacement_count": current_replacements,
        "_compliance_decision": "accept",
        "_compliance_issue_codes": list(raw_result.get("issue_codes", [])) if isinstance(raw_result, dict) else [],
        "_self_check_pass": bool(str(response.get("text", "") or "").strip()),
        **llm_trace_update,
    }


def doc_grader(state: dict) -> dict:
    """兼容壳层：内部委托给 pre_answer 通用反思。"""
    return reflection_pre(state)


# ---------------------------------------------------------------------------
# M7-T6: AnswerComposer — 回答组装
# ---------------------------------------------------------------------------

def answer_composer(state: dict) -> dict:
    """调 LLM 基于 evidence 组装带引用的回答。"""
    if state.get("_financial_guidance_refusal"):
        response = state.get("_final_response")
        if isinstance(response, dict):
            return {
                "_final_response": response,
                "_direct_answer": True,
                "_financial_guidance_refusal": True,
            }

    question = _effective_question(state)
    product_policy = policy_from_state(state)
    max_evidence_chunks = _budget_int(state, "max_evidence_chunks", 8)
    evidence = list(state.get("evidence", []))[:max_evidence_chunks]
    locale = str(state.get("response_locale") or state.get("locale") or product_policy.language.default_locale)
    request_id = state.get("request_id", "unknown")
    direct_mode = (
        str(state.get("retrieval_policy", "")).strip() == "none"
        or str(state.get("_route_decision", "")).strip() == "answer_direct"
    )
    image_paths = _image_paths_from_state(state)

    if not evidence:
        if direct_mode:
            user_prompt = prompts.DIRECT_ANSWER_USER.format(
                product_policy=product_policy.render_prompt_block(),
                question=question,
                locale=locale,
                turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
                conversation_context=_conversation_context_from_state(state),
                time_budget=_time_budget_prompt(state),
            )
            profile = _select_model_profile(
                state,
                "composing",
                node_name="direct_answer",
                require_json=False,
                fallback_tier="low",
            )
            compose_error: dict[str, str] | None = None
            try:
                answer_text = _call_llm_with_retry(
                    prompts.DIRECT_ANSWER_SYSTEM,
                    user_prompt,
                    **_profile_kwargs(profile),
                    max_attempts=2,
                    image_paths=image_paths,
                )
                llm_trace_update = _llm_update_for_call(
                    state,
                    node_name="direct_answer",
                    call_kind="business_text",
                    profile=profile,
                )
            except Exception as exc:
                answer_text = product_policy.message("direct_generation_failed", locale)
                llm_trace_update = {}
                compose_error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:280],
                }
                logger.warning(
                    "direct_answer failed request_id=%s type=%s msg=%s",
                    request_id,
                    compose_error["type"],
                    compose_error["message"],
                )

            response = {
                "request_id": request_id,
                "text": answer_text,
                "citations": [],
                "answer_mode": "direct",
            }
            refused_financial_guidance = _enforce_policy_response(state, response)
            if compose_error:
                response["trace_summary"] = (
                    f"direct_answer_error={compose_error['type']}: {compose_error['message']}"
                )
            update: dict[str, Any] = {
                "_final_response": response,
                "_direct_answer": True,
                **llm_trace_update,
            }
            if refused_financial_guidance:
                update["_financial_guidance_refusal"] = True
            if compose_error:
                update["_compose_error"] = compose_error
            else:
                logger.debug(
                    "direct_answer success request_id=%s chars=%d",
                    request_id,
                    len(answer_text),
                )
            return update

        return {
            "_final_response": {
                "request_id": request_id,
                "text": product_policy.message("insufficient_evidence", locale),
                "citations": [],
            },
            "_terminal_insufficient_evidence": True,
        }

    evidence_block = ""
    for i, ev in enumerate(evidence[:10], start=1):
        evidence_block += (
            f"\n--- Evidence E{i} ---\n"
            f"Evidence ID: E{i}\n"
            f"Source: {ev.get('source', '?')}\n"
            f"Title: {ev.get('title', '?')}\n"
            f"URL: {ev.get('url', '')}\n"
            f"Anchor: {ev.get('anchor', '')}\n"
            f"Content: {ev.get('snippet', '')}\n"
        )

    user_prompt = prompts.ANSWER_COMPOSER_USER.format(
        product_policy=product_policy.render_prompt_block(),
        question=question,
        locale=locale,
        turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
        conversation_context=_conversation_context_from_state(state),
        evidence_block=evidence_block or "(no usable evidence)",
        time_budget=_time_budget_prompt(state),
    )
    reflection_hints = state.get("reflection_hints", {})
    if isinstance(reflection_hints, dict):
        revise_instructions = str(reflection_hints.get("revise_instructions", "")).strip()
        if revise_instructions:
            user_prompt += f"\n\nReview rewrite instructions:\n{revise_instructions}"

    profile = _select_model_profile(
        state,
        "composing",
        node_name="answer_composer",
        require_json=False,
        fallback_tier="medium",
    )
    compose_error: dict[str, str] | None = None
    try:
        answer_text = _call_llm_with_retry(
            prompts.ANSWER_COMPOSER_SYSTEM,
            user_prompt,
            **_profile_kwargs(profile),
            max_attempts=2,
            image_paths=image_paths,
        )
        llm_trace_update = _llm_update_for_call(
            state,
            node_name="answer_composer",
            call_kind="business_text",
            profile=profile,
        )
    except Exception as exc:
        answer_text = product_policy.message("generation_failed", locale)
        llm_trace_update = {}
        compose_error = {
            "type": exc.__class__.__name__,
            "message": str(exc)[:280],
        }
        logger.warning(
            "answer_composer failed request_id=%s type=%s msg=%s",
            request_id,
            compose_error["type"],
            compose_error["message"],
        )

    response = {
        "request_id": request_id,
        "text": answer_text,
        "citations": [],
    }
    refused_financial_guidance = _enforce_policy_response(state, response)
    if compose_error:
        response["trace_summary"] = (
            f"answer_composer_error={compose_error['type']}: {compose_error['message']}"
        )

    update: dict[str, Any] = {"_final_response": response, **llm_trace_update}
    if refused_financial_guidance:
        update["_financial_guidance_refusal"] = True
    if compose_error:
        update["_compose_error"] = compose_error
    else:
        logger.debug(
            "answer_composer success request_id=%s chars=%d",
            request_id,
            len(answer_text),
        )
    return update


# ---------------------------------------------------------------------------
# M7-T7: SelfCheck — 自检
# ---------------------------------------------------------------------------

def self_check(state: dict) -> dict:
    """兼容壳层：内部委托给 post_answer 通用反思。"""
    return reflection_post(state)


# ---------------------------------------------------------------------------
# M7-T8: FormatRepair — 格式修复
# ---------------------------------------------------------------------------

def format_repair(state: dict) -> dict:
    """调用 normalizer 的 4 个函数做流水线清洗。"""
    response = state.get("_final_response", {})

    is_valid, errors = validate_response_shape(response)
    if not is_valid:
        response.setdefault("request_id", state.get("request_id", "unknown"))
        response.setdefault("text", "")
        response.setdefault("citations", [])

    text = response.get("text", "")
    citations = response.get("citations", [])
    if _enforce_policy_response(state, response):
        text = response.get("text", "")
        citations = response.get("citations", [])

    text = sanitize_markdown(text)
    if _has_inline_citation_tags(text):
        evidence = state.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        text, citations = _compile_inline_citations(text, evidence)
    else:
        text, citations = normalize_citations(text, citations)
    text = _append_reference_section(
        text,
        citations,
        locale=str(state.get("response_locale") or state.get("locale") or policy_from_state(state).language.default_locale),
        policy=policy_from_state(state),
    )

    response["text"] = text
    response["citations"] = citations
    tool_summary = str(state.get("_tool_execution_summary", "") or "").strip()
    if tool_summary:
        response["trace_summary"] = _merge_trace_summary(
            str(response.get("trace_summary", "") or ""),
            tool_summary,
        )
    user_msg = state.get("user_message", {})
    context = {}
    if isinstance(user_msg, dict) and isinstance(user_msg.get("context"), dict):
        context = dict(user_msg["context"])
    context.setdefault("platform", "discord")
    context.setdefault("user_id", "unknown_user")

    render_mode = str(state.get("render_mode", "markdown"))
    if render_mode not in ("markdown", "plain"):
        render_mode = "markdown"
    reply_to = (
        str(user_msg.get("reply_to_message_id"))
        if isinstance(user_msg, dict) and user_msg.get("reply_to_message_id")
        else None
    )
    if reply_to is None and isinstance(user_msg, dict):
        context_platform = ""
        if isinstance(user_msg.get("context"), dict):
            context_platform = str(user_msg["context"].get("platform", "") or "")
        if context_platform == "telegram" and user_msg.get("message_id"):
            reply_to = str(user_msg.get("message_id"))
    append_csat = bool(state.get("append_csat", False))

    try:
        outbound = format_response_to_outbound(
            response=response,
            context=context,  # type: ignore[arg-type]
            render_mode=render_mode,  # type: ignore[arg-type]
            append_csat=append_csat,
            reply_to_message_id=reply_to,
        )
        response["_chunks"] = [seg.get("text", "") for seg in outbound.get("segments", [])]
        return {"_final_response": response, "_outbound_message": outbound}
    except Exception:
        # Fallback to old behavior to keep runtime resilient.
        chunks = chunk_for_platform(text, max_chars=2000)
        response["_chunks"] = chunks
        return {"_final_response": response}


# ---------------------------------------------------------------------------
# AskUser — 反问用户 (复用 M3 逻辑，升级为使用 info_needs)
# ---------------------------------------------------------------------------

def ask_user(state: dict) -> dict:
    """When the contract requires private user input, generate one question."""
    info_needs = _normalize_info_needs_schema(state.get("info_needs", []))
    request_id = state.get("request_id", "unknown")
    checkpoint_id: str | None = None

    question = ""
    reflection_hints = state.get("reflection_hints", {})
    if isinstance(reflection_hints, dict):
        hinted_question = str(reflection_hints.get("clarify_question", "")).strip()
        if hinted_question:
            question = hinted_question

    if not (isinstance(reflection_hints, dict) and str(reflection_hints.get("clarify_question", "")).strip()):
        for need in info_needs:
            if need.get("required", False) and str(need.get("availability", "public")) == "user_owned":
                question = need.get("question", question)
                break

    has_required_question = bool(_required_questions(info_needs))
    has_concrete_reflection_question = (
        isinstance(reflection_hints, dict)
        and _reflection_has_user_required_clarification(info_needs, reflection_hints)
    )
    if not has_required_question and not has_concrete_reflection_question:
        guard_reason = "ask_user_without_required_info"
    else:
        guard_reason = ""
    if not guard_reason:
        question = _normalize_ask_user_question(question)
    else:
        product_policy = policy_from_state(state)
        locale = str(state.get("response_locale") or state.get("locale") or product_policy.language.default_locale)
        user_prompt = prompts.DIRECT_ANSWER_USER.format(
            product_policy=product_policy.render_prompt_block(),
            question=_effective_question(state),
            locale=locale,
            turn_contract=json.dumps(state.get("turn_contract", {}), ensure_ascii=False, default=str),
            conversation_context=_conversation_context_from_state(state),
            time_budget=_time_budget_prompt(state),
        )
        profile = _select_model_profile(
            state,
            "composing",
            node_name="direct_answer",
            require_json=False,
            fallback_tier="low",
        )
        question = _call_llm_with_retry(
            prompts.DIRECT_ANSWER_SYSTEM,
            user_prompt,
            **_profile_kwargs(profile),
            max_attempts=2,
        ).strip()

    # 写入线程 checkpoint，支持用户补参后恢复。
    svc = state.get("_memory_service")
    context = _get_message_context(state)
    thread_key = _thread_key_from_context(context)
    origin_question = _effective_question(state)
    if svc is not None and thread_key is not None and not guard_reason and question:
        try:
            checkpoint_id = svc.suspend_thread(
                key=thread_key,
                missing_params=_collect_missing_params(info_needs),
                resume_node="retriever_planner",
                context_payload={
                    "origin_question": origin_question,
                    "ask_user_question": question,
                },
            )
        except Exception:
            checkpoint_id = None

    response = {
        "request_id": request_id,
        "text": question,
        "citations": [],
        "need_user_input": True,
        "ask_user_question": question,
    }
    if checkpoint_id:
        response["trace_summary"] = f"thread_checkpoint={checkpoint_id}"
    if guard_reason:
        response["need_user_input"] = False
        response["ask_user_guard_reason"] = guard_reason
        response.pop("ask_user_question", None)

    logger.info(
        "ask_user request_id=%s question=%s checkpoint=%s",
        request_id,
        question,
        checkpoint_id or "-",
    )
    update = {"_final_response": response}
    if guard_reason:
        update["_ask_user_guard_reason"] = guard_reason
    return update
