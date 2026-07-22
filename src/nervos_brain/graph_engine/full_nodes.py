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
from .provider_registry import ProviderCapabilityRegistry
from .source_registry import (
    format_source_registry_for_prompt,
    normalize_tool_filters,
    should_retry_qdrant_without_filters,
)

_ANSWER_COMPOSER_FALLBACK_TEXT = "证据已收集，但回答生成暂时失败，请稍后重试。"
_DIRECT_ANSWER_FALLBACK_TEXT = "我可以直接回答低风险问题，但这次生成暂时失败了，请稍后重试。"
_FINANCIAL_GUIDANCE_REFUSAL_TEXT = (
    "I can't provide price predictions, target prices, or buy/sell guidance. "
    "I can help explain fundamentals, tokenomics, technical risks, or where to find neutral market data."
)
_RETRIEVAL_POLICIES = {"none", "single", "deep"}
_MARKET_PREDICTION_INTENT_RE = re.compile(
    r"(?i)("
    r"price\s*(?:prediction|forecast|target)|"
    r"(?:predict|forecast).{0,40}(?:price|market|value)|"
    r"(?:price|market|value).{0,40}(?:predict|forecast)|"
    r"target\s*price|best\s*price|"
    r"bull\s*cycle\s*target|bear\s*market\s*(?:bottom|target)|"
    r"(?:moon|moonshot)\s*(?:price|target)|"
    r"(?:price|target).{0,20}(?:moon|moonshot)|"
    r"\b[A-Z]{2,12}\b\s+(?:to|reach|hit)\s*(?:\$|usd|usdt)\s*\d|"
    r"(?:can|will|could)\s+\b[A-Z]{2,12}\b\s+(?:reach|hit)\s*(?:\$|usd|usdt)\s*\d|"
    r"价格预测|预测价格|目标价|指导价|走势预测|牛市目标|熊市底|涨到多少|跌到多少|"
    r"涨到\s*(?:\$|usd|usdt|美元)?\s*\d|跌到\s*(?:\$|usd|usdt|美元)?\s*\d|"
    r"会涨到|会跌到"
    r")"
)
_TRADING_DECISION_INTENT_RE = re.compile(
    r"(?i)("
    r"should\s+(?:i|we)\s+(?:buy|sell|hold)\b|"
    r"is\s+(?:now|today)\s+(?:a\s+)?good\s+time\s+to\s+(?:buy|sell)|"
    r"(?:buy|sell)\s+or\s+(?:buy|sell)|"
    r"(?:now|today).{0,30}\b(?:buy|sell|hold|entry|exit)\b|"
    r"\b(?:buy|sell|hold|entry|exit)\b.{0,30}(?:now|today)|"
    r"take\s*profit|stop\s*loss|"
    r"investment\s*advice|financial\s*advice|"
    r"该不该(?:买|卖|持有)|要不要(?:买|卖|持有)|现在.{0,12}(?:买入|卖出|持有|建仓|加仓|减仓)|"
    r"买入吗|卖出吗|持有吗|建仓吗|加仓吗|减仓吗|止盈|止损|投资建议|金融建议"
    r")"
)
_MARKET_TIMING_CONTEXT_RE = re.compile(
    r"(?i)(?:\bnow\b|\btoday\b|good\s+time|entry|exit|portfolio|investment|financial|现在|今天|入场|出场|仓位|投资|金融)"
)
_PRICE_RANGE_RE = re.compile(
    r"(?i)(?:\$|usd|usdt|美元)\s*\d+(?:\.\d+)?\s*(?:[-–—~至到]\s*(?:\$|usd|usdt|美元)?\s*\d+(?:\.\d+)?)?"
)
_TECHNICAL_CKB_CONTEXT_RE = re.compile(
    r"(?i)("
    r"\bcell\b|\bcapacity\b|\bckbyte\b|common\s+knowledge\s+(?:base|byte)|"
    r"ckb[-\s]?vm|lock\s+script|type\s+script|\btransaction\b|\bfee\b|\bcycles?\b|"
    r"dao\s+(?:deposit|withdraw)|\bxudt\b|\bspore\b|rgb\+\+|\bsdk\b|\bapi\b|\brpc\b|"
    r"\bnode\b|\bminer\b|"
    r"容量|存储|交易|手续费|脚本|节点|矿工|质押|存入|提取"
    r")"
)


def _normalize_info_needs_schema(info_needs: Any) -> list[dict]:
    """Normalize the LLM schema without changing its semantic decision."""
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
        sanitized.append(need)
    return sanitized


def _is_financial_guidance_request(text: str) -> bool:
    """Detect explicit price-prediction or trading-decision requests."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    if _MARKET_PREDICTION_INTENT_RE.search(normalized):
        return True
    if not _TRADING_DECISION_INTENT_RE.search(normalized):
        return False
    if _TECHNICAL_CKB_CONTEXT_RE.search(normalized) and not _MARKET_TIMING_CONTEXT_RE.search(normalized):
        return False
    if _TRADING_DECISION_INTENT_RE.search(normalized):
        return True
    return False


def _financial_guidance_refusal_text(locale: str | None = None) -> str:
    lang = str(locale or "").lower()
    if lang.startswith("zh"):
        return "我不能提供价格预测、目标价或买卖指导。可以帮你解释项目基本面、代币经济模型、技术风险，或告诉你去哪里查看中立市场数据。"
    return _FINANCIAL_GUIDANCE_REFUSAL_TEXT


def _force_financial_guidance_refusal(state: dict, *, question: str | None = None) -> dict[str, Any]:
    request_id = str(state.get("request_id", "unknown"))
    response = {
        "request_id": request_id,
        "text": _financial_guidance_refusal_text(str(state.get("locale", "zh-CN"))),
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


def _sanitize_financial_guidance_response(state: dict, response: dict[str, Any]) -> bool:
    """Replace any generated price/trading guidance with a refusal."""
    question = _effective_question(state)
    text = str(response.get("text", "") or "")
    if not _is_financial_guidance_request(question):
        return False
    if not (
        _PRICE_RANGE_RE.search(text)
        or _MARKET_PREDICTION_INTENT_RE.search(text)
        or _TRADING_DECISION_INTENT_RE.search(text)
    ):
        return False
    response["text"] = _financial_guidance_refusal_text(str(state.get("locale", "zh-CN")))
    response["citations"] = []
    response["answer_mode"] = "policy_refusal"
    return True


logger = logging.getLogger(__name__)

_MODEL_TIERS = {"low", "medium", "high"}
_NODE_FALLBACK_TIERS = {
    "info_gap_assessor": "low",
    "retriever_planner": "low",
    "reflection_pre": "low",
    "reflection_post": "medium",
    "direct_answer": "low",
    "answer_composer": "medium",
}
_LLM_ROUTER_SYSTEM = """Prompt ID: model_router
You are the Nervos Brain model-tier router.
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
- info_gap_assessor and retriever_planner default to low. Upgrade only when
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


def _budget_float(state: dict, key: str, default: float) -> float:
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        return default
    value = budget.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_decision(raw: Any) -> str:
    decision = str(raw or "").strip()
    if decision in {"ask_user", "has_needs", "answer_direct"}:
        return decision
    return "has_needs"


def _normalize_retrieval_policy(raw: Any, *, decision: str) -> str:
    if decision in {"answer_direct", "ask_user"}:
        return "none"
    policy = str(raw or "").strip()
    if policy == "none":
        return "single"
    if policy in _RETRIEVAL_POLICIES:
        return policy
    return "single"


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
    question = _effective_question(state)
    response = state.get("_final_response", {})
    draft = response.get("text", "") if isinstance(response, dict) else ""
    evidence = state.get("evidence", [])
    conflicts = state.get("conflicts", [])
    info_needs = state.get("info_needs", [])
    return {
        "has_code_request": bool(re.search(r"代码|code|typescript|javascript|python|rust|示例|example", question, re.I)),
        "has_error_or_logs": bool(re.search(r"报错|错误|error|exception|traceback|日志|log", question, re.I)),
        "has_architecture_request": bool(re.search(r"架构|方案|设计|完整|构建|开发", question, re.I)),
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
        "info_gap_assessor": "Decide whether to answer directly, retrieve, or ask the user, and choose a retrieval policy.",
        "retriever_planner": "Convert information needs into the smallest necessary retrieval plan.",
        "reflection_pre": "Decide whether current evidence is sufficient for the core question.",
        "reflection_post": "Check whether the draft is correct, citations match, and rewriting is needed.",
        "direct_answer": "Answer a low-risk question directly without citations.",
        "answer_composer": "Generate the final answer from evidence and context.",
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
    if isinstance(raw, list):
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


def _looks_like_self_contained_question(question: str) -> bool:
    """Best-effort context gate; semantic decisions still belong to the LLM."""
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if len(text) < 12:
        return False
    if not any(marker in text for marker in ("?", "？", "什么", "哪些", "哪个", "如何", "怎么", "为什么", "是否", "吗")):
        return False
    followup_markers = (
        "这个",
        "这个呢",
        "这些",
        "它",
        "他们",
        "上面",
        "刚才",
        "继续",
        "再说",
        "小白版",
        "展开",
        "那",
        "所以",
    )
    if any(text == marker or text.startswith(marker) for marker in followup_markers):
        return False
    context_dependent_patterns = (
        "有没有",
        "有链接",
        "哪里看",
        "怎么学",
        "靠谱吗",
        "资料",
        "来源",
        "引用",
    )
    if any(pattern in text for pattern in context_dependent_patterns) and not re.search(
        r"[A-Za-z0-9][A-Za-z0-9_.:/#-]{2,}",
        text,
    ):
        return False
    return True


def _conversation_context_from_state(state: dict) -> str:
    existing = str(state.get("conversation_context", "") or "").strip()
    question = _question_from_user_message(state)
    if existing and existing.startswith("Current message replies"):
        return existing
    if _looks_like_self_contained_question(question):
        return (
            "Context gate: the current user question appears self-contained. "
            "Ordinary recent history may resolve pronouns or omissions only; it "
            "must not change the retrieval target, answer topic, or evidence organization."
        )
    if existing:
        return existing
    return _format_conversation_context(_load_recent_messages(state))


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


def _merge_checkpoint_question(question: str, checkpoint: dict[str, object] | None) -> str:
    text = str(question or "").strip()
    if checkpoint is None:
        return text
    payload = checkpoint.get("context_payload", {})
    if not isinstance(payload, dict):
        return text
    origin = str(payload.get("origin_question", "") or "").strip()
    if not origin:
        return text
    if not text:
        return origin
    if origin in text or text in origin:
        return text if len(text) >= len(origin) else origin
    return f"{origin}\nUser supplement: {text}"


def _looks_like_new_user_intent(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.lower())
    if not normalized:
        return False
    markers = (
        "是什么",
        "什么是",
        "为什么",
        "怎么",
        "如何",
        "能不能",
        "可以",
        "给我",
        "写一个",
        "完整",
        "例子",
        "应用",
        "开发什么",
        "代码",
        "?",
        "？",
    )
    return any(marker in normalized for marker in markers)


def _looks_like_checkpoint_answer(text: str, checkpoint: dict[str, object]) -> bool:
    normalized = re.sub(r"\s+", "", text.lower())
    if not normalized:
        return False

    param_markers = (
        "ts",
        "typescript",
        "js",
        "javascript",
        "rust",
        "go",
        "python",
        "ckb-cli",
        "cli",
        "sdk",
        "ccc",
        "node",
        "browser",
        "testnet",
        "mainnet",
        "v0.",
        "0.",
        "版本",
        "环境",
        "调用脚本",
        "合约",
        "脚本",
    )
    if any(marker in normalized for marker in param_markers):
        return True

    payload = checkpoint.get("context_payload", {})
    ask_question = ""
    if isinstance(payload, dict):
        ask_question = str(payload.get("ask_user_question", "") or "").lower()
    missing = checkpoint.get("missing_params", [])
    missing_text = " ".join(str(item).lower() for item in missing) if isinstance(missing, list) else ""
    if any(marker in ask_question or marker in missing_text for marker in ("sdk", "语言", "版本", "environment", "env")):
        return len(normalized) <= 40 and not _looks_like_new_user_intent(text)

    return len(normalized) <= 24 and not _looks_like_new_user_intent(text)


def _should_resume_checkpoint(question: str, checkpoint: dict[str, object] | None) -> bool:
    if checkpoint is None:
        return False
    text = str(question or "").strip()
    if not text:
        return False
    if _looks_like_checkpoint_answer(text, checkpoint):
        return True
    if _looks_like_new_user_intent(text):
        return False
    return len(re.sub(r"\s+", "", text)) <= 24


def _resume_info_needs(info_needs: list[dict]) -> list[dict]:
    """Checkpoint 恢复后把用户本轮消息视为补参，避免继续卡在旧 required 缺参。"""
    kept: list[dict] = []
    for need in info_needs:
        if not isinstance(need, dict):
            continue
        if bool(need.get("required", False)):
            continue
        kept.append(need)
    if kept:
        return kept
    return [
        {
            "kind": "concept_gap",
            "question": "The user supplied the missing parameters; continue retrieval and answer composition.",
            "required": False,
        }
    ]


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
        text = text[:220].rstrip() + "…"

    question_markers = ("？", "?", "吗", "呢", "是否", "可否", "能否")
    if any(marker in text for marker in question_markers):
        return text

    return text


# ---------------------------------------------------------------------------
# M7-T1: InfoGapAssessor — 信息缺口评估
# ---------------------------------------------------------------------------

def info_gap_assessor(state: dict) -> dict:
    """调 LLM 分析问题，判断 ask_user / has_needs / answer_direct。"""
    raw_question = _question_from_user_message(state)
    facts = _load_memory_facts(state)
    evidence = state.get("evidence", [])
    checkpoint = _resume_thread_checkpoint(state)
    should_resume_checkpoint = _should_resume_checkpoint(raw_question, checkpoint)
    if checkpoint is not None and not should_resume_checkpoint:
        _complete_thread_checkpoint(state)
        checkpoint = None
    question = _merge_checkpoint_question(raw_question, checkpoint)
    recent_messages = _load_recent_messages(state)
    conversation_context = _conversation_context_from_state(state)

    if _is_financial_guidance_request(question):
        update = _force_financial_guidance_refusal(state, question=question)
        update["memory_facts"] = facts
        update["recent_messages"] = recent_messages
        update["conversation_context"] = conversation_context
        logger.info("info_gap_assessor blocked financial guidance request")
        return update

    user_prompt = prompts.INFO_GAP_USER.format(
        question=question,
        locale=str(state.get("locale", "zh-CN")),
        conversation_context=conversation_context,
        memory_facts=json.dumps(facts, ensure_ascii=False, default=str)[:500],
        evidence_count=len(evidence),
        time_budget=_time_budget_prompt(state),
    )

    if checkpoint:
        thread_state: dict[str, Any] = {
            "missing_params": checkpoint.get("missing_params", []),
            "resume_node": checkpoint.get("resume_node", ""),
        }
        context_payload = checkpoint.get("context_payload", {})
        if isinstance(context_payload, dict) and context_payload:
            thread_state["context_payload"] = context_payload
        user_prompt += (
            "\nThread resume state: "
            + json.dumps(thread_state, ensure_ascii=False)
        )

    profile = _select_model_profile(
        state,
        "planning",
        node_name="info_gap_assessor",
        require_json=True,
    )
    try:
        result = _call_llm_json_with_profile(
            prompts.INFO_GAP_SYSTEM,
            user_prompt,
            profile,
        )
        llm_trace_update = _llm_update_for_call(
            state,
            node_name="info_gap_assessor",
            call_kind="business_json",
            profile=profile,
        )
    except Exception:
        result = {
            "decision": "has_needs" if question.strip() else "ask_user",
            "info_needs": [],
        }
        llm_trace_update = {}

    decision = _normalized_decision(result.get("decision", "has_needs"))
    info_needs = _normalize_info_needs_schema(result.get("info_needs", []))
    if decision == "ask_user" and not _required_questions(info_needs):
        decision = "has_needs" if info_needs else "answer_direct"
    retrieval_policy = _normalize_retrieval_policy(
        result.get("retrieval_policy"),
        decision=decision,
    )
    existing_conflicts = state.get("conflicts", [])
    if (
        decision == "has_needs"
        and isinstance(existing_conflicts, list)
        and existing_conflicts
    ):
        retrieval_policy = "deep"

    # 显式兼容旧调用：只有调用方主动传 force_retrieval=True 时才强制检索。
    force_retrieval = bool(state.get("force_retrieval", False))
    if force_retrieval and decision == "answer_direct" and question.strip():
        decision = "has_needs"
        retrieval_policy = "single"
        if not info_needs:
            info_needs = [
                {
                    "kind": "concept_gap",
                    "question": "Retrieve the relevant public material before answering.",
                    "required": False,
                }
            ]

    # 如果线程里有待恢复 checkpoint，默认把本轮消息当作补参继续推进。
    if checkpoint is not None and should_resume_checkpoint and question.strip():
        decision = "has_needs"
        retrieval_policy = "single" if retrieval_policy == "none" else retrieval_policy
        _complete_thread_checkpoint(state)
        info_needs = _resume_info_needs(info_needs)
        if info_needs:
            info_needs[0].setdefault("hints", {})
            if isinstance(info_needs[0]["hints"], dict):
                info_needs[0]["hints"].setdefault(
                    "resume_node",
                    str(checkpoint.get("resume_node", "retriever_planner")),
                )

    update: dict[str, Any] = {
        "_route_decision": decision,
        "retrieval_policy": retrieval_policy,
        "info_needs": info_needs,
        "memory_facts": facts,
        "recent_messages": recent_messages,
        "conversation_context": conversation_context,
        "resolved_question": question,
        "budget": _merge_policy_budget(state, retrieval_policy),
        **llm_trace_update,
    }
    logger.debug(
        "info_gap_assessor decision=%s retrieval_policy=%s info_needs=%d checkpoint=%s",
        decision,
        retrieval_policy,
        len(info_needs),
        bool(checkpoint),
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


def _has_reference_section(text: str) -> bool:
    return bool(
        re.search(
            r"(?im)^\s{0,3}(?:#{1,6}\s*)?(参考来源|引用来源|References|Sources)\b",
            text,
        )
    )


def _append_reference_section(text: str, citations: list[dict[str, Any]], *, locale: str = "zh-CN") -> str:
    if not citations or _has_reference_section(text):
        return text.strip()

    is_english = str(locale or "").strip().lower().replace("_", "-").startswith("en")
    heading = "References" if is_english else "参考来源"
    fallback_title = "Source" if is_english else "来源"

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
        question = str(need.get("question", "")).strip()
        if question:
            questions.append(question)
    return questions


def _reflection_has_user_required_clarification(
    info_needs: list[dict],
    hints: dict[str, Any],
) -> bool:
    """Return true only when the ask_user action is backed by user-owned missing data."""
    if _required_questions(info_needs):
        return True
    clarify_question = str(hints.get("clarify_question", "") or "").strip()
    if not clarify_question:
        return False
    text = clarify_question.lower()
    user_owned_markers = (
        "你的",
        "您",
        "你使用",
        "你用",
        "请贴",
        "请提供你",
        "请补充你",
        "报错",
        "日志",
        "代码",
        "配置",
        "目标语言",
        "sdk 语言",
        "私有",
        "业务",
    )
    if not any(marker in text for marker in user_owned_markers):
        return False
    missing_params = hints.get("missing_params", [])
    if isinstance(missing_params, list):
        return any(str(item).strip() for item in missing_params)
    if isinstance(missing_params, str):
        return bool(missing_params.strip())
    return True


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

    req_questions = _required_questions(info_needs if isinstance(info_needs, list) else [])
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


def _localized_conflict_question(state: dict) -> str:
    locale = str(state.get("locale", "zh-CN") or "zh-CN").strip().lower()
    if locale.startswith("en"):
        return "Sources disagree. What specific version or scenario are you using?"
    return "不同来源结论不一致，能否补充你使用的具体版本或场景？"


def _localized_uncertainty_question(state: dict) -> str:
    locale = str(state.get("locale", "zh-CN") or "zh-CN").strip().lower()
    if locale.startswith("en"):
        return "The current information is uncertain. Please provide the specific version, environment, or goal."
    return "当前信息存在不确定性，请补充具体版本、环境或目标。"


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

    if text.strip() == _ANSWER_COMPOSER_FALLBACK_TEXT:
        return {
            "decision": "revise_answer",
            "reasoning": "The answer-generation fallback text was detected; retry answer generation.",
            "uncertainty_score": 0.90,
            "revise_instructions": "Retry answer generation; if it fails again, give a brief failure explanation.",
        }

    if conflicts:
        return {
            "decision": "ask_user",
            "reasoning": "The evidence conflict is unresolved, so the answer may not be reliable.",
            "uncertainty_score": 0.82,
            "clarify_question": _localized_conflict_question(state),
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
        locale=str(state.get("locale", "zh-CN")),
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
    req_questions = _required_questions(info_needs)
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
        isinstance(compose_error, dict) or draft_answer.strip() == _ANSWER_COMPOSER_FALLBACK_TEXT
    ):
        decision = "revise_answer"
        uncertainty = max(uncertainty, 0.90)
        if not reasoning:
            reasoning = "Answer generation failed and must be retried."
        hints.setdefault("revise_instructions", "Retry answer generation from the existing evidence.")

    # 主动追问策略：高不确定度时，优先让用户补充上下文。
    ask_threshold = _budget_float(state, "ask_user_uncertainty_threshold", 0.45)
    if decision == "continue_retrieval" and uncertainty >= ask_threshold:
        should_ask_user = True
        if stage == "pre_answer":
            # pre-answer 不因“高不确定度”单独触发追问；
            # 只有用户私有/现场必填信息才追问；公开资料冲突继续检索或回答。
            should_ask_user = has_user_required_clarification

        if should_ask_user:
            decision = "ask_user"
            if not hints.get("clarify_question"):
                hints["clarify_question"] = (
                    req_questions[0]
                    if req_questions
                    else _localized_uncertainty_question(state)
                )
        elif stage == "pre_answer" and evidence and pre_budget_exhausted:
            decision = "accept_answer"
            hints.pop("clarify_question", None)
            hints.pop("missing_params", None)

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
    max_evidence_chunks = _budget_int(state, "max_evidence_chunks", 8)
    evidence = list(state.get("evidence", []))[:max_evidence_chunks]
    locale = state.get("locale", "zh-CN")
    request_id = state.get("request_id", "unknown")
    direct_mode = (
        str(state.get("retrieval_policy", "")).strip() == "none"
        or str(state.get("_route_decision", "")).strip() == "answer_direct"
    )
    image_paths = _image_paths_from_state(state)

    if not evidence:
        if direct_mode:
            user_prompt = prompts.DIRECT_ANSWER_USER.format(
                question=question,
                locale=locale,
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
                answer_text = _DIRECT_ANSWER_FALLBACK_TEXT
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
            refused_financial_guidance = _sanitize_financial_guidance_response(state, response)
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
                "text": "当前知识库没有检索到足够证据，暂时不能给出可靠结论。请换一个更具体的关键词，或补充你想查的协议、工具、版本或报错信息。",
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
        question=question,
        locale=locale,
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
        answer_text = _ANSWER_COMPOSER_FALLBACK_TEXT
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
    refused_financial_guidance = _sanitize_financial_guidance_response(state, response)
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
    if _sanitize_financial_guidance_response(state, response):
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
    text = _append_reference_section(text, citations, locale=str(state.get("locale", "zh-CN")))

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
    """当 InfoGapAssessor 判断需要反问时，生成反问回答。"""
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
            if need.get("required", False):
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
        locale = state.get("locale", "zh-CN")
        user_prompt = prompts.DIRECT_ANSWER_USER.format(
            question=_effective_question(state),
            locale=locale,
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
