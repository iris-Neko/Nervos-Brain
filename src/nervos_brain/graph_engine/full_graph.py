"""M7-T10/T11/T12: 完整 LangGraph StateGraph 构建。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from nervos_brain.core_protocols import GraphState

from .full_nodes import (
    answer_composer,
    ask_user,
    doc_grader,
    evidence_merger,
    format_repair,
    info_gap_assessor,
    retrieval_executor,
    retriever_planner,
    self_check,
)


def _budget_int(state: dict[str, Any], key: str, default: int) -> int:
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        return default
    value = budget.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_like(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_required_info_need(state: dict[str, Any]) -> bool:
    info_needs = state.get("info_needs", [])
    if not isinstance(info_needs, list):
        return False
    return any(
        isinstance(need, dict)
        and bool(need.get("required", False))
        for need in info_needs
    )


def _has_user_clarification_hint(state: dict[str, Any]) -> bool:
    hints = state.get("reflection_hints", {})
    if not isinstance(hints, dict):
        return False
    clarify_question = str(hints.get("clarify_question", "") or "").strip()
    if not clarify_question:
        return False
    text = clarify_question.lower()
    user_owned_markers = (
        "你的",
        "您",
        "你使用",
        "报错",
        "日志",
        "代码",
        "配置",
        "运行环境",
        "系统",
        "版本号",
        "目标语言",
        "sdk 语言",
        "钱包",
        "地址",
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
    return False


def _has_evidence(state: dict[str, Any]) -> bool:
    evidence = state.get("evidence", [])
    return isinstance(evidence, list) and bool(evidence)


class FullGraphState(GraphState, total=False):
    """协议 GraphState 加上 LangGraph 节点内部写入的键（本类声明的键均为可选）。"""

    _route_decision: str
    _final_response: dict
    _grade: str
    _self_check_pass: bool
    _outbound_message: dict
    _tool_calls_executed: int
    _tool_execution_trace: list[dict]
    _tool_execution_summary: str
    _llm_trace: list[dict]
    _llm_service_tier: str
    _llm_usage_summary: dict
    _node_timings: list[dict]
    _graph_elapsed_ms: int
    _request_started_ts_ms: int
    _reflection_rounds_pre: int
    _reflection_rounds_post: int
    retrieval_policy: str
    recent_messages: list[dict]
    conversation_context: str

    # Runtime deps injected by invoke_full_graph/attach_runtime_to_state.
    _multi_retriever: Any
    _memory_service: Any
    _provider_registry: Any
    _provider_max_cost: str
    _archive_store: Any
    _tool_transport: Any
    _mcp_transport: Any

    # Optional runtime controls from gateway.
    render_mode: str
    append_csat: bool
    force_retrieval: bool
    _compose_error: dict
    _ask_user_guard_reason: str
    _terminal_insufficient_evidence: bool
    _direct_answer: bool


@dataclass
class FullGraphRuntime:
    """完整图执行时可注入的运行时依赖。"""

    multi_retriever: Any | None = None
    memory_service: Any | None = None
    provider_registry: Any | None = None
    provider_max_cost: str = "high"


def attach_runtime_to_state(
    state: dict[str, Any],
    runtime: FullGraphRuntime | None,
) -> dict[str, Any]:
    """把运行时依赖注入 state，供节点直接读取。"""
    merged = dict(state)
    if runtime is None:
        return merged
    if runtime.multi_retriever is not None:
        merged["_multi_retriever"] = runtime.multi_retriever
    if runtime.memory_service is not None:
        merged["_memory_service"] = runtime.memory_service
    if runtime.provider_registry is not None:
        merged["_provider_registry"] = runtime.provider_registry
    merged["_provider_max_cost"] = runtime.provider_max_cost
    return merged


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------

def route_after_assessment(state: FullGraphState) -> str:
    """InfoGapAssessor 之后的路由。"""
    decision = state.get("_route_decision", "has_needs")
    if decision == "ask_user":
        return "ask_user" if _has_required_info_need(state) else "answer_composer"
    elif decision == "answer_direct":
        return "answer_composer"
    else:
        return "retriever_planner"


def route_after_grading(state: FullGraphState) -> str:
    """Pre-answer 反思后的路由（兼容旧 _grade 字段）。"""
    decision = str(state.get("reflection_decision", "")).strip()
    if not decision:
        grade = str(state.get("_grade", "need_more"))
        decision = "accept_answer" if grade == "enough" else "continue_retrieval"

    hop_count = max(_int_like(state.get("hop_count", 0)), _int_like(state.get("retry_count", 0)))
    pre_rounds = _int_like(state.get("_reflection_rounds_pre", 0))
    max_hops = _budget_int(state, "max_hops", 3)
    max_pre_rounds = _budget_int(state, "max_reflection_rounds_pre", 2)

    if decision == "ask_user":
        if _has_required_info_need(state) or _has_user_clarification_hint(state):
            return "ask_user"
        if _has_evidence(state):
            return "answer_composer"
        if hop_count < max_hops and pre_rounds < max_pre_rounds:
            return "retriever_planner"
        return "answer_composer"
    if decision in {"accept_answer", "revise_answer"}:
        return "answer_composer"

    if hop_count >= max_hops or pre_rounds >= max_pre_rounds:
        if _has_required_info_need(state):
            return "ask_user"
        if _has_evidence(state):
            return "answer_composer"
        return "answer_composer"
    return "retriever_planner"


def route_after_answer_composer(state: FullGraphState) -> str:
    """AnswerComposer 之后的路由。

    直接回答和证据不足终止文案不需要再进入重反思循环。
    """
    if state.get("_direct_answer") or state.get("_terminal_insufficient_evidence"):
        return "format_repair"
    return "self_check"


def route_after_self_check(state: FullGraphState) -> str:
    """Post-answer 反思后的路由（兼容旧 _self_check_pass 字段）。"""
    if state.get("_terminal_insufficient_evidence"):
        return "format_repair"

    decision = str(state.get("reflection_decision", "")).strip()
    if not decision:
        passed = bool(state.get("_self_check_pass", True))
        decision = "accept_answer" if passed else "revise_answer"

    post_rounds = _int_like(state.get("_reflection_rounds_post", 0))
    max_post_rounds = _budget_int(state, "max_reflection_rounds_post", 2)
    if post_rounds >= max_post_rounds and decision in {"revise_answer", "continue_retrieval"}:
        return "format_repair"

    if decision == "accept_answer":
        return "format_repair"
    if decision == "ask_user":
        if _has_required_info_need(state) or _has_user_clarification_hint(state):
            return "ask_user"
        if _has_evidence(state):
            return "answer_composer"
        return "retriever_planner"
    if decision == "continue_retrieval":
        return "retriever_planner"
    if decision == "revise_answer":
        return "answer_composer"
    return "format_repair"


# ---------------------------------------------------------------------------
# 带重试计数的 planner 包装
# ---------------------------------------------------------------------------

def retriever_planner_with_retry(state: FullGraphState) -> dict:
    """调 retriever_planner 并递增 retry_count。"""
    result = retriever_planner(state)
    result["retry_count"] = state.get("retry_count", 0) + 1
    return result


def _timed_node(name: str, fn: Any) -> Any:
    def _wrapped(state: FullGraphState) -> dict:
        start = time.perf_counter()
        result = fn(state)
        if not isinstance(result, dict):
            result = {}
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        existing = state.get("_node_timings", [])
        timings = list(existing) if isinstance(existing, list) else []
        timings.append({"node": name, "elapsed_ms": elapsed_ms})
        result["_node_timings"] = timings[-80:]
        return result

    _wrapped.__name__ = f"timed_{name}"
    return _wrapped


# ---------------------------------------------------------------------------
# 构建完整图
# ---------------------------------------------------------------------------

def build_full_graph() -> Any:
    """构建并编译完整 LangGraph 工作流，返回编译后的 graph。"""
    graph = StateGraph(FullGraphState)

    graph.add_node("info_gap_assessor", _timed_node("info_gap_assessor", info_gap_assessor))
    graph.add_node("ask_user", _timed_node("ask_user", ask_user))
    graph.add_node("retriever_planner", _timed_node("retriever_planner", retriever_planner_with_retry))
    graph.add_node("retrieval_executor", _timed_node("retrieval_executor", retrieval_executor))
    graph.add_node("evidence_merger", _timed_node("evidence_merger", evidence_merger))
    graph.add_node("doc_grader", _timed_node("doc_grader", doc_grader))
    graph.add_node("answer_composer", _timed_node("answer_composer", answer_composer))
    graph.add_node("self_check", _timed_node("self_check", self_check))
    graph.add_node("format_repair", _timed_node("format_repair", format_repair))

    graph.set_entry_point("info_gap_assessor")

    # InfoGapAssessor -> ask_user | retriever_planner | answer_composer
    graph.add_conditional_edges(
        "info_gap_assessor",
        route_after_assessment,
        {
            "ask_user": "ask_user",
            "retriever_planner": "retriever_planner",
            "answer_composer": "answer_composer",
        },
    )

    graph.add_edge("ask_user", END)

    # RetrieverPlanner -> RetrievalExecutor -> EvidenceMerger -> DocGrader
    graph.add_edge("retriever_planner", "retrieval_executor")
    graph.add_edge("retrieval_executor", "evidence_merger")
    graph.add_edge("evidence_merger", "doc_grader")

    # DocGrader(兼容壳层, 实际为 pre-answer reflection)
    # -> retriever_planner | ask_user | answer_composer
    graph.add_conditional_edges(
        "doc_grader",
        route_after_grading,
        {
            "retriever_planner": "retriever_planner",
            "ask_user": "ask_user",
            "answer_composer": "answer_composer",
        },
    )

    # AnswerComposer -> SelfCheck | FormatRepair
    graph.add_conditional_edges(
        "answer_composer",
        route_after_answer_composer,
        {
            "self_check": "self_check",
            "format_repair": "format_repair",
        },
    )

    # SelfCheck(兼容壳层, 实际为 post-answer reflection)
    # -> format_repair | retriever_planner | ask_user | answer_composer
    graph.add_conditional_edges(
        "self_check",
        route_after_self_check,
        {
            "format_repair": "format_repair",
            "retriever_planner": "retriever_planner",
            "ask_user": "ask_user",
            "answer_composer": "answer_composer",
        },
    )

    graph.add_edge("format_repair", END)

    return graph.compile()


def invoke_full_graph(
    state: dict[str, Any],
    *,
    runtime: FullGraphRuntime | None = None,
    compiled_graph: Any | None = None,
) -> dict[str, Any]:
    """带 runtime 注入地执行完整图。"""
    graph = compiled_graph or build_full_graph()
    return graph.invoke(attach_runtime_to_state(state, runtime))
