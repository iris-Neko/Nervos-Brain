from __future__ import annotations

from unittest.mock import patch

from nervos_brain.graph_engine.full_graph import route_after_grading, route_after_self_check
from nervos_brain.graph_engine.full_nodes import ask_user, reflection_post, reflection_pre


def _base_state(**overrides):
    base = {
        "request_id": "r-ref-1",
        "user_message": {
            "content": "Fiber open_channel 怎么开？",
            "context": {"platform": "discord", "user_id": "u1"},
        },
        "memory_facts": [],
        "info_needs": [],
        "evidence": [],
        "conflicts": [],
        "budget": {
            "max_reflection_rounds_pre": 2,
            "max_reflection_rounds_post": 2,
            "ask_user_uncertainty_threshold": 0.45,
        },
    }
    base.update(overrides)
    return base


def test_reflection_pre_public_conflict_heuristic_continues_retrieval():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "v0.2",
                "score": 0.9,
                "payload": {"source": "rfcs", "version": "0.2"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[{"a_id": "ev1", "b_id": "ev2", "reason": "version_mismatch"}],
    )
    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=RuntimeError("boom")):
        out = reflection_pre(state)

    assert out["reflection_stage"] == "pre_answer"
    assert out["reflection_decision"] == "continue_retrieval"
    assert out["_grade"] == "need_more"
    assert out["uncertainty_score"] >= 0.45


def test_reflection_pre_public_conflict_does_not_ask_user_when_llm_is_inconsistent():
    state = _base_state(
        user_message={
            "content": "如何给我的 agent 搭建一个 Fiber 节点，让它掌握一个 CKB 钱包并通过 Fiber 转账？",
            "context": {"platform": "telegram", "user_id": "u1"},
        },
        info_needs=[
            {
                "kind": "latest_spec",
                "question": "需要确认 Fiber 当前官方节点启动方式、RPC/API 和钱包接口。",
                "required": False,
            }
        ],
        evidence=[
            {
                "id": "ev1",
                "source": "github",
                "title": "Fiber docs",
                "url": "https://example.com/fiber",
                "anchor": "docs",
                "snippet": "Fiber has node setup docs and RPC transfer examples.",
                "score": 0.9,
                "payload": {"source": "fiber-docs", "version": "0.7.1"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[{"a_id": "ev1", "b_id": "ev2", "reason": "version_mismatch"}],
        hop_count=1,
        budget={
            "max_reflection_rounds_pre": 3,
            "max_reflection_rounds_post": 2,
            "max_hops": 3,
            "ask_user_uncertainty_threshold": 0.45,
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "ask_user",
            "reasoning": "这些都是公开可检索信息，不应追问用户。",
            "uncertainty_score": 0.9,
            "clarify_question": "请补充具体版本、运行环境或目标。",
        },
    ):
        out = reflection_pre(state)

    assert out["reflection_decision"] == "continue_retrieval"
    assert out["_grade"] == "need_more"
    assert "clarify_question" not in out["reflection_hints"]
    assert out["reflection_hints"].get("next_query")


def test_reflection_pre_budget_exhausted_with_evidence_accepts_answer_instead_of_asking():
    state = _base_state(
        user_message={
            "content": "我是萌新，你自己决定",
            "context": {"platform": "telegram", "user_id": "u1"},
        },
        info_needs=[
            {
                "kind": "latest_spec",
                "question": "需要确认 Fiber 节点启动方式和 RPC/API。",
                "required": False,
            }
        ],
        evidence=[
            {
                "id": "ev1",
                "source": "github",
                "title": "Fiber transfer docs",
                "url": "https://example.com/fiber-transfer",
                "anchor": "docs",
                "snippet": "Basic transfer example for Fiber testnet.",
                "score": 0.9,
                "payload": {"source": "fiber-docs", "version": "0.7.1"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[{"a_id": "ev1", "b_id": "ev2", "reason": "version_mismatch"}],
        hop_count=3,
        budget={
            "max_reflection_rounds_pre": 3,
            "max_reflection_rounds_post": 2,
            "max_hops": 3,
            "ask_user_uncertainty_threshold": 0.45,
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "ask_user",
            "reasoning": "已有资料可支撑 testnet MVP，但细节不完整。",
            "uncertainty_score": 0.9,
            "clarify_question": "请补充具体版本、运行环境或目标。",
        },
    ):
        out = reflection_pre(state)

    assert out["reflection_decision"] == "accept_answer"
    assert out["_grade"] == "enough"
    assert "clarify_question" not in out["reflection_hints"]


def test_reflection_post_citation_mismatch_to_revise_answer():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "open_channel(...)",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        _final_response={
            "request_id": "r-ref-1",
            "text": "调用 open_channel 即可 [1]",
            "citations": [],
        },
    )
    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=RuntimeError("boom")):
        out = reflection_post(state)

    assert out["reflection_stage"] == "post_answer"
    assert out["reflection_decision"] == "revise_answer"
    assert out["_self_check_pass"] is False


def test_reflection_post_compat_old_self_check_format():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "open_channel(...)",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        _final_response={
            "request_id": "r-ref-1",
            "text": "调用 open_channel 即可 [1]",
            "citations": [{"label": "[1]", "url": "https://a.example", "anchor": "a", "title": "Doc A"}],
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={"pass": True, "issues": [], "reasoning": "ok"},
    ):
        out = reflection_post(state)

    assert out["reflection_decision"] == "accept_answer"
    assert out["_self_check_pass"] is True


def test_reflection_post_accepts_inline_citation_tags_in_fallback():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "open_channel(...)",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        _final_response={
            "request_id": "r-ref-1",
            "text": "调用 open_channel 即可 {{cite:E1}}",
            "citations": [],
        },
    )
    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=RuntimeError("boom")):
        out = reflection_post(state)

    assert out["reflection_decision"] == "accept_answer"
    assert out["_self_check_pass"] is True


def test_reflection_post_sends_compiled_inline_citations_to_self_check():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "open_channel(...)",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        _final_response={
            "request_id": "r-ref-1",
            "text": "调用 open_channel 即可 {{cite:E1}}",
            "citations": [],
        },
    )
    captured = {}

    def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
        _ = system_prompt
        captured["user_prompt"] = user_prompt
        return {"pass": True, "issues": [], "reasoning": "ok"}

    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
        out = reflection_post(state)

    assert out["reflection_decision"] == "accept_answer"
    assert "调用 open_channel 即可 [1]" in captured["user_prompt"]
    assert "[1]: Doc A (https://a.example)" in captured["user_prompt"]


def test_reflection_post_demotes_ask_user_without_required_info_to_revise():
    state = _base_state(
        info_needs=[
            {
                "kind": "latest_spec",
                "question": "CKB 官方入门文档和社区推荐资料",
                "required": False,
            }
        ],
        evidence=[
            {
                "id": "ev1",
                "source": "discourse",
                "title": "CKB learning resources",
                "url": "https://talk.nervos.org/t/example",
                "anchor": "topic",
                "snippet": "Community discussion about CKB resources.",
                "score": 0.9,
                "payload": {"source": "talk"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        _final_response={
            "request_id": "r-ref-1",
            "text": "我不需要你补充版本或环境。",
            "citations": [],
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "ask_user",
            "reasoning": "官方资料属于公开可检索缺口，应继续检索而不是让用户确认。",
            "uncertainty_score": 0.9,
            "clarify_question": "请补充具体版本、运行环境或目标。",
            "missing_params": ["version"],
        },
    ):
        out = reflection_post(state)

    assert out["reflection_decision"] == "revise_answer"
    assert "clarify_question" not in out["reflection_hints"]
    assert "missing_params" not in out["reflection_hints"]
    assert "revise_instructions" in out["reflection_hints"]


def test_reflection_pre_demote_ask_user_when_no_required_and_no_conflict():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "CKB is a layer-1 blockchain.",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[],
        hop_count=1,
        budget={
            "max_reflection_rounds_pre": 3,
            "max_reflection_rounds_post": 2,
            "max_hops": 3,
            "ask_user_uncertainty_threshold": 0.95,
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "ask_user",
            "reasoning": "uncertain",
            "uncertainty_score": 0.9,
            "clarify_question": "请补充更多信息",
        },
    ):
        out = reflection_pre(state)

    assert out["reflection_decision"] == "continue_retrieval"
    assert out["_grade"] == "need_more"
    assert out["reflection_hints"].get("next_query")
    assert "clarify_question" not in out["reflection_hints"]


def test_reflection_pre_demotes_ask_user_even_at_max_hops_without_required_or_conflict():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "CKB intro",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "CKB is a layer-1 blockchain.",
                "score": 0.9,
                "payload": {"source": "docs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[],
        hop_count=3,
        budget={
            "max_reflection_rounds_pre": 3,
            "max_reflection_rounds_post": 2,
            "max_hops": 3,
            "ask_user_uncertainty_threshold": 0.45,
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "ask_user",
            "reasoning": "uncertain",
            "uncertainty_score": 0.9,
            "clarify_question": "请补充更多信息",
        },
    ):
        out = reflection_pre(state)

    assert out["reflection_decision"] == "accept_answer"
    assert out["_grade"] == "enough"
    assert "clarify_question" not in out["reflection_hints"]


def test_reflection_pre_high_uncertainty_without_required_or_conflict_keeps_retrieval():
    state = _base_state(
        evidence=[
            {
                "id": "ev1",
                "source": "qdrant",
                "title": "Doc A",
                "url": "https://a.example",
                "anchor": "a",
                "snippet": "CKB is a layer-1 blockchain.",
                "score": 0.9,
                "payload": {"source": "rfcs"},
                "hash": "h1",
                "retrieved_ts_ms": 1,
            }
        ],
        conflicts=[],
        budget={
            "max_reflection_rounds_pre": 3,
            "max_reflection_rounds_post": 2,
            "max_hops": 3,
            "ask_user_uncertainty_threshold": 0.45,
        },
    )
    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "decision": "continue_retrieval",
            "reasoning": "uncertain",
            "uncertainty_score": 0.8,
        },
    ):
        out = reflection_pre(state)

    assert out["reflection_decision"] == "continue_retrieval"
    assert out["_grade"] == "need_more"


def test_route_after_grading_answers_when_max_hops_reached_without_user_missing_param():
    route = route_after_grading(
        {
            "reflection_decision": "continue_retrieval",
            "hop_count": 3,
            "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
        }
    )
    assert route == "answer_composer"


def test_route_after_grading_still_asks_when_required_param_missing():
    route = route_after_grading(
        {
            "reflection_decision": "continue_retrieval",
            "hop_count": 3,
            "info_needs": [
                {
                        "kind": "missing_param",
                        "question": "请问你使用的是哪个 SDK？",
                        "required": True,
                        "availability": "user_owned",
                }
            ],
            "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
        }
    )
    assert route == "ask_user"


def test_route_after_grading_invalid_ask_user_with_evidence_answers():
    route = route_after_grading(
        {
            "reflection_decision": "ask_user",
            "hop_count": 1,
            "evidence": [{"id": "ev-1", "snippet": "Fiber setup docs"}],
            "info_needs": [
                {
                    "kind": "latest_spec",
                    "question": "需要确认 Fiber 官方节点启动方式。",
                    "required": False,
                }
            ],
            "reflection_hints": {
                "clarify_question": "请补充具体版本、运行环境或目标。"
            },
            "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
        }
    )
    assert route == "answer_composer"


def test_route_after_grading_invalid_ask_user_without_evidence_retrieves():
    route = route_after_grading(
        {
            "reflection_decision": "ask_user",
            "hop_count": 0,
            "info_needs": [
                {
                    "kind": "latest_spec",
                    "question": "需要确认 Fiber 官方节点启动方式。",
                    "required": False,
                }
            ],
            "reflection_hints": {},
            "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
        }
    )
    assert route == "retriever_planner"


def test_route_after_self_check_decision_mapping():
    assert route_after_self_check({"reflection_decision": "revise_answer"}) == "answer_composer"
    assert route_after_self_check({"reflection_decision": "continue_retrieval"}) == "retriever_planner"
    assert route_after_self_check({"reflection_decision": "ask_user"}) == "retriever_planner"
    assert route_after_self_check(
        {
            "reflection_decision": "ask_user",
                "info_needs": [{"required": True, "availability": "user_owned", "question": "请贴完整报错日志"}],
        }
    ) == "ask_user"
    assert route_after_self_check({"reflection_decision": "accept_answer"}) == "format_repair"


def test_ask_user_normalizes_phrase_to_question():
    out = ask_user(
        {
            "request_id": "r-ask-1",
                "info_needs": [{"required": True, "availability": "user_owned", "question": "CKB框架的基本概念和介绍"}],
            "reflection_hints": {},
            "user_message": {"context": {"platform": "telegram", "user_id": "u-1"}},
        }
    )
    text = out["_final_response"]["text"]
    assert "CKB框架的基本概念和介绍" in text
    assert text == "CKB框架的基本概念和介绍"


def test_ask_user_does_not_wrap_generic_uncertainty_as_topic_question():
    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=RuntimeError("router down")), \
         patch("nervos_brain.graph_engine.full_nodes.call_llm", return_value="抱歉，我刚才可能答偏了。请把要继续回答的问题再发一次。"):
        out = ask_user(
            {
                "request_id": "r-ask-generic",
                "info_needs": [],
                "reflection_hints": {
                    "clarify_question": "当前信息存在不确定性，请补充具体版本、环境或目标。"
                },
                "user_message": {"content": "你是不是回复错问题了", "context": {"platform": "telegram", "user_id": "u-1"}},
            }
        )
    text = out["_final_response"]["text"]
    assert "你是想了解" not in text
    assert "当前信息存在不确定性" not in text
    assert "默认 testnet" not in text
    assert text.startswith("抱歉")
    assert out["_final_response"]["need_user_input"] is False


def test_ask_user_guard_avoids_generic_prompt_without_required_info():
    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=RuntimeError("router down")), \
         patch("nervos_brain.graph_engine.full_nodes.call_llm", return_value="我会按你当前问题继续推进，不再让你补充版本或环境。"):
        out = ask_user(
            {
                "request_id": "r-ask-guard",
                "info_needs": [
                    {
                        "kind": "latest_spec",
                        "question": "需要确认 Fiber 官方节点启动方式。",
                        "required": False,
                    }
                ],
                "reflection_hints": {
                    "clarify_question": "请补充具体版本、运行环境或目标。"
                },
                "user_message": {"content": "我是萌新，你自己决定", "context": {"platform": "telegram", "user_id": "u-1"}},
            }
        )
    response = out["_final_response"]
    assert response["need_user_input"] is False
    assert response["ask_user_guard_reason"] == "ask_user_without_required_info"
    assert "默认 testnet" not in response["text"]
    assert response["text"].startswith("我会按")
    assert out["_ask_user_guard_reason"] == "ask_user_without_required_info"
