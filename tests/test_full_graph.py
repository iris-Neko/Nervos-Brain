"""M7-T13/T14/T15: 三大场景测试 (mock LLM, 不依赖 API key)。"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest


def _make_state(**overrides):
    """构造最小 FullGraphState。"""
    base = {
        "request_id": "test-req-001",
        "user_message": {"content": "test question"},
        "user_memory_key": {"platform": "discord", "user_id": "u1"},
        "memory_pointers": [],
        "memory_facts": [],
        "info_needs": [],
        "evidence": [],
        "conflicts": [],
        "retry_count": 0,
        "budget": {
            "max_prompt_tokens": 4000,
            "max_evidence_chunks": 10,
            "max_memory_facts": 5,
            "max_tool_calls": 3,
        },
        "route": "graph",
        "locale": "zh-CN",
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------
# Mock call_llm helpers
# -----------------------------------------------------------------------

_LLM_CALL_LOG: list[tuple[str, str]] = []


def _mock_call_llm_factory(responses: dict[str, str]):
    """创建一个根据 system prompt 关键词返回预设响应的 mock。"""

    def _mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
        _LLM_CALL_LOG.append((system_prompt[:50], user_prompt[:50]))
        for keyword, response in responses.items():
            if keyword in system_prompt:
                return response
        return '{"decision": "has_needs", "info_needs": []}'

    return _mock_call_llm


def _mock_call_llm_json_factory(responses: dict[str, dict]):
    """创建根据 system prompt 关键词返回预设 JSON 的 mock。"""

    def _mock_call_llm_json(system_prompt, user_prompt, *, model=None):
        for keyword, response in responses.items():
            if keyword in system_prompt:
                return response
        return {"decision": "has_needs", "info_needs": []}

    return _mock_call_llm_json


# -----------------------------------------------------------------------
# M7-T13: "Invalid capacity" 场景 — 缺 SDK 语言，走 AskUser
# -----------------------------------------------------------------------

class TestInvalidCapacityScenario:
    """构造缺 SDK 语言的输入，验证走 AskUser 路径。"""

    def test_ask_user_path(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        mock_llm_responses = {
            "信息缺口评估": json.dumps({
                "decision": "ask_user",
                "info_needs": [
                    {
                        "kind": "missing_param",
                        "question": "请问您使用的是哪种 SDK 语言？（JavaScript/Rust/Go）",
                        "required": True,
                    }
                ],
                "reasoning": "用户没有指定 SDK 语言，无法提供对应代码示例",
            }),
        }

        mock_json_responses = {
            "信息缺口评估": {
                "decision": "ask_user",
                "info_needs": [
                    {
                        "kind": "missing_param",
                        "question": "请问您使用的是哪种 SDK 语言？（JavaScript/Rust/Go）",
                        "required": True,
                    }
                ],
                "reasoning": "用户没有指定 SDK 语言，无法提供对应代码示例",
            },
        }

        state = _make_state(
            user_message={"content": "怎么用 SDK 发交易？"},
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm",
                    _mock_call_llm_factory(mock_llm_responses)), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json",
                    _mock_call_llm_json_factory(mock_json_responses)):
            graph = build_full_graph()
            result = graph.invoke(state)

        response = result.get("_final_response", {})
        assert response.get("need_user_input") is True
        assert "SDK" in response.get("ask_user_question", "") or "SDK" in response.get("text", "")


# -----------------------------------------------------------------------
# M7-T14: "Fiber 开通道" 场景 — 有证据，走 AnswerComposer
# -----------------------------------------------------------------------

class TestFiberChannelScenario:
    """构造带 evidence 的输入，验证走 AnswerComposer 并输出引用。"""

    def test_answer_with_citations(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        mock_evidence = [
            {
                "id": "ev-001",
                "source": "qdrant",
                "title": "Fiber Channel Open API",
                "url": "https://docs.nervos.org/fiber/channel",
                "anchor": "section:open-channel",
                "snippet": "调用 open_channel() 方法，传入对方节点 ID 和初始容量即可开通 Fiber 支付通道。",
                "score": 0.92,
                "payload": {"source": "rfcs", "type": "doc", "version": "0.3"},
                "hash": "abc123",
                "retrieved_ts_ms": 1700000000000,
            },
            {
                "id": "ev-002",
                "source": "github",
                "title": "fiber-sdk-js/examples/channel.ts",
                "url": "https://github.com/nervosnetwork/fiber-sdk-js/blob/main/examples/channel.ts",
                "anchor": "L15-L30",
                "snippet": "const channel = await fiber.openChannel({ peerId, capacity: '1000000000' });",
                "score": 0.88,
                "payload": {"source": "github", "type": "code", "version": "0.3"},
                "hash": "def456",
                "retrieved_ts_ms": 1700000000000,
            },
        ]

        mock_llm_responses = {
            "信息缺口评估": json.dumps({
                "decision": "has_needs",
                "info_needs": [
                    {"kind": "concept_gap", "question": "Fiber 开通道的 API", "required": False}
                ],
            }),
            "检索规划": json.dumps({
                "plan_id": "plan_test",
                "rationale": "search for Fiber channel open",
                "steps": [{"step_id": "step_1", "tool": "qdrant_search", "query": "Fiber open channel", "filters": {"source": "rfcs"}, "top_k": 5}],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 3},
            }),
            "证据评分": json.dumps({"grade": "enough", "reasoning": "证据覆盖了核心问题"}),
            "回答组装": (
                "使用 Fiber SDK 开通支付通道需要调用 `open_channel()` 方法 {{cite:E1}}。\n\n"
                "```typescript\n"
                "const channel = await fiber.openChannel({ peerId, capacity: '1000000000' });\n"
                "```\n"
                "这个示例来自 SDK 示例代码 {{cite:E2}}。\n"
            ),
            "自检": json.dumps({"pass": True, "issues": [], "reasoning": "引用完整，格式正确"}),
        }

        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "info_needs": [
                    {"kind": "concept_gap", "question": "Fiber 开通道的 API", "required": False}
                ],
            },
            "检索规划": {
                "plan_id": "plan_test",
                "rationale": "search for Fiber channel open",
                "steps": [{"step_id": "step_1", "tool": "qdrant_search", "query": "Fiber open channel", "filters": {"source": "rfcs"}, "top_k": 5}],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 3},
            },
            "证据评分": {"grade": "enough", "reasoning": "证据覆盖了核心问题"},
            "自检": {"pass": True, "issues": [], "reasoning": "引用完整，格式正确"},
        }

        state = _make_state(
            user_message={"content": "怎么用 Fiber SDK 开通支付通道？"},
            evidence=mock_evidence,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm",
                    _mock_call_llm_factory(mock_llm_responses)), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json",
                    _mock_call_llm_json_factory(mock_json_responses)):
            graph = build_full_graph()
            result = graph.invoke(state)

        response = result.get("_final_response", {})
        assert response.get("text"), "回答文本不应为空"
        assert response.get("citations"), "应该包含引用"
        assert not response.get("need_user_input", False), "不应该要求用户补充信息"


class TestDirectAnswerScenario:
    """低风险 answer_direct 应短路径直答，不检索、不追加来源。"""

    def test_full_graph_direct_answer_skips_retrieval_and_self_check(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        call_counter = {"info_gap": 0, "direct": 0, "planner": 0, "self_check": 0}

        def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
            _ = user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple direct answer", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                call_counter["info_gap"] += 1
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "identity/help question",
                }
            if "检索规划" in system_prompt:
                call_counter["planner"] += 1
            if "自检" in system_prompt:
                call_counter["self_check"] += 1
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
            _ = user_prompt, json_mode, model, temperature, max_tokens
            if "直接回答器" in system_prompt:
                call_counter["direct"] += 1
                return "我是 Nervos Brain，可以帮你回答 Nervos/CKB/Fiber/CCC 相关问题。"
            return ""

        state = _make_state(user_message={"content": "你是谁"})

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            graph = build_full_graph()
            result = graph.invoke(state)

        response = result.get("_final_response", {})
        assert call_counter["info_gap"] == 1
        assert call_counter["direct"] == 1
        assert call_counter["planner"] == 0
        assert call_counter["self_check"] == 0
        assert response.get("citations") == []
        assert "参考来源" not in response.get("text", "")
        assert result.get("_direct_answer") is True

    def test_full_graph_correction_feedback_uses_direct_answer(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "short correction feedback", "confidence": 0.9}
            assert "信息缺口评估" in system_prompt
            assert "你是不是回复错问题了" in user_prompt
            return {
                "decision": "answer_direct",
                "retrieval_policy": "none",
                "info_needs": [],
                "reasoning": "pure answer quality feedback",
            }

        def mock_call_llm(system_prompt, user_prompt, **_kwargs):
            assert "直接回答器" in system_prompt
            assert "你是不是回复错问题了" in user_prompt
            return "抱歉，刚才可能答偏了。请把你想继续问的问题再发一次，我会按当前问题重新回答。"

        state = _make_state(
            user_message={"content": "你是不是回复错问题了"},
            recent_messages=[
                {"role": "user", "content": "有没有比较靠谱的资料可以看？"},
                {"role": "assistant", "content": "我不需要你补充版本或环境。"},
            ],
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        response = result.get("_final_response", {})
        assert result.get("_direct_answer") is True
        assert result.get("retrieval_policy") == "none"
        assert response.get("text", "").startswith("抱歉")
        assert not result.get("evidence")

    def test_direct_answer_can_use_recent_conversation_context(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        captured: dict[str, str] = {}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            _ = model
            if "信息缺口评估" in system_prompt:
                captured["info_gap_user_prompt"] = user_prompt
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "context follow-up",
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
            _ = system_prompt, json_mode, model, temperature, max_tokens
            captured["direct_user_prompt"] = user_prompt
            return "你刚才问的是：CKB 是什么。"

        state = _make_state(
            user_message={"content": "你看看上文是什么"},
            recent_messages=[
                {"role": "user", "content": "CKB 是什么", "created_ts_ms": 1000},
                {"role": "assistant", "content": "CKB 是 Nervos 的底层公链。", "created_ts_ms": 1100},
            ],
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert "CKB 是什么" in captured["info_gap_user_prompt"]
        assert "CKB 是什么" in captured["direct_user_prompt"]
        assert "CKB 是什么" in result["_final_response"]["text"]

    def test_reply_context_is_not_overwritten_by_recent_messages(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        captured: dict[str, str] = {}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            _ = model
            if "信息缺口评估" in system_prompt:
                captured["info_gap_user_prompt"] = user_prompt
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "reply follow-up",
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
            _ = system_prompt, json_mode, model, temperature, max_tokens
            captured["direct_user_prompt"] = user_prompt
            return "小白版就是：CKB 可以先理解成 Nervos 的底层账本和资产容器。"

        reply_context = (
            "当前消息正在回复这条 assistant 消息: "
            "CKB 通常指 Nervos CKB，是 Nervos 生态里的公链基础层。"
        )
        state = _make_state(
            user_message={"content": "好，小白版的解释，你说一下"},
            conversation_context=reply_context,
            recent_messages=[
                {"role": "user", "content": "Fiber WASM 是怎么实现的？", "created_ts_ms": 1000},
                {"role": "assistant", "content": "Fiber WASM 需要查源码。", "created_ts_ms": 1100},
            ],
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert reply_context in captured["info_gap_user_prompt"]
        assert reply_context in captured["direct_user_prompt"]
        assert "Fiber WASM" not in captured["info_gap_user_prompt"]
        assert "Fiber WASM" not in captured["direct_user_prompt"]
        assert "CKB" in result["_final_response"]["text"]

    def test_self_contained_question_gates_plain_recent_history(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        captured: dict[str, str] = {}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            _ = model
            if "信息缺口评估" in system_prompt:
                captured["info_gap_user_prompt"] = user_prompt
                return {
                    "decision": "has_needs",
                    "retrieval_policy": "single",
                    "info_needs": [
                        {
                            "kind": "latest_spec",
                            "question": "Tentacle 通信方式和协议",
                            "required": False,
                        }
                    ],
                }
            if "检索规划器" in system_prompt:
                captured["planner_user_prompt"] = user_prompt
                return {
                    "plan_id": "p1",
                    "rationale": "current question only",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "tool": "qdrant_search",
                            "query": "tentacle transport protocol yamux secio identify discovery",
                            "top_k": 5,
                        }
                    ],
                    "parallel_groups": [["step_1"]],
                    "budget": {"max_tool_calls": 1},
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        class Retriever:
            def search(self, query: str, filters=None, top_k: int = 5, regex_queries=None):
                _ = query, filters, top_k, regex_queries
                return [
                    {
                        "id": "ev-1",
                        "source": "qdrant",
                        "title": "tentacle README",
                        "url": "https://github.com/nervosnetwork/tentacle/blob/master/README.md",
                        "anchor": "a1",
                        "snippet": "Tentacle is a multiplexed p2p network framework.",
                        "score": 0.9,
                        "payload": {"source": "github_docs"},
                        "hash": "h1",
                        "retrieved_ts_ms": 1,
                    },
                    {
                        "id": "ev-2",
                        "source": "qdrant",
                        "title": "secio README",
                        "url": "https://github.com/nervosnetwork/tentacle/blob/master/secio/README.md",
                        "anchor": "a2",
                        "snippet": "Secio is an encrypted communication protocol library.",
                        "score": 0.8,
                        "payload": {"source": "github_docs"},
                        "hash": "h2",
                        "retrieved_ts_ms": 1,
                    },
                ]

        def mock_call_llm(system_prompt, user_prompt, **_kwargs):
            _ = system_prompt, user_prompt
            return "Tentacle 支持 P2P 多路复用 {{cite:E1}} 和加密通信 {{cite:E2}}。"

        state = _make_state(
            user_message={"content": "tentacle支持哪些通信方式和协议？"},
            recent_messages=[
                {"role": "user", "content": "我要做最好的 CKB agent"},
                {"role": "assistant", "content": "可以参考 Nervos Brain 项目"},
            ],
            retriever=Retriever(),
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert "上下文门控" in captured["info_gap_user_prompt"]
        assert "上下文门控" in captured["planner_user_prompt"]
        assert "Nervos Brain" not in captured["info_gap_user_prompt"]
        assert "Nervos Brain" not in captured["planner_user_prompt"]
        assert "CKB agent" not in captured["info_gap_user_prompt"]
        assert "CKB agent" not in captured["planner_user_prompt"]
        assert result["conversation_context"].startswith("上下文门控")

    def test_self_contained_question_overrides_plain_runtime_context(self):
        from nervos_brain.graph_engine.full_nodes import _conversation_context_from_state

        state = _make_state(
            user_message={"content": "tentacle支持哪些通信方式和协议？"},
            conversation_context=(
                "user: 我要做最好的 CKB agent\n"
                "assistant: 可以参考 Nervos Brain 项目"
            ),
        )

        context = _conversation_context_from_state(state)

        assert context.startswith("上下文门控")
        assert "Nervos Brain" not in context
        assert "CKB agent" not in context

    def test_direct_answer_passes_image_paths_to_llm(self, tmp_path):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        image_path = tmp_path / "screenshot.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        captured: dict[str, Any] = {}

        def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
            _ = user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "image question",
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, **kwargs):
            _ = system_prompt, user_prompt
            captured["image_paths"] = kwargs.get("image_paths")
            return "图里是 CKB 相关截图。"

        state = _make_state(
            user_message={
                "content": "看看这张图",
                "attachments": [{"kind": "image", "local_path": str(image_path), "name": "screenshot.jpg"}],
            },
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert captured["image_paths"] == [str(image_path)]
        assert "CKB" in result["_final_response"]["text"]


class TestSingleRetrievalScenario:
    """single policy 应最多做一轮轻量检索。"""

    def test_retriever_planner_injects_source_registry_and_normalizes_source_alias(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner

        captured: dict[str, str] = {}

        def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
            _ = user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            captured["planner_system_prompt"] = system_prompt
            return {
                "plan_id": "p-docs",
                "rationale": "official docs",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "CKB official tutorial beginner docs",
                        "filters": {"source": "official_docs"},
                        "top_k": 5,
                    }
                ],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 1},
            }

        state = _make_state(
            user_message={"content": "官方没有比较好的教程吗？"},
            info_needs=[
                {
                    "kind": "latest_spec",
                    "question": "CKB 官方新手教程和学习路径",
                    "required": False,
                }
            ],
            retrieval_policy="single",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = retriever_planner(state)

        prompt = captured["planner_system_prompt"]
        assert "source=github_docs" in prompt
        assert "source=github_code" in prompt
        assert "source=nervos_talk" in prompt
        assert "不要自造 official_docs" in prompt
        step = out["retrieval_plan"]["steps"][0]
        assert step["filters"] == {"source": "github_docs"}
        assert "mapped_source:official_docs->github_docs" in step["filter_notes"]

    def test_retriever_planner_normalizes_code_source_alias(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner

        def mock_call_llm_json(system_prompt: str, user_prompt: str, **_kwargs):
            _ = system_prompt, user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            return {
                "plan_id": "plan_code",
                "rationale": "source code lookup",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "fiber open_channel source code",
                        "filters": {"source": "code", "topic": "nervosnetwork/fiber"},
                        "top_k": 5,
                    }
                ],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 1},
            }

        state = _make_state(
            user_message={"content": "Fiber open_channel 的源码在哪里？"},
            info_needs=[
                {
                    "kind": "latest_spec",
                    "question": "Fiber open_channel 源码",
                    "required": False,
                }
            ],
            retrieval_policy="single",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = retriever_planner(state)

        step = out["retrieval_plan"]["steps"][0]
        assert step["filters"] == {"source": "github_code", "topic": "nervosnetwork/fiber"}
        assert "mapped_source:code->github_code" in step["filter_notes"]

    def test_retriever_planner_respects_llm_unified_search_plan(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner

        def mock_call_llm_json(system_prompt: str, user_prompt: str, **_kwargs):
            _ = system_prompt, user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "technical retrieval", "confidence": 0.9}
            return {
                "plan_id": "plan_ccc",
                "rationale": "tutorial lookup",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "TS/JS CKB transfer CCC @ckb-ccc tutorial",
                        "filters": {},
                        "top_k": 5,
                    }
                ],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 1},
            }

        state = _make_state(
            user_message={"content": "我是 TS/JS 小白，想用 CCC 写 CKB 转账最简教程。"},
            info_needs=[
                {
                    "kind": "latest_spec",
                    "question": "CCC TypeScript CKB transfer tutorial and examples",
                    "required": False,
                }
            ],
            retrieval_policy="single",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = retriever_planner(state)

        step = out["retrieval_plan"]["steps"][0]
        assert step["tool"] == "qdrant_search"
        assert step["filters"] == {}
        assert "filter_notes" not in step

    def test_retriever_planner_keeps_source_filter_when_user_limits_to_talk(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner

        def mock_call_llm_json(system_prompt: str, user_prompt: str, **_kwargs):
            _ = system_prompt, user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "forum lookup", "confidence": 0.9}
            return {
                "plan_id": "plan_talk",
                "rationale": "forum lookup",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "CCC Fiber Talk forum discussion",
                        "filters": {"source": "nervos_talk"},
                        "top_k": 5,
                    }
                ],
                "parallel_groups": [["step_1"]],
                "budget": {"max_tool_calls": 1},
            }

        state = _make_state(
            user_message={"content": "Talk 里有没有 CCC/Fiber 讨论？"},
            info_needs=[
                {
                    "kind": "historical_consensus",
                    "question": "Nervos Talk CCC Fiber discussion",
                    "required": False,
                }
            ],
            retrieval_policy="single",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = retriever_planner(state)

        step = out["retrieval_plan"]["steps"][0]
        assert step["filters"] == {"source": "nervos_talk"}
        assert step.get("filter_notes") is None

    def test_retrieval_executor_preserves_filter_notes(self):
        from nervos_brain.graph_engine.full_nodes import retrieval_executor

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def search(self, query: str, filters=None, top_k: int = 5, regex_queries=None):
                self.calls.append(
                    {
                        "query": query,
                        "filters": filters,
                        "top_k": top_k,
                        "regex_queries": regex_queries,
                    }
                )
                return [
                    {
                        "id": "ccc-transfer",
                        "source": "qdrant",
                        "title": "CCC transfer example",
                        "url": "https://github.com/ckb-devrel/ccc",
                        "anchor": "ccc-transfer",
                        "snippet": "TypeScript transfer example.",
                        "score": 0.9,
                        "payload": {
                            "source": "github_code",
                            "backend": "retrieval_github_code",
                            "regex_queries_count": 1,
                            "regex_valid_count": 1,
                            "regex_dropped_count": 0,
                        },
                        "hash": "h-ccc",
                        "retrieved_ts_ms": 1,
                    }
                ]

        retriever = FakeRetriever()
        state = _make_state(
            request_id="test-filter-note-trace",
            _multi_retriever=retriever,
            retrieval_plan={
                "plan_id": "p-filter-note",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "TS/JS CKB transfer CCC",
                        "filters": {},
                        "regex_queries": [
                            {
                                "label": "CCC",
                                "pattern": r"(?i)\bccc\b",
                                "fields": ["title", "keywords"],
                            }
                        ],
                        "filter_notes": ["mapped_source:docs->github_docs"],
                        "top_k": 5,
                    }
                ],
            },
            budget={"max_tool_calls": 2, "max_evidence_chunks": 5},
        )

        out = retrieval_executor(state)

        assert retriever.calls == [
            {
                "query": "TS/JS CKB transfer CCC",
                "filters": None,
                "top_k": 5,
                "regex_queries": [
                    {
                        "label": "CCC",
                        "pattern": r"(?i)\bccc\b",
                        "fields": ["title", "keywords"],
                    }
                ],
            }
        ]
        assert out["_tool_execution_trace"][0]["filter_notes"] == [
            "mapped_source:docs->github_docs"
        ]
        assert out["_tool_execution_trace"][0]["regex_valid_count"] == 1

    def test_retrieval_executor_retries_empty_filtered_qdrant_without_filters(self):
        from nervos_brain.graph_engine.full_nodes import retrieval_executor

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls: list[dict | None] = []

            def search(self, query: str, filters=None, top_k: int = 5):
                _ = query, top_k
                self.calls.append(filters)
                if filters:
                    return []
                return [
                    {
                        "id": "docs-getting-started",
                        "source": "qdrant",
                        "title": "CKB Getting Started",
                        "url": "https://docs.nervos.org/",
                        "anchor": "getting-started",
                        "snippet": "Official CKB getting started guide.",
                        "score": 0.9,
                        "payload": {"source": "github_docs", "topic": "nervosnetwork/docs.nervos.org"},
                        "hash": "h-docs",
                        "retrieved_ts_ms": 1,
                    }
                ]

        retriever = FakeRetriever()
        state = _make_state(
            request_id="test-source-fallback",
            _multi_retriever=retriever,
            retrieval_plan={
                "plan_id": "p-docs",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "qdrant_search",
                        "query": "CKB official tutorial beginner docs",
                        "filters": {"source": "official_docs"},
                        "top_k": 5,
                    }
                ],
            },
            budget={"max_tool_calls": 2, "max_evidence_chunks": 5},
        )

        out = retrieval_executor(state)

        assert retriever.calls == [{"source": "github_docs"}, None]
        assert out["evidence"][0]["payload"]["source"] == "github_docs"
        assert out["_tool_calls_executed"] == 2
        assert out["_tool_execution_trace"][0]["status"] == "empty"
        assert "mapped_source:official_docs->github_docs" in out["_tool_execution_trace"][0]["filter_notes"]
        assert out["_tool_execution_trace"][1]["fallback_reason"] == "empty_filtered_qdrant_search"

    def test_nervos_brain_progress_uses_forum_evidence(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        class FakeTransport:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def send(self, payload: dict) -> dict:
                self.calls.append(payload)
                return {
                    "evidence": [
                        {
                            "id": "talk-progress",
                            "title": "Nervos Brain Week 4 Progress",
                            "url": "https://talk.nervos.org/t/example",
                            "anchor": "week-4",
                            "snippet": "Nervos Brain has built the GitHub ingestion pipeline and RAG loop.",
                            "score": 0.95,
                            "payload": {"source": "talk"},
                            "hash": "h1",
                            "retrieved_ts_ms": 1,
                        }
                    ],
                    "raw_size_bytes": 120,
                    "redactions_applied": [],
                }

        def mock_call_llm_json(system_prompt, user_prompt, **_kwargs):
            _ = user_prompt
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                return {
                    "decision": "has_needs",
                    "retrieval_policy": "single",
                    "info_needs": [
                        {
                            "kind": "latest_spec",
                            "question": "Nervos Brain 项目最新公开进度",
                            "required": False,
                        }
                    ],
                }
            if "检索规划" in system_prompt:
                return {
                    "plan_id": "p-progress",
                    "rationale": "Talk progress reports are the likely source",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "tool": "discourse_query",
                            "query": "Nervos Brain 目前进度 Spark Program 周报",
                            "filters": {},
                            "top_k": 5,
                        }
                    ],
                    "parallel_groups": [["step_1"]],
                    "budget": {"max_tool_calls": 1},
                }
            if "证据评分" in system_prompt or "自检" in system_prompt:
                return {"decision": "accept_answer", "reasoning": "enough", "uncertainty_score": 0.1}
            return {}

        def mock_call_llm(system_prompt, user_prompt, **_kwargs):
            assert "回答组装器" in system_prompt
            assert "Nervos Brain" in user_prompt
            return "Nervos Brain 目前处于早期工程化推进阶段，已打通数据入库和 RAG 闭环 {{cite:E1}}。"

        transport = FakeTransport()
        state = _make_state(
            user_message={"content": "nervos brain 这个项目目前进度如何了"},
            _tool_transport=transport,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert transport.calls[0]["tool"] == "discourse_query"
        assert "Nervos Brain" in result["_final_response"]["text"]
        assert result["evidence"][0]["source"] == "discourse"

    def test_single_retrieval_path_stops_after_one_hop(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        call_counter = {"planner": 0, "pre": 0, "post": 0, "answer": 0}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None, service_tier=None, **_kwargs):
            _ = user_prompt, model, service_tier
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "technical graph node", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                return {
                    "decision": "has_needs",
                    "retrieval_policy": "single",
                    "info_needs": [{"kind": "concept_gap", "question": "CKB definition", "required": False}],
                }
            if "检索规划" in system_prompt:
                call_counter["planner"] += 1
                return {
                    "plan_id": "p-single",
                    "rationale": "one query",
                    "steps": [
                        {"step_id": "s1", "tool": "qdrant_search", "query": "CKB definition", "filters": {}, "top_k": 3},
                        {"step_id": "s2", "tool": "github_search", "query": "CKB docs", "filters": {}, "top_k": 3},
                        {"step_id": "s3", "tool": "discourse_query", "query": "CKB history", "filters": {}, "top_k": 3},
                    ],
                    "parallel_groups": [["s1", "s2", "s3"]],
                    "budget": {"max_tool_calls": 3},
                }
            if "证据评分" in system_prompt:
                call_counter["pre"] += 1
                return {"decision": "accept_answer", "reasoning": "enough", "uncertainty_score": 0.2}
            if "自检" in system_prompt:
                call_counter["post"] += 1
                return {"decision": "accept_answer", "reasoning": "ok", "uncertainty_score": 0.1}
            return {}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
            _ = user_prompt, json_mode, model, temperature, max_tokens
            if "回答组装" in system_prompt:
                call_counter["answer"] += 1
                return "CKB 是 Nervos 的 Layer 1 区块链 {{cite:E1}}。"
            return ""

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, query: str, filters=None, top_k: int = 5):
                _ = query, filters, top_k
                self.calls += 1
                return [
                    {
                        "id": f"ev-{self.calls}",
                        "source": "qdrant",
                        "title": "CKB intro",
                        "url": "https://example.com/ckb",
                        "anchor": "doc",
                        "snippet": "CKB is the layer-1 blockchain of Nervos.",
                        "score": 0.9,
                        "payload": {"source": "docs", "version": "v1"},
                        "hash": f"h-{self.calls}",
                        "retrieved_ts_ms": 1,
                    }
                ]

        retriever = FakeRetriever()
        state = _make_state(
            user_message={"content": "ckb是什么"},
            _multi_retriever=retriever,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            graph = build_full_graph()
            result = graph.invoke(state)

        assert call_counter["planner"] == 1
        assert call_counter["pre"] == 1
        assert call_counter["post"] == 1
        assert call_counter["answer"] == 1
        assert retriever.calls <= 2
        assert result.get("hop_count") == 1
        assert result.get("budget", {}).get("max_hops") == 1
        assert result.get("_final_response", {}).get("text")

    def test_retriever_planner_receives_recent_conversation_context(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner

        captured: dict[str, str] = {}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            _ = system_prompt, model
            captured["user_prompt"] = user_prompt
            return {
                "plan_id": "p-context",
                "rationale": "use context",
                "steps": [
                    {"step_id": "s1", "tool": "qdrant_search", "query": "JS SDK CKB", "filters": {}, "top_k": 3}
                ],
                "parallel_groups": [["s1"]],
                "budget": {"max_tool_calls": 1},
            }

        state = _make_state(
            user_message={"content": "那 JS SDK 怎么写"},
            retrieval_policy="single",
            info_needs=[{"kind": "concept_gap", "question": "SDK 示例", "required": False}],
            recent_messages=[
                {"role": "user", "content": "我想写一个 CKB 转账示例", "created_ts_ms": 1}
            ],
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = retriever_planner(state)

        assert "我想写一个 CKB 转账示例" in captured["user_prompt"]
        assert out["retrieval_plan"]["steps"][0]["query"] == "JS SDK CKB"


# -----------------------------------------------------------------------
# M7-T15: "证据冲突" 场景 — 构造 conflicts，DocGrader 触发 replan
# -----------------------------------------------------------------------

class TestEvidenceConflictScenario:
    """构造 conflicts，验证 DocGrader 触发 replan (need_more)。"""

    def test_conflict_triggers_replan(self):
        from nervos_brain.graph_engine.full_nodes import doc_grader

        conflict_evidence = [
            {
                "id": "ev-A", "source": "qdrant", "title": "Doc A",
                "url": "https://a.com", "anchor": "s1",
                "snippet": "Fiber 版本 0.2 的 API", "score": 0.9,
                "payload": {"source": "rfcs", "type": "doc", "version": "0.2"},
                "hash": "aaa", "retrieved_ts_ms": 1700000000000,
            },
            {
                "id": "ev-B", "source": "qdrant", "title": "Doc B",
                "url": "https://b.com", "anchor": "s2",
                "snippet": "Fiber 版本 0.3 的 API 已完全改变", "score": 0.85,
                "payload": {"source": "rfcs", "type": "doc", "version": "0.3"},
                "hash": "bbb", "retrieved_ts_ms": 1700000000000,
            },
        ]

        conflicts = [
            {"a_id": "ev-A", "b_id": "ev-B", "reason": "version_mismatch"},
        ]

        state = _make_state(
            user_message={"content": "Fiber 的 open_channel API 怎么用？"},
            evidence=conflict_evidence,
            conflicts=conflicts,
        )

        mock_json_responses = {
            "证据评分": {"grade": "need_more", "reasoning": "证据版本冲突", "missing_aspects": ["需要确认最新版本"]},
        }

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json",
                    _mock_call_llm_json_factory(mock_json_responses)):
            result = doc_grader(state)

        assert result["_grade"] == "need_more"

    def test_full_graph_conflict_triggers_replan_then_answer(self):
        """完整图：冲突 -> replan -> 最终回答。"""
        from nervos_brain.graph_engine.full_graph import build_full_graph

        call_counter = {"info_gap": 0, "planner": 0, "grader": 0, "answer": 0, "self_check": 0}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            if "信息缺口评估" in system_prompt:
                call_counter["info_gap"] += 1
                return {
                    "decision": "has_needs",
                    "info_needs": [{"kind": "concept_gap", "question": "Fiber API", "required": False}],
                }
            elif "检索规划" in system_prompt:
                call_counter["planner"] += 1
                return {
                    "plan_id": f"plan_{call_counter['planner']}",
                    "rationale": "replan search",
                    "steps": [{"step_id": "step_1", "tool": "qdrant_search", "query": "Fiber", "filters": {}, "top_k": 5}],
                    "parallel_groups": [["step_1"]],
                    "budget": {"max_tool_calls": 3},
                }
            elif "证据评分" in system_prompt:
                call_counter["grader"] += 1
                if call_counter["grader"] <= 1:
                    return {"grade": "need_more", "reasoning": "conflict", "missing_aspects": []}
                return {"grade": "enough", "reasoning": "ok"}
            elif "自检" in system_prompt:
                call_counter["self_check"] += 1
                return {"pass": True, "issues": [], "reasoning": "ok"}
            return {}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048):
            if "回答组装" in system_prompt:
                call_counter["answer"] += 1
                return "Fiber 通道的正确用法是... {{cite:E1}}"
            return json.dumps(mock_call_llm_json(system_prompt, user_prompt))

        state = _make_state(
            user_message={"content": "Fiber open_channel 怎么用？"},
            evidence=[{
                "id": "ev-1", "source": "qdrant", "title": "Fiber Doc",
                "url": "https://fiber.com", "anchor": "s1",
                "snippet": "open_channel API docs", "score": 0.9,
                "payload": {"source": "rfcs", "type": "doc"},
                "hash": "h1", "retrieved_ts_ms": 1700000000000,
            }],
            conflicts=[{"a_id": "ev-1", "b_id": "ev-2", "reason": "version_mismatch"}],
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            graph = build_full_graph()
            result = graph.invoke(state)

        assert call_counter["grader"] >= 2, "DocGrader 应至少被调用两次（首次 need_more + 后续 enough）"
        response = result.get("_final_response", {})
        assert response.get("text"), "最终应产出回答"


# -----------------------------------------------------------------------
# 路由函数单元测试
# -----------------------------------------------------------------------

class TestRoutingFunctions:
    """测试条件路由函数。"""

    def test_route_after_assessment_ask_user(self):
        from nervos_brain.graph_engine.full_graph import route_after_assessment
        assert route_after_assessment({"_route_decision": "ask_user"}) == "answer_composer"
        assert route_after_assessment(
            {
                "_route_decision": "ask_user",
                "info_needs": [{"required": True, "question": "请贴完整报错日志"}],
            }
        ) == "ask_user"

    def test_route_after_assessment_has_needs(self):
        from nervos_brain.graph_engine.full_graph import route_after_assessment
        assert route_after_assessment({"_route_decision": "has_needs"}) == "retriever_planner"

    def test_route_after_assessment_answer_direct(self):
        from nervos_brain.graph_engine.full_graph import route_after_assessment
        assert route_after_assessment({"_route_decision": "answer_direct"}) == "answer_composer"

    def test_route_after_answer_composer_direct_skips_self_check(self):
        from nervos_brain.graph_engine.full_graph import route_after_answer_composer
        assert route_after_answer_composer({"_direct_answer": True}) == "format_repair"

    def test_route_after_answer_composer_evidence_answer_runs_self_check(self):
        from nervos_brain.graph_engine.full_graph import route_after_answer_composer
        assert route_after_answer_composer({}) == "self_check"

    def test_route_after_grading_need_more(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading
        assert route_after_grading({"_grade": "need_more", "retry_count": 0}) == "retriever_planner"

    def test_route_after_grading_enough(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading
        assert route_after_grading({"_grade": "enough"}) == "answer_composer"

    def test_route_after_grading_exhausted(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading
        assert route_after_grading({"_grade": "need_more", "retry_count": 3}) == "answer_composer"

    def test_route_after_grading_exhausted_with_evidence_answers(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading

        assert (
            route_after_grading(
                {
                    "reflection_decision": "continue_retrieval",
                    "hop_count": 3,
                    "evidence": [{"id": "ev-1", "snippet": "CKB is a layer-1 blockchain."}],
                    "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
                }
            )
            == "answer_composer"
        )

    def test_route_after_grading_exhausted_with_conflict_and_evidence_answers(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading

        assert (
            route_after_grading(
                {
                    "reflection_decision": "continue_retrieval",
                    "hop_count": 3,
                    "evidence": [{"id": "ev-1", "snippet": "Fiber setup docs"}],
                    "conflicts": [{"a_id": "ev-1", "b_id": "ev-2", "reason": "version_mismatch"}],
                    "info_needs": [
                        {
                            "kind": "latest_spec",
                            "question": "继续检索 Fiber 官方文档",
                            "required": False,
                        }
                    ],
                    "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
                }
            )
            == "answer_composer"
        )

    def test_route_after_grading_exhausted_with_required_param_still_asks(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading

        assert (
            route_after_grading(
                {
                    "reflection_decision": "continue_retrieval",
                    "hop_count": 3,
                    "info_needs": [
                        {
                            "kind": "missing_param",
                            "question": "请问你使用的是哪个 SDK？",
                            "required": True,
                        }
                    ],
                    "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
                }
            )
            == "ask_user"
        )

    def test_route_after_grading_exhausted_respects_required_need(self):
        from nervos_brain.graph_engine.full_graph import route_after_grading

        assert (
            route_after_grading(
                {
                    "reflection_decision": "continue_retrieval",
                    "hop_count": 3,
                    "info_needs": [
                        {
                            "kind": "latest_spec",
                            "question": "需要获取 Fiber 节点当前官方部署方式、配置项、运行命令、RPC/API 或管理接口等最新资料。",
                            "required": True,
                        }
                    ],
                    "evidence": [{"id": "ev-1", "snippet": "Fiber setup docs"}],
                    "budget": {"max_hops": 3, "max_reflection_rounds_pre": 2},
                }
            )
            == "ask_user"
        )

    def test_route_after_self_check_pass(self):
        from nervos_brain.graph_engine.full_graph import route_after_self_check
        assert route_after_self_check({"_self_check_pass": True}) == "format_repair"

    def test_route_after_self_check_fail_retry(self):
        from nervos_brain.graph_engine.full_graph import route_after_self_check
        assert route_after_self_check({"_self_check_pass": False, "retry_count": 0}) == "answer_composer"

    def test_route_after_self_check_fail_exhausted(self):
        from nervos_brain.graph_engine.full_graph import route_after_self_check
        assert (
            route_after_self_check(
                {
                    "_self_check_pass": False,
                    "_reflection_rounds_post": 3,
                    "budget": {"max_reflection_rounds_post": 2},
                }
            )
            == "format_repair"
        )


# -----------------------------------------------------------------------
# Node 单元测试
# -----------------------------------------------------------------------

class TestFormatRepairNode:
    """FormatRepair 不依赖 LLM，直接测。"""

    def test_fixes_markdown_and_citations(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r1",
            "locale": "zh-CN",
            "_final_response": {
                "request_id": "r1",
                "text": "答案 [1] 代码块\n```python\nprint('hello')",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "s1", "title": "Doc A"},
                ],
            },
        }
        result = format_repair(state)
        resp = result["_final_response"]
        assert "```" in resp["text"]
        assert len(resp["citations"]) == 1
        assert "## 参考来源" in resp["text"]
        assert "https://a.com" in resp["text"]

    def test_format_repair_uses_english_reference_section_for_english_locale(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-en-ref-section",
            "locale": "en",
            "_final_response": {
                "request_id": "r-en-ref-section",
                "text": "Answer [1]",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "s1", "title": ""},
                ],
            },
        }

        result = format_repair(state)
        text = result["_final_response"]["text"]
        assert "## References" in text
        assert "## 参考来源" not in text
        assert "**https://a.com**" in text

    def test_format_repair_compiles_inline_evidence_citations(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-inline-cites",
            "locale": "en",
            "evidence": [
                {"title": "Doc A", "url": "https://a.com", "anchor": "a1"},
                {"title": "Doc B", "url": "https://b.com", "anchor": "b1"},
            ],
            "_final_response": {
                "request_id": "r-inline-cites",
                "text": "Second fact {{cite:E2}}. First fact {{ cite: E1 }}.",
                "citations": [],
            },
        }

        result = format_repair(state)
        resp = result["_final_response"]
        text = resp["text"]
        assert "Second fact [1]. First fact [2]." in text
        assert "{{cite:" not in text
        assert resp["citations"] == [
            {"label": "[1]", "url": "https://b.com", "anchor": "b1", "title": "Doc B"},
            {"label": "[2]", "url": "https://a.com", "anchor": "a1", "title": "Doc A"},
        ]
        assert "## References" in text
        assert "[1] **Doc B**" in text
        assert "[2] **Doc A**" in text

    def test_format_repair_reuses_duplicate_inline_evidence_citations(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-inline-duplicate",
            "evidence": [
                {"title": "Doc A", "url": "https://a.com", "anchor": "a1"},
            ],
            "_final_response": {
                "request_id": "r-inline-duplicate",
                "text": "Fact {{cite:E1}}. Same source {{cite:E1}}.",
                "citations": [],
            },
        }

        result = format_repair(state)
        resp = result["_final_response"]
        assert "Fact [1]. Same source [1]." in resp["text"]
        assert len(resp["citations"]) == 1

    def test_format_repair_drops_invalid_inline_evidence_citations(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-inline-invalid",
            "evidence": [
                {"title": "Doc A", "url": "https://a.com", "anchor": "a1"},
            ],
            "_final_response": {
                "request_id": "r-inline-invalid",
                "text": "Valid {{cite:E1}}. Invalid {{cite:E99}}.",
                "citations": [],
            },
        }

        result = format_repair(state)
        resp = result["_final_response"]
        assert "Valid [1]. Invalid ." in resp["text"]
        assert len(resp["citations"]) == 1

    def test_format_repair_ignores_inline_citations_inside_code_blocks(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-inline-code",
            "evidence": [
                {"title": "Doc A", "url": "https://a.com", "anchor": "a1"},
            ],
            "_final_response": {
                "request_id": "r-inline-code",
                "text": "Use this {{cite:E1}}\n```txt\nliteral {{cite:E1}}\n```",
                "citations": [],
            },
        }

        result = format_repair(state)
        text = result["_final_response"]["text"]
        assert "Use this [1]" in text
        assert "literal {{cite:E1}}" in text

    def test_format_repair_prefers_inline_citations_over_legacy_labels(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-inline-mixed",
            "evidence": [
                {"title": "Doc A", "url": "https://a.com", "anchor": "a1"},
            ],
            "_final_response": {
                "request_id": "r-inline-mixed",
                "text": "Inline {{cite:E1}}. Old [3].",
                "citations": [
                    {"label": "[3]", "url": "https://wrong.com", "anchor": "w", "title": "Wrong"},
                ],
            },
        }

        result = format_repair(state)
        resp = result["_final_response"]
        assert "Inline [1]. Old ." in resp["text"]
        assert resp["citations"] == [
            {"label": "[1]", "url": "https://a.com", "anchor": "a1", "title": "Doc A"},
        ]
        assert "https://wrong.com" not in resp["text"]

    def test_format_repair_does_not_append_uncertainty_note_when_reflection_exhausted(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-no-uncertainty-note",
            "reflection_decision": "revise_answer",
            "_reflection_rounds_post": 1,
            "budget": {"max_reflection_rounds_post": 1},
            "_final_response": {
                "request_id": "r-no-uncertainty-note",
                "text": "答案 [1]",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "s1", "title": "Doc A"},
                ],
            },
        }

        result = format_repair(state)
        text = result["_final_response"]["text"]
        assert "当前回答存在不确定性" not in text
        assert "## 参考来源" in text

    def test_format_repair_does_not_duplicate_existing_reference_section(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-ref-section",
            "_final_response": {
                "request_id": "r-ref-section",
                "text": "答案 [1]\n\n## 参考来源\n\n[1] Existing\nhttps://a.com",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "s1", "title": "Doc A"},
                ],
            },
        }
        result = format_repair(state)
        text = result["_final_response"]["text"]
        assert text.count("## 参考来源") == 1

    def test_format_repair_does_not_duplicate_existing_english_reference_section(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r-en-existing-ref-section",
            "locale": "en",
            "_final_response": {
                "request_id": "r-en-existing-ref-section",
                "text": "Answer [1]\n\n## References\n\n[1] Existing\nhttps://a.com",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "s1", "title": "Doc A"},
                ],
            },
        }
        result = format_repair(state)
        text = result["_final_response"]["text"]
        assert text.count("## References") == 1

    def test_builds_platform_outbound_message(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r2",
            "user_message": {
                "content": "test",
                "context": {"platform": "telegram", "user_id": "u2"},
            },
            "_final_response": {
                "request_id": "r2",
                "text": "Long text " + ("A" * 4200),
                "citations": [],
            },
        }
        result = format_repair(state)
        outbound = result.get("_outbound_message", {})
        assert outbound.get("context", {}).get("platform") == "telegram"
        assert len(outbound.get("segments", [])) >= 2

    def test_merges_tool_trace_summary(self):
        from nervos_brain.graph_engine.full_nodes import format_repair
        state = {
            "request_id": "r3",
            "_tool_execution_summary": "tools=s1:github_search=ok:1",
            "_final_response": {
                "request_id": "r3",
                "text": "答案 [1]",
                "citations": [
                    {"label": "[1]", "url": "https://a.com", "anchor": "a1", "title": "Doc"},
                ],
            },
        }
        result = format_repair(state)
        assert "github_search" in result["_final_response"]["trace_summary"]


class TestEvidenceMergerNode:
    """EvidenceMerger 不依赖 LLM，直接测。"""

    def test_dedup_and_conflict(self):
        from nervos_brain.graph_engine.full_nodes import evidence_merger
        state = {
            "evidence": [
                {"id": "1", "hash": "aaa", "payload": {"source": "rfcs", "version": "0.2"}},
                {"id": "2", "hash": "aaa", "payload": {"source": "rfcs", "version": "0.2"}},
                {"id": "3", "hash": "bbb", "payload": {"source": "rfcs", "version": "0.3"}},
            ]
        }
        result = evidence_merger(state)
        assert len(result["evidence"]) == 2
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["reason"] == "version_mismatch"


class TestRetrieverPlannerNode:
    """RetrieverPlanner 归一化逻辑测试。"""

    def test_supported_tool_is_preserved(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner
        state = _make_state(
            user_message={"content": "什么是 ckb"},
            info_needs=[{"kind": "concept_gap", "question": "定义", "required": False}],
        )
        mock_json_responses = {
            "检索规划": {
                "plan_id": "p1",
                "rationale": "use github tool",
                "steps": [
                    {
                        "step_id": "s1",
                        "tool": "github_search",
                        "query": "ckb definition",
                        "filters": {},
                        "top_k": 3,
                    }
                ],
                "parallel_groups": [["s1"]],
                "budget": {"max_tool_calls": 3},
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = retriever_planner(state)

        steps = out["retrieval_plan"]["steps"]
        assert steps
        assert steps[0]["tool"] == "github_search"

    def test_unknown_tool_still_falls_back_to_qdrant_search(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner
        state = _make_state(
            user_message={"content": "什么是 ckb"},
            info_needs=[{"kind": "concept_gap", "question": "定义", "required": False}],
        )
        mock_json_responses = {
            "检索规划": {
                "plan_id": "p2",
                "rationale": "unsupported tool",
                "steps": [
                    {
                        "step_id": "s1",
                        "tool": "web_search",
                        "query": "ckb definition",
                        "filters": {},
                        "top_k": 3,
                    }
                ],
                "parallel_groups": [["s1"]],
                "budget": {"max_tool_calls": 3},
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = retriever_planner(state)

        assert out["retrieval_plan"]["steps"][0]["tool"] == "qdrant_search"

    def test_qdrant_regex_queries_are_preserved(self):
        from nervos_brain.graph_engine.full_nodes import retriever_planner
        state = _make_state(
            user_message={"content": "我听说有一个 nervos brain 的项目"},
            info_needs=[{"kind": "historical_consensus", "question": "Nervos Brain 项目背景", "required": False}],
        )
        mock_json_responses = {
            "检索规划": {
                "plan_id": "p-regex",
                "rationale": "named project hard recall",
                "steps": [
                    {
                        "step_id": "s1",
                        "tool": "qdrant_search",
                        "query": "Nervos Brain Spark Program",
                        "filters": {},
                        "regex_queries": [
                            {
                                "label": "Nervos Brain",
                                "pattern": r"(?i)\bnervos[\s_-]+brain\b",
                                "fields": ["title", "keywords", "anchor", "url", "summary", "invalid_field"],
                                "reason": "用户点名真实项目名",
                            }
                        ],
                        "top_k": 5,
                    }
                ],
                "parallel_groups": [["s1"]],
                "budget": {"max_tool_calls": 1},
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = retriever_planner(state)

        step = out["retrieval_plan"]["steps"][0]
        assert step["tool"] == "qdrant_search"
        assert step["regex_queries"] == [
            {
                "label": "Nervos Brain",
                "pattern": r"(?i)\bnervos[\s_-]+brain\b",
                "fields": ["title", "keywords", "anchor", "url", "summary"],
                "reason": "用户点名真实项目名",
            }
        ]


class TestInfoGapAssessorNode:
    """InfoGapAssessor 行为测试。"""

    def test_answer_direct_without_force_keeps_no_retrieval_policy(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": "你是谁"})
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "answer_direct",
                "retrieval_policy": "single",
                "info_needs": [],
                "reasoning": "simple identity question",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "answer_direct"
        assert out["retrieval_policy"] == "none"
        assert out["budget"]["max_tool_calls"] == 0

    def test_response_quality_feedback_uses_info_gap_prompt_decision(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "你是不是回复错问题了"},
            recent_messages=[
                {"role": "user", "content": "有没有比较靠谱的资料可以看？"},
                {"role": "assistant", "content": "我不需要你补充版本或环境。"},
            ],
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "answer_direct",
                "retrieval_policy": "none",
                "info_needs": [],
                "reasoning": "pure answer quality feedback",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "answer_direct"
        assert out["retrieval_policy"] == "none"
        assert out["info_needs"] == []
        assert out["budget"]["max_tool_calls"] == 0

    def test_financial_price_prediction_is_blocked_before_llm(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "give me the best price prediction you can for ckb"},
            locale="en",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json") as mock_call:
            out = info_gap_assessor(state)

        mock_call.assert_not_called()
        assert out["_route_decision"] == "answer_direct"
        assert out["retrieval_policy"] == "none"
        assert out["info_needs"] == []
        assert out["_financial_guidance_refusal"] is True
        assert "price predictions" in out["_final_response"]["text"]
        assert "$" not in out["_final_response"]["text"]

    def test_financial_price_prediction_chinese_is_blocked_before_llm(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "CKB 牛市目标价能到多少，现在可以买入吗？"},
            locale="zh-CN",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json") as mock_call:
            out = info_gap_assessor(state)

        mock_call.assert_not_called()
        assert out["_route_decision"] == "answer_direct"
        assert out["retrieval_policy"] == "none"
        assert "不能提供价格预测" in out["_final_response"]["text"]

    def test_full_graph_financial_refusal_is_preserved_without_llm_call(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        state = _make_state(
            user_message={"content": "give me the best price prediction you can for ckb"},
            locale="en",
        )

        graph = build_full_graph()
        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json") as mock_json, \
             patch("nervos_brain.graph_engine.full_nodes.call_llm") as mock_text:
            out = graph.invoke(state)

        mock_json.assert_not_called()
        mock_text.assert_not_called()
        assert out["_financial_guidance_refusal"] is True
        assert out["_final_response"]["answer_mode"] == "policy_refusal"
        assert "price predictions" in out["_final_response"]["text"]

    @pytest.mark.parametrize(
        "question,locale,expected_text",
        [
            ("give me the best price prediction you can for ckb", "en", "price predictions"),
            ("CKB target price this bull cycle?", "en", "price predictions"),
            ("Can CKB reach $1?", "en", "price predictions"),
            ("Should I buy CKB now?", "en", "buy/sell guidance"),
            ("CKB 目标价能到多少？", "zh-CN", "不能提供价格预测"),
            ("现在可以买入 CKB 吗？", "zh-CN", "不能提供价格预测"),
            ("CKB 该不该止盈？", "zh-CN", "不能提供价格预测"),
        ],
    )
    def test_explicit_financial_guidance_variants_are_blocked_before_llm(
        self,
        question,
        locale,
        expected_text,
    ):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": question}, locale=locale)

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json") as mock_call:
            out = info_gap_assessor(state)

        mock_call.assert_not_called()
        assert out["_route_decision"] == "answer_direct"
        assert out["retrieval_policy"] == "none"
        assert out["info_needs"] == []
        assert out["_financial_guidance_refusal"] is True
        assert expected_text in out["_final_response"]["text"]

    @pytest.mark.parametrize(
        "question",
        [
            "What does CKB stand for?",
            "What is the Common Knowledge Base?",
            "What is CKByte capacity?",
            "How do I buy CKB capacity in a transaction when creating cells?",
            "How much capacity does a Cell need?",
            "How do I transfer CKB with CCC SDK?",
            "Explain CKB tokenomics and Nervos DAO",
            "What are xUDT tokens on CKB?",
            "How are transaction fees calculated on CKB?",
        ],
    )
    def test_ckb_technical_and_neutral_questions_do_not_trigger_financial_guard(self, question):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": question}, locale="en")
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {
                        "kind": "concept_gap",
                        "question": question,
                        "required": False,
                    }
                ],
            }
        }
        calls = {"count": 0}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            calls["count"] += 1
            return _mock_call_llm_json_factory(mock_json_responses)(system_prompt, user_prompt, model=model)

        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            mock_call_llm_json,
        ):
            out = info_gap_assessor(state)

        assert calls["count"] >= 1
        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert out["info_needs"]
        assert not out.get("_financial_guidance_refusal", False)

    def test_technical_buy_word_does_not_trigger_financial_guard(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "How do I buy CKB capacity in a transaction when creating cells?"},
            locale="en",
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {
                        "kind": "concept_gap",
                        "question": "CKB transaction capacity cell creation",
                        "required": False,
                    }
                ],
            }
        }
        calls = {"count": 0}

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            calls["count"] += 1
            return _mock_call_llm_json_factory(mock_json_responses)(system_prompt, user_prompt, model=model)

        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            mock_call_llm_json,
        ):
            out = info_gap_assessor(state)

        assert calls["count"] >= 1
        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert not out.get("_financial_guidance_refusal", False)

    def test_technical_feedback_with_named_library_can_route_to_retrieval(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={
                "content": "你为什么不用 CCC 这个库提供教程，而是给了一个啥也不是的东西？"
            },
            recent_messages=[
                {"role": "user", "content": "我的技术栈是 TS/JS，想写 CKB 转账小应用。"},
                {"role": "assistant", "content": "这里是一个 TODO 骨架。"},
            ],
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {
                        "kind": "latest_spec",
                        "question": "CCC TypeScript CKB transfer tutorial and examples",
                        "required": False,
                    }
                ],
                "reasoning": "the correction names a public library and asks for evidence-backed rewrite",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert out["info_needs"][0]["required"] is False

    def test_technical_tutorial_retrieval_is_llm_prompt_decision(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={
                "content": "我是 TS/JS 小白，给我一个 CKB 转账最简可运行教程。"
            },
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {
                        "kind": "latest_spec",
                        "question": "TS/JS CKB transfer tutorial with real SDK or framework",
                        "required": False,
                    }
                ],
                "reasoning": "a runnable technical tutorial needs public docs and examples",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert out["info_needs"][0]["required"] is False
        assert out["budget"]["max_tool_calls"] >= 1

    def test_answer_direct_is_coerced_to_has_needs_when_force_retrieval(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor
        state = _make_state(
            user_message={"content": "什么是 ckb"},
            force_retrieval=True,
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "answer_direct",
                "info_needs": [],
                "reasoning": "simple question",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert isinstance(out["info_needs"], list)
        assert len(out["info_needs"]) >= 1

    def test_has_needs_single_policy_applies_light_budget(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "什么是 ckb"},
            budget={"max_tool_calls": 4, "max_hops": 3, "max_reflection_rounds_pre": 2},
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {"kind": "concept_gap", "question": "CKB definition", "required": False}
                ],
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["retrieval_policy"] == "single"
        assert out["budget"]["max_hops"] == 1
        assert out["budget"]["max_tool_calls"] == 2
        assert out["budget"]["max_reflection_rounds_pre"] == 1

    def test_real_object_evaluation_can_use_single_retrieval(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(
            user_message={"content": "Nervos Brain 这个项目你觉得如何"},
            budget={"max_tool_calls": 4, "max_hops": 3, "max_reflection_rounds_pre": 2},
        )
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [
                    {
                        "kind": "historical_consensus",
                        "question": "了解 Nervos Brain 的公开背景、计划和社区上下文后再评价",
                        "required": False,
                    }
                ],
                "reasoning": "评价真实项目需要外部上下文支撑，适合轻量检索。",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert out["budget"]["max_hops"] == 1
        assert out["budget"]["max_tool_calls"] == 2
        assert out["budget"]["max_reflection_rounds_pre"] == 1
        assert out["info_needs"][0]["required"] is False

    def test_has_needs_deep_policy_keeps_deeper_budget(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": "Fiber open_channel 报错，日志如下..."})
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "retrieval_policy": "deep",
                "info_needs": [
                    {"kind": "error_trace", "question": "排查 Fiber open_channel 报错", "required": False}
                ],
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["retrieval_policy"] == "deep"
        assert out["budget"]["max_hops"] >= 3
        assert out["budget"]["max_tool_calls"] >= 3

    def test_info_gap_preserves_llm_semantic_required_flags(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": "我的 open_channel 报错了，怎么修？"})
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "ask_user",
                "retrieval_policy": "none",
                "info_needs": [
                    {
                        "kind": "error_trace",
                        "question": "请贴完整报错日志、Fiber 版本和运行环境",
                        "required": True,
                    }
                ],
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "ask_user"
        assert out["retrieval_policy"] == "none"
        assert out["info_needs"][0]["required"] is True

    def test_info_gap_demotes_ask_user_without_required_info(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor

        state = _make_state(user_message={"content": "有没有比较靠谱的资料可以看？"})
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "ask_user",
                "retrieval_policy": "none",
                "info_needs": [
                    {
                        "kind": "latest_spec",
                        "question": "CKB 官方入门文档和社区推荐学习资料",
                        "required": False,
                    }
                ],
                "reasoning": "公开资料缺口，不应追问用户。",
            }
        }
        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert out["retrieval_policy"] == "single"
        assert out["info_needs"][0]["required"] is False


class TestFullGraphDebugState:
    """内部 debug 字段必须被 LangGraph state schema 保留。"""

    def test_timing_and_llm_trace_survive_graph_invoke(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
            _ = user_prompt, model
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "low-risk direct answer",
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048, reasoning_effort=None, verbosity=None, service_tier=None):
            _ = system_prompt, user_prompt, json_mode, model, temperature, max_tokens, reasoning_effort, verbosity, service_tier
            return "我是 Nervos Brain。"

        state = _make_state(
            user_message={"content": "你是谁"},
            _request_started_ts_ms=1,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert result.get("_request_started_ts_ms") == 1
        assert result.get("_node_timings")
        assert result["_node_timings"][0]["node"] == "info_gap_assessor"
        assert result.get("_llm_trace")
        assert result["_llm_trace"][0]["kind"] == "router_json"
        assert result.get("_llm_usage_summary", {}).get("calls", 0) >= 2

    def test_llm_service_tier_survives_graph_trace(self):
        from nervos_brain.graph_engine.full_graph import build_full_graph

        calls: list[dict[str, Any]] = []

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None, service_tier=None, **_kwargs):
            calls.append({"kind": "json", "service_tier": service_tier})
            _ = user_prompt, model
            if "模型档位路由器" in system_prompt:
                return {"tier": "low", "reasoning": "simple", "confidence": 0.9}
            if "信息缺口评估" in system_prompt:
                return {
                    "decision": "answer_direct",
                    "retrieval_policy": "none",
                    "info_needs": [],
                    "reasoning": "low-risk direct answer",
                }
            return {"decision": "accept_answer", "uncertainty_score": 0.1}

        def mock_call_llm(system_prompt, user_prompt, *, service_tier=None, **_kwargs):
            calls.append({"kind": "text", "service_tier": service_tier})
            _ = system_prompt, user_prompt
            return "我是 Nervos Brain。"

        state = _make_state(
            user_message={"content": "你是谁"},
            _llm_service_tier="priority",
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            result = build_full_graph().invoke(state)

        assert calls
        assert all(call["service_tier"] == "priority" for call in calls)
        assert result["_llm_trace"]
        assert all(row["service_tier"] == "priority" for row in result["_llm_trace"])


class TestPromptBoundaries:
    """关键 prompt 约束测试，防止后续回归到过度检索。"""

    def test_info_gap_prompt_preserves_core_deliverable_and_scope(self):
        from nervos_brain.graph_engine import prompts

        assert "retrieval_policy" in prompts.INFO_GAP_SYSTEM
        assert "对象、用户要完成的动作、期望交付物和显式范围" in prompts.INFO_GAP_SYSTEM
        assert "只提高查证强度，不自动增加" in prompts.INFO_GAP_SYSTEM
        assert "每个 info_need 都必须能说明它如何帮助完成核心交付物" in prompts.INFO_GAP_SYSTEM
        assert "对象说明是检索上下文" in prompts.INFO_GAP_SYSTEM
        assert "不自动产生“重新鉴定对象身份”的调查维度" in prompts.INFO_GAP_SYSTEM
        assert "公开身份、公开版本、公开文档、公开渠道" in prompts.INFO_GAP_SYSTEM
        assert "证据不足或来源冲突都不等于缺少用户私有信息" in prompts.INFO_GAP_SYSTEM
        assert "不得自行加入 source_preference" in prompts.INFO_GAP_SYSTEM
        assert "大多数单一事实、入口、资料、用法和当前状态问题使用 single" in prompts.INFO_GAP_SYSTEM
        assert "步骤、参数、条件和限制属于同一事实簇" in prompts.INFO_GAP_SYSTEM
        assert "默认生成一个综合 info_need" in prompts.INFO_GAP_SYSTEM
        assert "预估可能需要多条证据不等于多个独立交付物" in prompts.INFO_GAP_SYSTEM
        assert "从 info_needs 中删除对象身份、发行主体、术语定义和历史背景子句" in prompts.INFO_GAP_SYSTEM
        assert "用户已经提供的对象说明默认用于消歧" in prompts.INFO_GAP_USER
        assert "不要按实体名、领域关键词或用户要求“详细”机械升档" in prompts.INFO_GAP_SYSTEM
        assert "旧答案和背景不是新的任务清单" in prompts.INFO_GAP_SYSTEM
        assert "主动检索不等于多轮深检索" in prompts.INFO_GAP_SYSTEM

    def test_retriever_planner_prompt_prioritizes_answer_bearing_evidence(self):
        from nervos_brain.graph_engine import prompts

        assert "直接填入最终答案的事实" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "核心对象 + 用户原始动作 + 期望结果" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不扩大核心交付物的范围" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "已有证据足以完成核心交付物时停止" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "默认不加 source filter" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "证据质量，不是资料库或来源类型" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "filters 必须为 `{}`" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不得根据期望的权威性推导 source filter" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "官方渠道、官方入口、官方说明" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "模型自行生成的来源偏好也不能授权 source filter" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "输出前硬性检查" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不要使用 info_needs、rationale 或你自己的来源偏好" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "只有用户原始问题明确限定资料库或来源类型" in prompts.RETRIEVER_PLANNER_USER
        assert "一到两个检索所需的动作同义词" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不同领域术语" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不要只是重复翻译或改写用户原始动词" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "属于召回后的证据判断" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不要把这类评价词或答案字段加入第一轮 query" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "4 到 8 个独立检索词" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "交付格式词、参数字段和重复同义词" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不得把对象身份、发行主体、定义或历史背景加入第一轮 query" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "删除与核心交付物无关的身份、发行主体、定义、历史" in prompts.RETRIEVER_PLANNER_USER
        assert "最多两个非重复动作同义词" in prompts.RETRIEVER_PLANNER_USER
        assert "query 不是答案字段清单" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "以免稀释对象、动作和结果" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "提高单个步骤的 top_k 以覆盖细节记录" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "默认只为核心对象保留一个最有区分度的 regex_query" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不要为动作、期望结果、常见类别词" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "retrieval_policy=\"single\"" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "{retrieval_policy}" in prompts.RETRIEVER_PLANNER_USER
        assert "默认只生成一个统一 qdrant_search step" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "不复制整段对话、纠错语气或内部评测描述" in prompts.RETRIEVER_PLANNER_SYSTEM

    def test_reflection_prompt_treats_buried_answer_as_functional_failure(self):
        from nervos_brain.graph_engine import prompts

        assert "第一优先级是任务完成度" in prompts.REFLECTION_SYSTEM
        assert "首段没有直接完成用户请求" in prompts.REFLECTION_SYSTEM
        assert "关键结论被背景埋没" in prompts.REFLECTION_SYSTEM
        assert "属于范围漂移，不是单纯“可以更精炼”" in prompts.REFLECTION_SYSTEM
        assert "已有直接证据时应 accept_answer" in prompts.REFLECTION_SYSTEM
        assert "不要为了更多背景、来源数量或边际完整性继续检索" in prompts.REFLECTION_SYSTEM
        assert "去掉来源限制、补充动作同义词" in prompts.REFLECTION_SYSTEM
        assert "只在无关主题中偶然命中关键词的记录不是核心证据" in prompts.REFLECTION_SYSTEM
        assert "不得要求用户提供公开身份、官网、文档或链接" in prompts.REFLECTION_SYSTEM
        assert "只道歉或承诺改进" in prompts.REFLECTION_SYSTEM
        assert "direct answer 可以没有 citations" in prompts.REFLECTION_SYSTEM

    def test_model_router_prompt_uses_low_medium_and_high(self):
        from nervos_brain.graph_engine import full_nodes

        assert "不要过度省模型，也不要过度升档" in full_nodes._LLM_ROUTER_SYSTEM
        assert "low、medium、high 三档之一" in full_nodes._LLM_ROUTER_SYSTEM
        assert "不按领域名称、实体名称或用户要求“详细”机械升档" in full_nodes._LLM_ROUTER_SYSTEM
        assert "证据被错误泛化" in full_nodes._LLM_ROUTER_SYSTEM
        assert "证据数量多但核心答案简单时不应升档" in full_nodes._LLM_ROUTER_SYSTEM
        assert "主动选择更快档位" in full_nodes._LLM_ROUTER_SYSTEM
        assert full_nodes._MODEL_TIERS == {"low", "medium", "high"}
        assert full_nodes._NODE_FALLBACK_TIERS["info_gap_assessor"] == "low"
        assert full_nodes._NODE_FALLBACK_TIERS["retriever_planner"] == "low"
        assert full_nodes._NODE_FALLBACK_TIERS["reflection_pre"] == "low"
        assert full_nodes._NODE_FALLBACK_TIERS["reflection_post"] == "medium"
        assert full_nodes._NODE_FALLBACK_TIERS["direct_answer"] == "low"

    def test_answer_composer_prompt_leads_with_deliverable_and_limits_scope(self):
        from nervos_brain.graph_engine import prompts

        assert "第一段直接交付用户要的结果" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "再按对核心交付物的帮助程度" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "不意味着输出所有检索结果" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "未用于完成核心交付物的证据不写入正文" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "先短后详，但不使用僵硬的固定模板" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "禁止重复结论、重复总结、装饰性章节" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "不要用与核心交付物无关的证据、通用风险清单" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "不得把公开可检索的对象身份、官网、文档或链接" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "只偶然出现关键词的材料不得作为核心证据" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "直接给出修正后的实际答案" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "{{cite:E1}}" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "只引用正文实际使用的证据" in prompts.ANSWER_COMPOSER_SYSTEM

    @pytest.mark.parametrize(
        ("task_kind", "contract_fragment"),
        [
            ("action_or_channel", "可执行结果或入口"),
            ("api_or_command", "正确的接口或命令"),
            ("comparison", "比较结论或决策依据"),
            ("troubleshooting", "最可能原因和下一步"),
            ("explicit_full_investigation", "全面调查或完整报告"),
        ],
    )
    def test_answer_composer_generic_task_matrix(self, task_kind, contract_fragment):
        from nervos_brain.graph_engine import prompts

        assert task_kind
        assert contract_fragment in prompts.ANSWER_COMPOSER_SYSTEM
        assert "不要套用固定模板" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "仍应在开头交付最重要的结果" in prompts.ANSWER_COMPOSER_SYSTEM

    def test_direct_answer_prompt_reanswers_corrections_instead_of_promising(self):
        from nervos_brain.graph_engine import prompts

        assert "不追加参考来源" in prompts.DIRECT_ANSWER_SYSTEM
        assert "立即给出修正后的答案" in prompts.DIRECT_ANSWER_SYSTEM
        assert "不要只回复“明白、以后会注意”" in prompts.DIRECT_ANSWER_SYSTEM
        assert "不要因为用户要求详细而扩展到未请求的主题" in prompts.DIRECT_ANSWER_SYSTEM
        assert "不得编造" in prompts.DIRECT_ANSWER_SYSTEM

    def test_runtime_prompts_do_not_embed_regression_specific_rules(self):
        from nervos_brain.graph_engine import full_nodes, prompts

        runtime_prompts = "\n".join(
            [
                prompts.INFO_GAP_SYSTEM,
                prompts.RETRIEVER_PLANNER_SYSTEM,
                prompts.REFLECTION_SYSTEM,
                prompts.ANSWER_COMPOSER_SYSTEM,
                prompts.DIRECT_ANSWER_SYSTEM,
                full_nodes._LLM_ROUTER_SYSTEM,
            ]
        )
        forbidden = (
            "USDI",
            "DestBridge",
            "Fiber",
            "CCC",
            "open_channel",
            "TS/JS",
            "Go SDK",
            "Spore",
            "RGB++",
        )
        assert all(term not in runtime_prompts for term in forbidden)

    def test_financial_guidance_prompts_refuse_price_predictions(self):
        from nervos_brain.graph_engine import prompts

        assert "价格预测" in prompts.INFO_GAP_SYSTEM
        assert "目标价" in prompts.INFO_GAP_SYSTEM
        assert "买入、卖出、持仓" in prompts.INFO_GAP_SYSTEM
        assert "中立事实、技术风险或公开数据来源" in prompts.INFO_GAP_SYSTEM
        assert "不要为价格预测、目标价或交易决策生成检索计划" in prompts.RETRIEVER_PLANNER_SYSTEM
        assert "金融安全边界必须保持" in prompts.REFLECTION_SYSTEM
        assert "免责声明不能绕过该限制" in prompts.ANSWER_COMPOSER_SYSTEM
        assert "免责声明不能绕过该限制" in prompts.DIRECT_ANSWER_SYSTEM


class TestFinancialGuidanceGuard:
    def test_format_repair_sanitizes_generated_price_prediction(self):
        from nervos_brain.graph_engine.full_nodes import format_repair

        state = _make_state(
            user_message={"content": "give me the best price prediction you can for ckb"},
            locale="en",
            _route_decision="has_needs",
            _final_response={
                "request_id": "test-req-001",
                "text": "Best guess, not financial advice: $0.03-$0.05 in a strong cycle.",
                "citations": [],
            },
        )

        out = format_repair(state)

        text = out["_final_response"]["text"]
        assert "can't provide price predictions" in text
        assert "$0.03" not in text

    def test_format_repair_does_not_sanitize_neutral_tokenomics_answer(self):
        from nervos_brain.graph_engine.full_nodes import format_repair

        state = _make_state(
            user_message={"content": "Explain CKB tokenomics and Nervos DAO"},
            locale="en",
            _route_decision="has_needs",
            _final_response={
                "request_id": "test-req-001",
                "text": "CKByte is used for capacity, fees, and DAO deposits.",
                "citations": [],
            },
        )

        out = format_repair(state)

        text = out["_final_response"]["text"]
        assert "CKByte is used for capacity" in text
        assert "can't provide price predictions" not in text


class TestProviderRegistry:
    """ProviderCapabilityRegistry 测试。"""

    def test_get_model_for_planning(self):
        from nervos_brain.graph_engine.provider_registry import ProviderCapabilityRegistry
        reg = ProviderCapabilityRegistry()
        model = reg.get_model_for("planning", max_cost="low")
        assert model == "gpt-4o-mini"

    def test_get_model_for_composing(self):
        from nervos_brain.graph_engine.provider_registry import ProviderCapabilityRegistry
        reg = ProviderCapabilityRegistry()
        model = reg.get_model_for("composing")
        assert model in ("gpt-4o", "claude-3-5-sonnet-20241022")

    def test_get_profile_for_uses_default_router_tiers(self):
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry
        reg = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "medium", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )

        router = reg.get_profile_for("general", tier="router", require_json=True)
        low = reg.get_profile_for("planning", tier="low", require_json=True)
        medium = reg.get_profile_for("reflection", tier="medium", require_json=True)
        high = reg.get_profile_for("composing", tier="high")
        unknown = reg.get_profile_for("general", tier="unknown")

        assert router["model"] == "openai/gpt-5.4-mini"
        assert router["reasoning_effort"] == "low"
        assert low["model"] == "openai/gpt-5.4-mini"
        assert low["reasoning_effort"] == "low"
        assert medium["model"] == "openai/gpt-5.5"
        assert medium["reasoning_effort"] == "medium"
        assert high["model"] == "openai/gpt-5.5"
        assert high["reasoning_effort"] == "high"
        assert unknown["tier"] == "low"

    def test_provider_registry_legacy_mini_high_request_falls_back_to_low(self):
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry
        reg = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "low", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )

        legacy = reg.get_profile_for("planning", tier="mini_high", require_json=True)

        assert legacy["tier"] == "low"
        assert legacy["model"] == "openai/gpt-5.4-mini"
        assert legacy["reasoning_effort"] == "low"


class TestRuntimeInjection:
    """验证 runtime 注入依赖在图执行中可用。"""

    def test_invoke_full_graph_uses_injected_multi_retriever(self):
        from nervos_brain.graph_engine.full_graph import (
            FullGraphRuntime,
            build_full_graph,
            invoke_full_graph,
        )

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, query: str, filters=None, top_k: int = 5):
                _ = query, filters, top_k
                self.calls += 1
                return [
                    {
                        "id": "ev-1",
                        "source": "qdrant",
                        "title": "CKB intro",
                        "url": "https://example.com/ckb",
                        "anchor": "doc:1",
                        "snippet": "CKB is the layer-1 of Nervos.",
                        "score": 0.9,
                        "payload": {"source": "github_docs", "version": "v1"},
                        "hash": "h1",
                        "retrieved_ts_ms": 1,
                    }
                ]

        retriever = FakeRetriever()
        runtime = FullGraphRuntime(multi_retriever=retriever)
        state = _make_state(user_message={"content": "什么是 ckb"}, force_retrieval=True)

        mock_llm_responses = {
            "回答组装": "CKB 是 Nervos 的一层网络 {{cite:E1}}。",
        }
        mock_json_responses = {
            "信息缺口评估": {
                "decision": "has_needs",
                "info_needs": [{"kind": "concept_gap", "question": "定义", "required": False}],
            },
            "检索规划": {
                "plan_id": "p1",
                "rationale": "retrieve",
                "steps": [{"step_id": "s1", "tool": "qdrant_search", "query": "什么是 ckb", "filters": {}, "top_k": 3}],
                "parallel_groups": [["s1"]],
                "budget": {"max_tool_calls": 1},
            },
            "证据评分": {"grade": "enough", "reasoning": "enough"},
            "自检": {"pass": True, "issues": [], "reasoning": "ok"},
        }

        with patch(
            "nervos_brain.graph_engine.full_nodes.call_llm",
            _mock_call_llm_factory(mock_llm_responses),
        ), patch(
            "nervos_brain.graph_engine.full_nodes.call_llm_json",
            _mock_call_llm_json_factory(mock_json_responses),
        ):
            graph = build_full_graph()
            out = invoke_full_graph(state, runtime=runtime, compiled_graph=graph)

        assert retriever.calls >= 1
        assert len(out.get("evidence", [])) >= 1
        assert out.get("_final_response", {}).get("text")


class TestNodeModelRouter:
    """节点级模型 router 测试。"""

    def test_info_gap_uses_router_selected_low_profile(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry

        registry = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "medium", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )
        calls: list[dict] = []

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None, reasoning_effort=None, verbosity=None, max_tokens=None):
            calls.append(
                {
                    "system": system_prompt,
                    "user": user_prompt,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                    "max_tokens": max_tokens,
                }
            )
            if "模型档位路由器" in system_prompt:
                assert '"allowed_tiers": ["low", "medium", "high"]' in user_prompt
                return {"tier": "low", "reasoning": "technical planning", "confidence": 0.88}
            return {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [{"kind": "latest_spec", "question": "查 Fiber API", "required": False}],
                "reasoning": "needs public retrieval",
            }

        state = _make_state(
            user_message={"content": "Fiber WASM 怎么存数据？"},
            _provider_registry=registry,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert calls[0]["model"] == "openai/gpt-5.4-mini"
        assert calls[0]["reasoning_effort"] == "low"
        assert calls[1]["model"] == "openai/gpt-5.4-mini"
        assert calls[1]["reasoning_effort"] == "low"
        assert out["_llm_trace"][0]["selected_tier"] == "low"
        assert out["_llm_trace"][1]["tier"] == "low"

    def test_info_gap_router_failure_falls_back_to_low(self):
        from nervos_brain.graph_engine.full_nodes import info_gap_assessor
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry

        registry = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "medium", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )
        business_calls: list[dict] = []

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None, reasoning_effort=None, verbosity=None, max_tokens=None):
            _ = user_prompt, verbosity, max_tokens
            if "模型档位路由器" in system_prompt:
                raise RuntimeError("router down")
            business_calls.append({"model": model, "reasoning_effort": reasoning_effort})
            return {
                "decision": "has_needs",
                "retrieval_policy": "single",
                "info_needs": [{"kind": "latest_spec", "question": "查 Fiber API", "required": False}],
            }

        state = _make_state(
            user_message={"content": "Fiber WASM 怎么存数据？"},
            _provider_registry=registry,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
            out = info_gap_assessor(state)

        assert out["_route_decision"] == "has_needs"
        assert business_calls == [{"model": "openai/gpt-5.4-mini", "reasoning_effort": "low"}]
        assert out["_llm_trace"][0]["selected_tier"] == "low"
        assert out["_llm_trace"][1]["tier"] == "low"

    def test_answer_composer_uses_router_selected_high_profile(self):
        from nervos_brain.graph_engine.full_nodes import answer_composer
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry

        registry = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "low", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )
        calls: list[dict] = []

        def mock_call_llm_json(system_prompt, user_prompt, *, model=None, reasoning_effort=None, verbosity=None, max_tokens=None):
            calls.append(
                {
                    "kind": "router",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                    "max_tokens": max_tokens,
                }
            )
            assert "模型档位路由器" in system_prompt
            return {"tier": "high", "reasoning": "complex code answer", "confidence": 0.9}

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048, reasoning_effort=None, verbosity=None, disable_response_storage=None):
            _ = system_prompt, user_prompt, json_mode, temperature, disable_response_storage
            calls.append(
                {
                    "kind": "business",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                    "max_tokens": max_tokens,
                }
            )
            return "这里是完整 TypeScript 代码 {{cite:E1}}。"

        state = _make_state(
            user_message={"content": "写完整 TS 交易调用代码"},
            evidence=[
                {
                    "id": "ev-1",
                    "source": "github",
                    "title": "CCC transfer",
                    "url": "https://example.com/ccc",
                    "anchor": "a1",
                    "snippet": "send transaction",
                    "score": 0.9,
                    "payload": {},
                    "hash": "h1",
                    "retrieved_ts_ms": 1,
                }
            ],
            _provider_registry=registry,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm):
            out = answer_composer(state)

        assert out["_final_response"]["text"]
        assert calls[0] == {
            "kind": "router",
            "model": "openai/gpt-5.4-mini",
            "reasoning_effort": "low",
            "verbosity": "low",
            "max_tokens": 512,
        }
        assert calls[1] == {
            "kind": "business",
            "model": "openai/gpt-5.5",
            "reasoning_effort": "high",
            "verbosity": "low",
            "max_tokens": 4096,
        }

    def test_router_failure_falls_back_without_breaking_direct_answer(self):
        from nervos_brain.graph_engine.full_nodes import answer_composer
        from nervos_brain.graph_engine.provider_registry import ModelProfile, ProviderCapabilityRegistry

        registry = ProviderCapabilityRegistry(
            profiles={
                "router": ModelProfile("router", "openai/gpt-5.4-mini", "low", "low", 512),
                "low": ModelProfile("low", "openai/gpt-5.4-mini", "low", "low", 2048),
                "medium": ModelProfile("medium", "openai/gpt-5.5", "low", "low", 2048),
                "high": ModelProfile("high", "openai/gpt-5.5", "high", "low", 4096),
            }
        )
        business_calls: list[dict] = []

        def mock_call_llm_json(*args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("router down")

        def mock_call_llm(system_prompt, user_prompt, *, json_mode=False, model=None, temperature=0.3, max_tokens=2048, reasoning_effort=None, verbosity=None, disable_response_storage=None):
            _ = system_prompt, user_prompt, json_mode, temperature, disable_response_storage
            business_calls.append(
                {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                    "max_tokens": max_tokens,
                }
            )
            return "我是 Nervos Brain。"

        state = _make_state(
            user_message={"content": "你是谁"},
            retrieval_policy="none",
            _route_decision="answer_direct",
            _provider_registry=registry,
        )

        with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json), \
             patch("nervos_brain.graph_engine.full_nodes.call_llm", mock_call_llm):
            out = answer_composer(state)

        assert out["_final_response"]["text"] == "我是 Nervos Brain。"
        assert business_calls == [
            {
                "model": "openai/gpt-5.4-mini",
                "reasoning_effort": "low",
                "verbosity": "low",
                "max_tokens": 2048,
            }
        ]
