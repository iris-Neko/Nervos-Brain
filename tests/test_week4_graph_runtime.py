from __future__ import annotations

from unittest.mock import patch

from nervos_brain.graph_engine.full_nodes import ask_user, turn_interpreter, retrieval_executor
from nervos_brain.retrieval import ArchiveRecord, ArchiveStore, RetrievalConfig


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None, int | None]] = []

    def search(self, query: str, filters=None, top_k=None):
        self.calls.append((query, filters, top_k))
        return [
            {
                "id": "ev-1",
                "source": "qdrant",
                "title": "Fiber Guide",
                "url": "https://example.com/fiber",
                "anchor": "doc:fiber#chunk:0",
                "snippet": "openChannel with peerId and capacity",
                "score": 0.9,
                "payload": {"source": "fiber", "version": "0.3"},
                "hash": "hash1",
                "retrieved_ts_ms": 1,
            }
        ]


class _FakeMemoryService:
    def __init__(self) -> None:
        self.suspended: list[dict] = []
        self.completed: list[dict] = []
        self._checkpoint = {
            "checkpoint_id": "ck-1",
            "missing_params": ["sdk_language"],
            "resume_node": "retriever_planner",
            "context_payload": {
                "origin_question": "写一个交易记账app示例吧",
                "ask_user_question": "你想用什么技术栈？",
            },
            "expires_ts_ms": 9_999_999_999_999,
            "version": 1,
        }

    def list_user_facts(self, *, key):
        return [
            {
                "id": "fact-u-1",
                "namespace": "user",
                "key": "default_sdk",
                "value": "js",
                "confidence": 0.9,
                "updated_ts_ms": 1000,
                "source_event_ids": ["ev-1"],
            }
        ]

    def list_channel_facts(self, *, key):
        return [
            {
                "id": "fact-c-1",
                "namespace": "channel",
                "key": "fiber_version",
                "value": "0.3",
                "confidence": 0.95,
                "updated_ts_ms": 1200,
                "source_event_ids": ["ev-2"],
            }
        ]

    def suspend_thread(
        self,
        *,
        key,
        missing_params,
        resume_node,
        context_payload=None,
        ttl_hours=24,
        now_ts_ms=None,
    ):
        self.suspended.append(
            {
                "key": key,
                "missing_params": list(missing_params),
                "resume_node": resume_node,
                "context_payload": dict(context_payload or {}),
            }
        )
        return "checkpoint-123"

    def resume_thread(self, *, key, now_ts_ms=None):
        self.last_resume_key = key
        return self._checkpoint

    def complete_thread(self, *, key, now_ts_ms=None):
        self.completed.append({"key": key})
        return True


def test_retrieval_executor_uses_multi_retriever():
    retriever = _FakeRetriever()
    state = {
        "request_id": "r1",
        "retrieval_plan": {
            "steps": [
                {"step_id": "s1", "tool": "qdrant_search", "query": "open channel", "filters": {"topic": "fiber"}, "top_k": 3}
            ]
        },
        "evidence": [],
        "budget": {"max_tool_calls": 2, "max_evidence_chunks": 5},
        "_multi_retriever": retriever,
    }

    result = retrieval_executor(state)
    assert len(result["evidence"]) == 1
    assert retriever.calls[0][0] == "open channel"


def test_retrieval_executor_supports_discourse_and_github_archive_search(tmp_path):
    cfg = RetrievalConfig(archive_db=str(tmp_path / "archive.db"))
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    archive.upsert(
        ArchiveRecord(
            id="disc-1",
            source="nervos_talk",
            doc_type="forum_post",
            url="https://talk.nervos.org/t/fiber/1/2",
            anchor="doc:nervos-talk-1#post:2",
            title="Fiber timeout discussion",
            summary="Discussion around HTLC timeout.",
            keywords="fiber,htlc,timeout",
            raw_text="The forum thread discusses HTLC timeout handling for Fiber channels.",
            raw_format="html",
            lang="en",
            version="latest",
            topic="fiber",
            content_hash="hash-disc-1",
        )
    )
    archive.upsert(
        ArchiveRecord(
            id="gh-1",
            source="github_docs",
            doc_type="github_doc",
            url="https://github.com/nervosnetwork/fiber/blob/main/docs/channel.md",
            anchor="doc:github-fiber#blob:1",
            title="nervosnetwork/fiber/docs/channel.md",
            summary="openChannel example in docs.",
            keywords="fiber,openChannel,docs",
            raw_text="Use openChannel with peerId and capacity when opening a Fiber channel.",
            raw_format="markdown",
            lang="en",
            version="main",
            topic="nervosnetwork/fiber",
            content_hash="hash-gh-1",
        )
    )

    state = {
        "request_id": "r-archive",
        "retrieval_plan": {
            "steps": [
                {"step_id": "s1", "tool": "discourse_query", "query": "HTLC timeout", "filters": {"topic": "fiber"}, "top_k": 3},
                {"step_id": "s2", "tool": "github_search", "query": "openChannel", "filters": {"repo": "nervosnetwork/fiber"}, "top_k": 3},
            ]
        },
        "evidence": [],
        "budget": {"max_tool_calls": 3, "max_evidence_chunks": 5},
        "_archive_store": archive,
    }

    result = retrieval_executor(state)
    assert len(result["evidence"]) == 2
    assert {row["source"] for row in result["evidence"]} == {"discourse", "github"}
    assert "tools=" in result["_tool_execution_summary"]


def test_ask_user_suspends_thread_checkpoint():
    memory = _FakeMemoryService()
    state = {
        "request_id": "r2",
        "info_needs": [
            {
                "kind": "missing_param",
                "question": "请补充 SDK 语言",
                "required": True,
                "availability": "user_owned",
                "hints": {"sdk_language": ""},
            }
        ],
        "user_message": {
            "context": {
                "platform": "discord",
                "guild_id": "g1",
                "channel_id": "c1",
                "thread_id": "t1",
                "user_id": "u1",
            }
        },
        "_memory_service": memory,
    }

    result = ask_user(state)
    response = result["_final_response"]
    assert response["need_user_input"] is True
    assert memory.suspended, "AskUser 应写入线程 checkpoint"
    assert memory.suspended[0]["key"]["thread_id"] == "t1:user:u1"
    assert response.get("trace_summary", "").startswith("thread_checkpoint=")


def test_turn_interpreter_resumes_checkpoint():
    memory = _FakeMemoryService()
    state = {
        "request_id": "r3",
        "user_message": {
            "content": "我用的是 js sdk 0.3",
            "context": {
                "platform": "discord",
                "guild_id": "g1",
                "channel_id": "c1",
                "thread_id": "t1",
                "user_id": "u1",
            },
        },
        "memory_facts": [],
        "evidence": [],
        "budget": {"max_memory_facts": 5},
        "_memory_service": memory,
    }

    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "route": "retrieve",
            "turn_relation": "clarification_answer",
            "resolved_request": "写一个交易记账app示例吧；用户补充使用 js sdk 0.3。",
            "retrieval_policy": "single",
            "info_needs": [
                {
                    "kind": "public_docs",
                    "question": "Retrieve the public SDK documentation for the requested example.",
                    "required": False,
                    "availability": "public",
                }
            ],
        },
    ):
        result = turn_interpreter(state)

    assert result["_route_decision"] == "has_needs"
    assert "写一个交易记账app示例吧" in result.get("resolved_question", "")
    assert "js sdk 0.3" in result.get("resolved_question", "")
    assert len(result["memory_facts"]) >= 1
    assert memory.last_resume_key["thread_id"] == "t1:user:u1"
    assert memory.completed, "恢复路径应尝试完成 checkpoint"


def test_turn_interpreter_resume_drops_required_missing_param_from_llm():
    memory = _FakeMemoryService()
    state = {
        "request_id": "r-resume-ts",
        "user_message": {
            "content": "ts调用脚本",
            "context": {
                "platform": "telegram",
                "guild_id": "-100123",
                "channel_id": "-100123",
                "user_id": "42",
            },
        },
        "memory_facts": [],
        "evidence": [],
        "budget": {"max_memory_facts": 5},
        "_memory_service": memory,
    }

    with patch(
        "nervos_brain.graph_engine.full_nodes.call_llm_json",
        return_value={
            "route": "retrieve",
            "turn_relation": "clarification_answer",
            "resolved_request": "继续处理 ts 调用脚本；用户已说明当前环境和目标。",
            "retrieval_policy": "single",
            "info_needs": [
                {
                    "kind": "public_docs",
                    "question": "Retrieve public documentation relevant to the script call.",
                    "required": False,
                    "availability": "public",
                }
            ],
        },
    ):
        result = turn_interpreter(state)

    assert result["_route_decision"] == "has_needs"
    assert result["retrieval_policy"] == "single"
    assert all(not need.get("required") for need in result["info_needs"])
    assert "ts" in result["resolved_question"]
    assert memory.completed


def test_turn_interpreter_discards_stale_checkpoint_for_new_user_question():
    memory = _FakeMemoryService()
    state = {
        "request_id": "r-stale-checkpoint",
        "user_message": {
            "content": "ckb是什么",
            "context": {
                "platform": "telegram",
                "guild_id": "-100123",
                "channel_id": "-100123",
                "user_id": "42",
            },
        },
        "memory_facts": [],
        "evidence": [],
        "budget": {"max_memory_facts": 5},
        "_memory_service": memory,
    }

    captured: dict[str, str] = {}

    def mock_call_llm_json(system_prompt, user_prompt, *, model=None):
        _ = system_prompt, model
        captured["user_prompt"] = user_prompt
        return {
            "decision": "answer_direct",
            "retrieval_policy": "none",
            "info_needs": [],
        }

    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", mock_call_llm_json):
        result = turn_interpreter(state)

    assert result["_route_decision"] == "answer_direct"
    assert result["resolved_question"] == "ckb是什么"
    assert "写一个交易记账app示例吧" not in result["resolved_question"]
    assert "Thread resume state" not in captured["user_prompt"]
    assert memory.completed, "新问题应丢弃旧 pending checkpoint"


def test_thread_checkpoint_uses_default_group_thread_and_user_id():
    memory = _FakeMemoryService()
    state = {
        "request_id": "r-default-thread",
        "info_needs": [
            {
                "kind": "missing_param",
                "question": "请补充 SDK 语言",
                "required": True,
                "availability": "user_owned",
            }
        ],
        "user_message": {
            "context": {
                "platform": "telegram",
                "guild_id": "-100123",
                "channel_id": "-100123",
                "user_id": "42",
            }
        },
        "_memory_service": memory,
    }

    ask_user(state)

    assert memory.suspended[0]["key"] == {
        "platform": "telegram",
        "guild_id": "-100123",
        "channel_id": "-100123",
        "thread_id": "__default__:user:42",
    }
