"""M6-T12/T13/T14: ToolRuntime contract、超时、幂等测试。"""

import asyncio
import time

import pytest

from nervos_brain.retrieval import ArchiveRecord, ArchiveStore, RetrievalConfig
from nervos_brain.tool_runtime import (
    build_idempotency_key,
    build_tool_call_request,
    check_idempotency,
    execute_tool,
    handle_discourse_query,
    handle_github_search,
    reset_idempotency_cache,
    validate_tool_args,
    select_transport,
    MockTransportAdapter,
)


class TestToolWhitelistAndSchema:
    """M6-T12: 白名单拒绝非法 tool / 非法 args。"""

    def test_reject_unknown_tool(self):
        errors = validate_tool_args("hack_the_planet", {})
        assert any("whitelist" in e for e in errors)

    def test_reject_missing_required_arg(self):
        errors = validate_tool_args("qdrant_search", {"query": "test"})
        assert any("filters" in e for e in errors)

    def test_accept_valid_qdrant_search(self):
        errors = validate_tool_args(
            "qdrant_search",
            {"query": "cell model", "filters": {"source": "rfcs"}, "top_k": 5},
        )
        assert errors == []

    def test_reject_top_k_out_of_range(self):
        errors = validate_tool_args(
            "qdrant_search",
            {"query": "test", "filters": {}, "top_k": 999},
        )
        assert any("maximum" in e for e in errors)

    def test_reject_query_too_long(self):
        errors = validate_tool_args(
            "qdrant_search",
            {"query": "x" * 600, "filters": {}},
        )
        assert any("maxLength" in e for e in errors)

    def test_reject_unexpected_arg(self):
        errors = validate_tool_args(
            "qdrant_search",
            {"query": "test", "filters": {}, "evil_param": "bad"},
        )
        assert any("unexpected" in e for e in errors)

    def test_ignore_internal_runtime_args(self):
        errors = validate_tool_args(
            "github_search",
            {"query": "fiber docs", "top_k": 3, "_transport": object()},
        )
        assert errors == []

    def test_build_request_rejects_bad_tool(self):
        with pytest.raises(ValueError, match="validation failed"):
            build_tool_call_request(
                request_id="r1",
                step_id="s1",
                tool="invalid",
                args={},
            )

    def test_build_request_ok(self):
        req = build_tool_call_request(
            request_id="r1",
            step_id="s1",
            tool="qdrant_search",
            args={"query": "cell", "filters": {"source": "rfcs"}},
        )
        assert req["tool"] == "qdrant_search"
        assert req["deadline_ts_ms"] > req["issued_ts_ms"]
        assert req["timeout_ms"] == 60_000


class TestIdempotencyKey:
    """M6-T5: idempotency_key 稳定性。"""

    def test_same_input_same_key(self):
        k1 = build_idempotency_key("qdrant_search", {"query": "a"}, "s1")
        k2 = build_idempotency_key("qdrant_search", {"query": "a"}, "s1")
        assert k1 == k2

    def test_different_input_different_key(self):
        k1 = build_idempotency_key("qdrant_search", {"query": "a"}, "s1")
        k2 = build_idempotency_key("qdrant_search", {"query": "b"}, "s1")
        assert k1 != k2

    def test_internal_runtime_args_do_not_affect_key(self):
        k1 = build_idempotency_key("github_search", {"query": "fiber", "_transport": object()}, "s1")
        k2 = build_idempotency_key("github_search", {"query": "fiber"}, "s1")
        assert k1 == k2


class TestTimeout:
    """M6-T13: 超时后结果被丢弃。"""

    def test_timeout_returns_cancelled(self):
        async def slow_handler(req):
            await asyncio.sleep(5)
            return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}

        req = build_tool_call_request(
            request_id="r1",
            step_id="s1",
            tool="qdrant_search",
            args={"query": "cell", "filters": {"source": "rfcs"}},
            timeout_ms=100,
        )

        result = asyncio.run(execute_tool(req, slow_handler))
        assert result["status"] == "cancelled"
        assert result["ok"] is False

    def test_deadline_already_passed(self):
        async def fast_handler(req):
            return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}

        req = build_tool_call_request(
            request_id="r1",
            step_id="s1",
            tool="qdrant_search",
            args={"query": "cell", "filters": {"source": "rfcs"}},
            timeout_ms=10_000,
        )
        req["deadline_ts_ms"] = int(time.time() * 1000) - 1000

        result = asyncio.run(execute_tool(req, fast_handler))
        assert result["status"] == "cancelled"

    def test_handler_exception_returns_error(self):
        async def boom_handler(req):
            _ = req
            raise RuntimeError("boom")

        req = build_tool_call_request(
            request_id="r1",
            step_id="s1",
            tool="github_search",
            args={"query": "fiber"},
            timeout_ms=1_000,
        )

        result = asyncio.run(execute_tool(req, boom_handler))
        assert result["status"] == "error"
        assert result["error"]["code"] == "ERR_TOOL_EXECUTION_FAILED"


class TestIdempotencyDedup:
    """M6-T14: 同 idempotency_key 去重。"""

    def setup_method(self):
        reset_idempotency_cache()

    def test_first_call_not_duplicate(self):
        assert check_idempotency("key_abc") is False

    def test_second_call_is_duplicate(self):
        check_idempotency("key_xyz")
        assert check_idempotency("key_xyz") is True

    def test_different_keys_not_duplicate(self):
        check_idempotency("key_1")
        assert check_idempotency("key_2") is False


class TestTransport:
    """M6-T9/T10/T11: Transport 抽象测试。"""

    def test_mock_transport(self):
        adapter = MockTransportAdapter()
        result = adapter.send({"tool": "test"})
        assert result["data"]["mock"] is True

    def test_select_transport_finds_sse(self):
        proto = select_transport(available={"sse"})
        assert proto == "sse"

    def test_select_transport_prefers_streamable(self):
        proto = select_transport(available={"sse", "streamable_http"})
        assert proto == "streamable_http"

    def test_select_transport_none_available(self):
        proto = select_transport(available=set())
        assert proto is None


class TestWeek7ArchiveHandlers:
    def test_discourse_handler_queries_archive(self, tmp_path):
        cfg = RetrievalConfig(archive_db=str(tmp_path / "archive.db"))
        archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
        archive.upsert(
            ArchiveRecord(
                id="arc-1",
                source="nervos_talk",
                doc_type="forum_post",
                url="https://talk.nervos.org/t/fiber/1/2",
                anchor="doc:nervos-talk-1#post:2",
                title="Fiber timeout discussion",
                summary="Community discussion about HTLC timeout.",
                keywords="fiber,htlc,timeout",
                raw_text="The community recommends checking HTLC timeout before opening the channel.",
                raw_format="html",
                lang="en",
                version="latest",
                topic="fiber",
                content_hash="hash-discourse-1",
            )
        )
        req = build_tool_call_request(
            request_id="r1",
            step_id="s1",
            tool="discourse_query",
            args={"query": "HTLC timeout", "category": "fiber", "top_k": 3, "_archive_store": archive},
        )
        result = handle_discourse_query(req)
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["source"] == "discourse"

    def test_github_handler_prefers_transport_when_present(self):
        transport = MockTransportAdapter(
            fixed_response={
                "data": {"provider": "mock"},
                "evidence": [
                    {
                        "id": "gh-1",
                        "source": "github",
                        "title": "nervosnetwork/fiber/docs/channel.md",
                        "url": "https://github.com/nervosnetwork/fiber/blob/main/docs/channel.md",
                        "anchor": "doc:github-fiber#blob:1",
                        "snippet": "openChannel example",
                        "score": 0.8,
                        "payload": {"source": "github_docs", "type": "github_doc", "version": "main"},
                        "hash": "hash-gh-1",
                        "retrieved_ts_ms": 1,
                    }
                ],
                "raw_size_bytes": 18,
                "redactions_applied": [],
            }
        )
        req = build_tool_call_request(
            request_id="r2",
            step_id="s2",
            tool="github_search",
            args={"query": "openChannel", "repo": "nervosnetwork/fiber", "_transport": transport},
        )
        result = handle_github_search(req)
        assert len(result["evidence"]) == 1
        assert result["data"]["provider"] == "mock"
