# ============================================================
# test_protocols_valid.py —— M2-T24 协议合法数据测试
# ============================================================
# 这个文件测试的是：当你给一个"完全正确"的字典时，
# 校验函数应该返回空列表（没有错误）。
#
# 为什么要写这种测试？
#   如果你只测"非法数据能不能被拦住"，
#   万一校验函数写得太严格，把合法数据也拦了，你根本发现不了。
#   所以必须同时测"合法数据能不能顺利通过"。
#
# 运行方式：
#   conda activate NervosBrain
#   cd 项目根目录
#   pytest tests/test_protocols_valid.py -v
# ============================================================

import time

from nervos_brain.core_protocols.validators import (
    validate_assistant_response,
    validate_evidence,
    validate_message_envelope,
    validate_tool_call_request,
)


class TestValidMessageEnvelope:
    """测试合法的 MessageEnvelope 数据"""

    def test_minimal_message(self):
        """最小合法消息：只有必填字段"""
        msg = {
            "kind": "message",
            "ts_ms": 1710000000000,
            "message_id": "msg_001",
            "context": {
                "platform": "discord",
                "user_id": "u123",
            },
            "content": "你好，请问 CKB 的 Cell 模型是什么？",
        }
        errors = validate_message_envelope(msg)
        assert errors == [], f"合法消息不应报错，但收到: {errors}"

    def test_command_message(self):
        """命令消息（kind=command）也应该通过"""
        msg = {
            "kind": "command",
            "ts_ms": 1710000000000,
            "message_id": "msg_002",
            "context": {
                "platform": "telegram",
                "user_id": "u456",
            },
            "content": "/tldr https://example.com/fiber-rfc",
            "command": "/tldr",
            "command_args": "https://example.com/fiber-rfc",
        }
        errors = validate_message_envelope(msg)
        assert errors == [], f"命令消息不应报错，但收到: {errors}"

    def test_full_message_with_optional_fields(self):
        """包含所有可选字段的完整消息"""
        msg = {
            "kind": "message",
            "ts_ms": 1710000000000,
            "message_id": "msg_003",
            "context": {
                "platform": "discord",
                "user_id": "u789",
                "guild_id": "g001",
                "channel_id": "c001",
                "thread_id": "t001",
            },
            "content": "Fiber SDK 怎么发交易？",
            "reply_to_message_id": "msg_000",
            "attachments": [
                {"kind": "image", "url": "https://example.com/screenshot.png"},
            ],
            "platform_locale_hint": "zh-CN",
        }
        errors = validate_message_envelope(msg)
        assert errors == [], f"完整消息不应报错，但收到: {errors}"


class TestValidEvidence:
    """测试合法的 Evidence 数据"""

    def test_minimal_evidence(self):
        """最小合法证据"""
        evidence = {
            "id": "ev_001",
            "source": "qdrant",
            "title": "CKB Cell Model 介绍",
            "url": "https://docs.nervos.org/cell-model",
            "anchor": "section:2.1",
            "snippet": "Cell 是 CKB 的基本数据单元，类似于比特币的 UTXO...",
            "score": 0.92,
            "payload": {
                "source": "docs",
                "type": "doc",
                "version": "latest",
                "lang": "zh",
            },
            "hash": "abc123def456",
            "retrieved_ts_ms": 1710000000000,
        }
        errors = validate_evidence(evidence)
        assert errors == [], f"合法证据不应报错，但收到: {errors}"

    def test_evidence_from_discourse(self):
        """来自论坛的证据"""
        evidence = {
            "id": "ev_002",
            "source": "discourse",
            "title": "关于 Fiber 通道开通的讨论",
            "url": "https://talk.nervos.org/t/fiber-channel/1234",
            "anchor": "topic:1234#post:47",
            "snippet": "开通 Fiber 通道需要先在链上锁定 CKB...",
            "score": 0.85,
            "payload": {
                "source": "talk",
                "type": "forum",
                "version": "unknown",
                "lang": "en",
            },
            "hash": "def789ghi012",
            "retrieved_ts_ms": 1710000000000,
        }
        errors = validate_evidence(evidence)
        assert errors == [], f"论坛证据不应报错，但收到: {errors}"

    def test_evidence_score_boundary(self):
        """边界值：score=0.0 和 score=1.0 都应该通过"""
        base = {
            "id": "ev_003",
            "source": "github",
            "title": "test",
            "url": "https://github.com/test",
            "anchor": "L1-L10",
            "snippet": "test code",
            "payload": {"source": "unknown"},
            "hash": "aaa",
            "retrieved_ts_ms": 1710000000000,
        }
        # score = 0.0
        base["score"] = 0.0
        errors = validate_evidence(base)
        assert errors == [], f"score=0.0 不应报错，但收到: {errors}"

        # score = 1.0
        base["score"] = 1.0
        errors = validate_evidence(base)
        assert errors == [], f"score=1.0 不应报错，但收到: {errors}"


class TestValidAssistantResponse:
    """测试合法的 AssistantResponse 数据"""

    def test_simple_response_with_citation(self):
        """带一个引用的简单回答"""
        resp = {
            "request_id": "req_001",
            "text": "CKB 的 Cell 模型类似于比特币的 UTXO [1]。",
            "citations": [
                {
                    "label": "[1]",
                    "url": "https://docs.nervos.org/cell-model",
                    "anchor": "section:2.1",
                    "title": "CKB Cell Model 介绍",
                },
            ],
        }
        errors = validate_assistant_response(resp)
        assert errors == [], f"合法回答不应报错，但收到: {errors}"

    def test_response_with_multiple_citations(self):
        """带多个引用的回答"""
        resp = {
            "request_id": "req_002",
            "text": "Fiber 是 CKB 上的支付通道 [1]，开通需要锁定资金 [2]。",
            "citations": [
                {
                    "label": "[1]",
                    "url": "https://docs.nervos.org/fiber",
                    "anchor": "section:1",
                    "title": "Fiber 介绍",
                },
                {
                    "label": "[2]",
                    "url": "https://talk.nervos.org/t/1234",
                    "anchor": "post:47",
                    "title": "Fiber 通道讨论",
                },
            ],
        }
        errors = validate_assistant_response(resp)
        assert errors == [], f"多引用回答不应报错，但收到: {errors}"

    def test_response_with_no_citations(self):
        """没有引用的回答（比如纯问候）也应该通过"""
        resp = {
            "request_id": "req_003",
            "text": "你好！有什么可以帮你的吗？",
            "citations": [],
        }
        errors = validate_assistant_response(resp)
        assert errors == [], f"无引用回答不应报错，但收到: {errors}"

    def test_response_with_ask_user(self):
        """需要反问用户的回答"""
        resp = {
            "request_id": "req_004",
            "text": "请问您使用的是哪个版本的 Fiber SDK？",
            "citations": [],
            "need_user_input": True,
            "ask_user_question": "请问您使用的是哪个版本的 Fiber SDK？",
        }
        errors = validate_assistant_response(resp)
        assert errors == [], f"反问回答不应报错，但收到: {errors}"


class TestValidToolCallRequest:
    """测试合法的 ToolCallRequest 数据"""

    def test_qdrant_search_request(self):
        """Qdrant 搜索请求"""
        now = int(time.time() * 1000)
        req = {
            "request_id": "req_001",
            "step_id": "step_1",
            "tool": "qdrant_search",
            "args": {"query": "CKB Cell 模型", "top_k": 5},
            "timeout_ms": 5000,
            "issued_ts_ms": now,
            "deadline_ts_ms": now + 10000,
            "idempotency_key": "idem_abc123",
            "allow_parallel": True,
        }
        errors = validate_tool_call_request(req)
        assert errors == [], f"合法工具请求不应报错，但收到: {errors}"

    def test_discourse_query_request(self):
        """Discourse 论坛搜索请求"""
        now = int(time.time() * 1000)
        req = {
            "request_id": "req_002",
            "step_id": "step_2",
            "tool": "discourse_query",
            "args": {"query": "Fiber channel open", "category": "dev"},
            "timeout_ms": 8000,
            "issued_ts_ms": now,
            "deadline_ts_ms": now + 15000,
            "idempotency_key": "idem_def456",
            "allow_parallel": False,
        }
        errors = validate_tool_call_request(req)
        assert errors == [], f"论坛搜索请求不应报错，但收到: {errors}"
