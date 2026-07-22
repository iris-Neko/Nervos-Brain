from nervos_brain.tool_runtime.telegram_bot_protocol_adapter import (
    message_envelope_to_graph_state,
    outbound_message_to_telegram_requests,
    telegram_update_to_message_envelope,
)


def test_update_to_envelope_basic_text():
    update = {
        "update_id": 1001,
        "message": {
            "message_id": 99,
            "date": 1711111111,
            "text": "hello fiber",
            "chat": {"id": 123456, "type": "private"},
            "from": {"id": 42, "language_code": "en"},
        },
    }
    env = telegram_update_to_message_envelope(update)

    assert env["kind"] == "message"
    assert env["message_id"] == "99"
    assert env["ts_ms"] == 1711111111000
    assert env["content"] == "hello fiber"
    assert env["context"]["platform"] == "telegram"
    assert env["context"]["user_id"] == "42"
    assert env["context"]["channel_id"] == "123456"
    assert env["platform_locale_hint"] == "en"


def test_update_to_envelope_prefers_english_text_over_chinese_account_locale():
    update = {
        "update_id": 10011,
        "message": {
            "message_id": 99,
            "date": 1711111111,
            "text": "@NBCKB_Bot What is CKB, I am a rookie",
            "chat": {"id": -10088, "type": "supergroup"},
            "from": {"id": 42, "language_code": "zh-hans"},
        },
    }
    env = telegram_update_to_message_envelope(update)

    assert env["content"] == "What is CKB, I am a rookie"
    assert env["platform_locale_hint"] == "zh-hans"


def test_update_to_envelope_prefers_chinese_text_over_english_account_locale():
    update = {
        "update_id": 10012,
        "message": {
            "message_id": 99,
            "date": 1711111111,
            "text": "@NBCKB_Bot CKB 是什么？我是新手",
            "chat": {"id": -10088, "type": "supergroup"},
            "from": {"id": 42, "language_code": "en"},
        },
    }
    env = telegram_update_to_message_envelope(update)

    assert env["content"] == "CKB 是什么？我是新手"
    assert env["platform_locale_hint"] == "en"


def test_update_to_envelope_strips_bot_mention_from_group_message():
    update = {
        "update_id": 1001,
        "message": {
            "message_id": 99,
            "date": 1711111111,
            "text": "@NBCKB_Bot ckb是什么",
            "chat": {"id": -10088, "type": "supergroup"},
            "from": {"id": 42, "language_code": "zh-hans"},
        },
    }
    env = telegram_update_to_message_envelope(update)

    assert env["content"] == "ckb是什么"


def test_update_to_envelope_command_and_args():
    update = {
        "update_id": 1002,
        "message": {
            "message_id": 10,
            "date": 1711111112,
            "text": "/ask fiber open channel",
            "chat": {"id": -100100, "type": "supergroup"},
            "from": {"id": 8},
            "message_thread_id": 7,
        },
    }
    env = telegram_update_to_message_envelope(update)
    assert env["kind"] == "command"
    assert env["command"] == "/ask"
    assert env["command_args"] == "fiber open channel"
    assert env["context"]["guild_id"] == "-100100"
    assert env["context"]["thread_id"] == "7"


def test_update_to_envelope_command_with_bot_username_and_args():
    update = {
        "update_id": 10021,
        "message": {
            "message_id": 10,
            "date": 1711111112,
            "text": "/ask@NBCKB_Bot fiber open channel",
            "chat": {"id": -100100, "type": "supergroup"},
            "from": {"id": 8},
        },
    }

    env = telegram_update_to_message_envelope(update)

    assert env["kind"] == "command"
    assert env["command"] == "/ask"
    assert env["command_args"] == "fiber open channel"


def test_update_to_envelope_preserves_reply_content_snapshot():
    update = {
        "update_id": 10022,
        "message": {
            "message_id": 12,
            "date": 1711111112,
            "text": "你都没给引用凭什么说有",
            "chat": {"id": -100100, "type": "supergroup"},
            "from": {"id": 8},
            "reply_to_message": {
                "message_id": 11,
                "date": 1711111111,
                "text": "有，CCC 相关的基础概念我可以直接聊。",
                "chat": {"id": -100100, "type": "supergroup"},
                "from": {"id": 999, "is_bot": True, "username": "NBCKB_Bot"},
            },
        },
    }

    env = telegram_update_to_message_envelope(update)

    assert env["reply_to_message_id"] == "11"
    assert env["reply_to_role"] == "assistant"
    assert "CCC 相关" in env["reply_to_content"]


def test_update_to_envelope_with_attachments():
    update = {
        "update_id": 1003,
        "message": {
            "message_id": 11,
            "date": 1711111113,
            "caption": "see docs https://example.com/page",
            "chat": {"id": -200, "type": "group"},
            "from": {"id": 9},
            "photo": [
                {"file_id": "small"},
                {"file_id": "large", "file_unique_id": "p1"},
            ],
            "document": {"file_id": "doc1", "file_name": "a.pdf"},
            "entities": [{"type": "url", "offset": 9, "length": 24}],
        },
    }
    env = telegram_update_to_message_envelope(update)
    assert "attachments" in env
    assert len(env["attachments"]) >= 2
    assert any(att["kind"] == "image" for att in env["attachments"])
    assert any(att["kind"] == "file" for att in env["attachments"])


def test_envelope_to_graph_state_shape():
    env = {
        "kind": "message",
        "ts_ms": 1711111111000,
        "message_id": "99",
        "context": {"platform": "telegram", "user_id": "42", "channel_id": "123456"},
        "content": "hello",
        "platform_locale_hint": "zh-CN",
    }
    state = message_envelope_to_graph_state(env, request_id="req-x")
    assert state["request_id"] == "req-x"
    assert state["user_message"]["content"] == "hello"
    assert state["user_memory_key"]["platform"] == "telegram"
    assert state["route"] == "graph"


def test_outbound_to_telegram_requests_markdown_reply():
    outbound = {
        "request_id": "req1",
        "context": {"platform": "telegram", "user_id": "42", "channel_id": "-10088"},
        "reply_to_message_id": "333",
        "segments": [
            {
                "segment_id": "req1:0",
                "index": 0,
                "text": "part1",
                "char_count": 5,
                "citation_labels": [],
            },
            {
                "segment_id": "req1:1",
                "index": 1,
                "text": "part2",
                "char_count": 5,
                "citation_labels": [],
            },
        ],
        "render_mode": "markdown",
        "append_csat": False,
    }
    reqs = outbound_message_to_telegram_requests(outbound)
    assert len(reqs) == 2
    assert reqs[0]["method"] == "sendMessage"
    assert reqs[0]["payload"]["chat_id"] == -10088
    assert reqs[0]["payload"]["parse_mode"] == "MarkdownV2"
    assert reqs[0]["payload"]["reply_to_message_id"] == 333
    assert "reply_to_message_id" not in reqs[1]["payload"]


def test_outbound_to_telegram_requests_uses_entities_without_parse_mode():
    outbound = {
        "request_id": "req1",
        "context": {"platform": "telegram", "user_id": "42", "channel_id": "-10088"},
        "reply_to_message_id": "333",
        "segments": [
            {
                "segment_id": "req1:0",
                "index": 0,
                "text": "part1",
                "char_count": 5,
                "citation_labels": [],
                "entities": [{"type": "bold", "offset": 0, "length": 5}],
            },
            {
                "segment_id": "req1:1",
                "index": 1,
                "text": "part2",
                "char_count": 5,
                "citation_labels": [],
                "entities": [{"type": "italic", "offset": 0, "length": 5}],
            },
        ],
        "render_mode": "markdown",
        "append_csat": False,
    }

    reqs = outbound_message_to_telegram_requests(outbound)

    assert reqs[0]["payload"]["entities"] == [{"type": "bold", "offset": 0, "length": 5}]
    assert "parse_mode" not in reqs[0]["payload"]
    assert reqs[0]["payload"]["reply_to_message_id"] == 333
    assert "parse_mode" not in reqs[1]["payload"]
    assert "reply_to_message_id" not in reqs[1]["payload"]


def test_outbound_to_telegram_requests_can_disable_parse_mode_without_entities():
    outbound = {
        "request_id": "req1",
        "context": {"platform": "telegram", "user_id": "42", "channel_id": "-10088"},
        "segments": [
            {
                "segment_id": "req1:0",
                "index": 0,
                "text": "plain telegramify tail",
                "char_count": 22,
                "citation_labels": [],
                "parse_mode_enabled": False,
            }
        ],
        "render_mode": "markdown",
        "append_csat": False,
    }

    reqs = outbound_message_to_telegram_requests(outbound)

    assert "entities" not in reqs[0]["payload"]
    assert "parse_mode" not in reqs[0]["payload"]


def test_outbound_to_telegram_requests_attaches_csat_to_last_segment_only():
    outbound = {
        "request_id": "tg-req-1",
        "context": {"platform": "telegram", "user_id": "42", "channel_id": "-10088"},
        "segments": [
            {
                "segment_id": "tg-req-1:0",
                "index": 0,
                "text": "part1",
                "char_count": 5,
                "citation_labels": [],
            },
            {
                "segment_id": "tg-req-1:1",
                "index": 1,
                "text": "part2",
                "char_count": 5,
                "citation_labels": [],
            },
        ],
        "render_mode": "markdown",
        "append_csat": True,
    }

    reqs = outbound_message_to_telegram_requests(outbound)

    assert "reply_markup" not in reqs[0]["payload"]
    keyboard = reqs[1]["payload"]["reply_markup"]["inline_keyboard"][0]
    assert [button["text"] for button in keyboard] == ["1", "2", "3", "4", "5"]
    assert keyboard[0]["callback_data"] == "csat:tg-req-1:1"
    assert keyboard[4]["callback_data"] == "csat:tg-req-1:5"
