from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nervos_brain.tool_runtime.discord_bot_protocol_adapter import (
    discord_message_envelope_to_graph_state,
    discord_message_to_message_envelope,
    outbound_message_to_discord_requests,
)


def test_discord_message_to_envelope_basic_text():
    payload = {
        "id": "m-100",
        "content": "hello fiber",
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": "u-1", "locale": "en-US"},
        "channel_id": "c-1",
        "guild_id": "g-1",
    }
    env = discord_message_to_message_envelope(payload)

    expected_ts = int(datetime(2024, 3, 22, 10, 11, 12, tzinfo=timezone.utc).timestamp() * 1000)
    assert env["kind"] == "message"
    assert env["message_id"] == "m-100"
    assert env["ts_ms"] == expected_ts
    assert env["content"] == "hello fiber"
    assert env["context"]["platform"] == "discord"
    assert env["context"]["user_id"] == "u-1"
    assert env["context"]["channel_id"] == "c-1"
    assert env["context"]["guild_id"] == "g-1"
    assert env["locale_hint"] == "en"


def test_discord_message_to_envelope_prefers_english_text_over_chinese_author_locale():
    payload = {
        "id": "m-100",
        "content": "What is CKB, I am a rookie",
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": "u-1", "locale": "zh-CN"},
        "channel_id": "c-1",
        "guild_id": "g-1",
    }
    env = discord_message_to_message_envelope(payload)

    assert env["locale_hint"] == "en"


def test_discord_message_to_envelope_prefers_chinese_text_over_english_author_locale():
    payload = {
        "id": "m-100",
        "content": "CKB 是什么？我是新手",
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": "u-1", "locale": "en-US"},
        "channel_id": "c-1",
        "guild_id": "g-1",
    }
    env = discord_message_to_message_envelope(payload)

    assert env["locale_hint"] == "zh-CN"


def test_discord_message_to_envelope_command_reply_and_attachments():
    payload = {
        "id": "m-101",
        "content": "/ask fiber open channel",
        "timestamp": 1711111111,
        "author": {"id": "u-2"},
        "channel_id": "c-2",
        "guild_id": "g-2",
        "thread_id": "t-2",
        "reference": {"message_id": "m-previous"},
        "attachments": [
            {"id": "a1", "url": "https://example.com/a.png", "filename": "a.png", "content_type": "image/png"},
            {"id": "a2", "url": "https://example.com/b.pdf", "filename": "b.pdf", "content_type": "application/pdf"},
        ],
        "embeds": [{"url": "https://example.com/doc", "title": "Doc"}],
    }
    env = discord_message_to_message_envelope(payload)

    assert env["kind"] == "command"
    assert env["command"] == "/ask"
    assert env["command_args"] == "fiber open channel"
    assert env["reply_to_message_id"] == "m-previous"
    assert env["context"]["thread_id"] == "t-2"
    assert env["locale_hint"] == "en"
    kinds = {item["kind"] for item in env["attachments"]}
    assert kinds == {"image", "file", "link"}


def test_discord_message_envelope_to_graph_state_shape():
    env = {
        "kind": "message",
        "ts_ms": 1711111111000,
        "message_id": "m-200",
        "context": {
            "platform": "discord",
            "user_id": "u-9",
            "guild_id": "g-9",
            "channel_id": "c-9",
            "thread_id": "t-9",
        },
        "content": "what is ckb",
        "locale_hint": "zh-CN",
    }
    state = discord_message_envelope_to_graph_state(env, request_id="dc-req-1")
    assert state["request_id"] == "dc-req-1"
    assert state["route"] == "graph"
    assert state["locale"] == "zh-CN"
    assert state["user_memory_key"]["platform"] == "discord"
    assert state["channel_memory_key"] == {"platform": "discord", "guild_id": "g-9", "channel_id": "c-9"}
    assert state["thread_key"] == {
        "platform": "discord",
        "guild_id": "g-9",
        "channel_id": "c-9",
        "thread_id": "t-9",
    }


def test_outbound_to_discord_requests_reply_on_first_segment_only():
    outbound = {
        "request_id": "dc-1",
        "context": {"platform": "discord", "user_id": "u1", "channel_id": "c-1"},
        "reply_to_message_id": "m-reply",
        "segments": [
            {"segment_id": "dc-1:0", "index": 0, "text": "part1", "char_count": 5, "citation_labels": []},
            {"segment_id": "dc-1:1", "index": 1, "text": "part2", "char_count": 5, "citation_labels": []},
        ],
        "render_mode": "markdown",
        "append_csat": False,
    }
    reqs = outbound_message_to_discord_requests(outbound)
    assert len(reqs) == 2
    assert reqs[0]["method"] == "create_message"
    assert reqs[0]["payload"]["channel_id"] == "c-1"
    assert reqs[0]["payload"]["reply_to_message_id"] == "m-reply"
    assert "reply_to_message_id" not in reqs[1]["payload"]


def test_outbound_to_discord_requests_requires_channel():
    outbound = {
        "request_id": "dc-2",
        "context": {"platform": "discord", "user_id": "u1"},
        "segments": [{"segment_id": "dc-2:0", "index": 0, "text": "x", "char_count": 1, "citation_labels": []}],
        "render_mode": "markdown",
        "append_csat": False,
    }
    with pytest.raises(ValueError, match="channel_id"):
        outbound_message_to_discord_requests(outbound)  # type: ignore[arg-type]


def test_discord_message_to_envelope_rejects_bad_payload():
    with pytest.raises(ValueError, match="payload must be dict"):
        discord_message_to_message_envelope("bad")  # type: ignore[arg-type]


def test_discord_message_to_envelope_reply_snapshot_and_role():
    payload = {
        "id": "m-300",
        "content": "接着解释",
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": "u-3"},
        "channel_id": "c-3",
        "guild_id": "g-3",
        "reference": {
            "message_id": "m-bot",
            "content": "上一条 bot 回答里提到了 CCC transfer API。",
            "author": {"id": "bot-1", "bot": True},
        },
    }

    env = discord_message_to_message_envelope(payload)

    assert env["reply_to_message_id"] == "m-bot"
    assert env["reply_to_role"] == "assistant"
    assert "CCC transfer API" in env["reply_to_content"]


def test_discord_message_to_envelope_preserves_attachment_metadata():
    payload = {
        "id": "m-301",
        "content": "看附件",
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": "u-3"},
        "channel_id": "c-3",
        "attachments": [
            {
                "id": "a-md",
                "url": "https://example.com/readme.md",
                "filename": "readme.md",
                "content_type": "text/markdown",
                "size": 1234,
            }
        ],
    }

    env = discord_message_to_message_envelope(payload)

    attachment = env["attachments"][0]
    assert attachment["name"] == "readme.md"
    assert attachment["mime_type"] == "text/markdown"
    assert attachment["file_size"] == "1234"


def test_outbound_to_discord_requests_allowed_mentions_and_csat_buttons():
    outbound = {
        "request_id": "dc-csat",
        "context": {"platform": "discord", "user_id": "u1", "channel_id": "c-1"},
        "segments": [
            {"segment_id": "dc-csat:0", "index": 0, "text": "part1", "char_count": 5, "citation_labels": []},
            {"segment_id": "dc-csat:1", "index": 1, "text": "part2", "char_count": 5, "citation_labels": []},
        ],
        "render_mode": "markdown",
        "append_csat": True,
    }

    reqs = outbound_message_to_discord_requests(outbound)

    assert all(req["payload"]["allowed_mentions"] == {"parse": []} for req in reqs)
    assert "components" not in reqs[0]["payload"]
    buttons = reqs[1]["payload"]["components"][0]["components"]
    assert len(buttons) == 5
    assert buttons[0]["custom_id"] == "csat:dc-csat:1"
    assert buttons[-1]["custom_id"] == "csat:dc-csat:5"
