from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from nervos_brain.pathing import project_root

from nervos_brain.tool_runtime.discord_bot_runtime import (
    DiscordBotConfig,
    DiscordBotRuntime,
    DiscordBotRuntimeError,
    DiscordGateway,
    _detect_discord_progress_locale,
    _send_discord_progress_updates,
)
from nervos_brain.tool_runtime.fast_mode import FastModeStateStore
from nervos_brain.tool_runtime.progress import ProgressUpdateConfig


def _load_script_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sample_payload(
    *,
    message_id: str = "m-1",
    guild_id: str | None = "g-1",
    channel_id: str = "c-1",
    user_id: str = "u-1",
    text: str = "hello",
    is_bot: bool = False,
    mention_user_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "content": text,
        "timestamp": "2024-03-22T10:11:12Z",
        "author": {"id": user_id, "bot": is_bot},
        "guild_id": guild_id,
        "channel_id": channel_id,
        "mention_user_ids": mention_user_ids or [],
    }


def test_discord_bot_config_from_env_requires_token(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    with pytest.raises(DiscordBotRuntimeError, match="DISCORD_BOT_TOKEN"):
        DiscordBotConfig.from_env()


def test_discord_bot_config_from_env_parses_mention_flag(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "abc")
    monkeypatch.setenv("DISCORD_MENTION_ONLY_IN_GUILD", "false")
    cfg = DiscordBotConfig.from_env()
    assert cfg.bot_token == "abc"
    assert cfg.mention_only_in_guild is False


def test_discord_bot_runtime_clamps_worker_count():
    gateway = DiscordGateway(graph_runner=lambda state: {})
    runtime = DiscordBotRuntime(
        config=DiscordBotConfig(bot_token="token"),
        gateway=gateway,
        max_worker_threads=999,
    )

    assert runtime._max_worker_threads == 32


def test_gateway_ignores_bot_sender():
    gateway = DiscordGateway(graph_runner=lambda state: {})
    row = gateway.process_message_payload(_sample_payload(is_bot=True), bot_user_id="b-1")
    assert row["ignored"] is True
    assert row["reason"] == "bot_sender"


def test_gateway_guild_filter_blocks_not_allowed():
    gateway = DiscordGateway(
        graph_runner=lambda state: {},
        allowed_guild_ids={"g-allowed"},
    )
    row = gateway.process_message_payload(_sample_payload(guild_id="g-other"), bot_user_id="b-1")
    assert row["ignored"] is True
    assert row["reason"] == "guild_not_allowed"


def test_gateway_channel_filter_blocks_not_allowed():
    gateway = DiscordGateway(
        graph_runner=lambda state: {},
        allowed_channel_ids={"c-allowed"},
    )
    row = gateway.process_message_payload(_sample_payload(channel_id="c-other"), bot_user_id="b-1")
    assert row["ignored"] is True
    assert row["reason"] == "channel_not_allowed"


def test_gateway_guild_requires_bot_user_id_when_mention_only_enabled():
    gateway = DiscordGateway(graph_runner=lambda state: {}, mention_only_in_guild=True)
    row = gateway.process_message_payload(_sample_payload(guild_id="g-1", text="hello"), bot_user_id=None)
    assert row["ignored"] is True
    assert row["reason"] == "missing_bot_user_id"


def test_gateway_guild_mention_only_ignores_if_not_mentioned():
    gateway = DiscordGateway(graph_runner=lambda state: {}, mention_only_in_guild=True)
    row = gateway.process_message_payload(_sample_payload(guild_id="g-1", text="hello"), bot_user_id="999")
    assert row["ignored"] is True
    assert row["reason"] == "not_mentioned"


def test_gateway_processes_when_mentioned_and_strips_mention():
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {
            "_final_response": {
                "request_id": state["request_id"],
                "text": "ok",
                "citations": [],
            }
        }

    gateway = DiscordGateway(graph_runner=runner, mention_only_in_guild=True)
    row = gateway.process_message_payload(
        _sample_payload(
            guild_id="g-1",
            text="<@!999> 什么是 ckb",
            mention_user_ids=["999"],
        ),
        bot_user_id="999",
    )
    assert row["ignored"] is False
    assert row["request_id"].startswith("dc-")
    assert len(row["send_requests"]) == 1
    assert "force_retrieval" not in captured
    assert captured["user_message"]["content"] == "什么是 ckb"


def test_gateway_processes_dm_without_mention():
    gateway = DiscordGateway(
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        mention_only_in_guild=True,
    )
    row = gateway.process_message_payload(_sample_payload(guild_id=None, text="hello"), bot_user_id=None)
    assert row["ignored"] is False
    assert len(row["send_requests"]) == 1


def test_gateway_runner_error_returns_fallback_message():
    def boom(_state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    gateway = DiscordGateway(graph_runner=boom, mention_only_in_guild=False)
    row = gateway.process_message_payload(_sample_payload(guild_id=None, text="hello"), bot_user_id=None)
    assert row["ignored"] is False
    assert len(row["send_requests"]) == 1
    assert "处理请求时发生错误" in row["send_requests"][0]["payload"]["content"]


def test_gateway_splits_long_response():
    gateway = DiscordGateway(
        graph_runner=lambda state: {
            "_final_response": {
                "request_id": state["request_id"],
                "text": "A" * 4500,
                "citations": [],
            }
        },
        mention_only_in_guild=False,
    )
    row = gateway.process_message_payload(_sample_payload(guild_id=None, text="hello"), bot_user_id=None)
    assert row["ignored"] is False
    assert len(row["send_requests"]) >= 3


def test_build_runtime_defaults_to_dynamic_provider_registry(monkeypatch, tmp_path):
    from nervos_brain.graph_engine.provider_registry import ProviderCapabilityRegistry

    runner = _load_script_module("run_discord_bot_under_test", "run_discord_bot.py")

    class FakeRetriever:
        backend_names = ["fake"]

    monkeypatch.setattr(runner, "build_configured_retriever", lambda: FakeRetriever())

    runtime, memory_service = runner._build_runtime(model="", memory_db=tmp_path / "memory.db")

    assert memory_service is not None
    assert isinstance(runtime.provider_registry, ProviderCapabilityRegistry)
    assert runtime.provider_registry.get_profile_for("general", tier="router")["tier"] == "router"
    mini_high = runtime.provider_registry.get_profile_for("planning", tier="mini_high")
    assert mini_high["tier"] == "mini_high"
    assert mini_high["model"] == "openai/gpt-5.4-mini"
    assert mini_high["reasoning_effort"] == "high"


def test_build_runtime_model_argument_forces_fixed_registry(monkeypatch, tmp_path):
    runner = _load_script_module("run_discord_bot_under_test_override", "run_discord_bot.py")

    class FakeRetriever:
        backend_names = ["fake"]

    monkeypatch.setattr(runner, "build_configured_retriever", lambda: FakeRetriever())

    runtime, memory_service = runner._build_runtime(model="openai/gpt-5.5", memory_db=tmp_path / "memory.db")

    assert memory_service is not None
    profile = runtime.provider_registry.get_profile_for("reflection", tier="mini_high")
    assert profile["model"] == "openai/gpt-5.5"
    assert profile["reasoning_effort"] == ""


def test_discord_runner_config_loads_when_cwd_differs(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("discord_bot:\n  memory_db: data/discord_bot/memory.db\n", encoding="utf-8")
    monkeypatch.setenv("NERVOS_BRAIN_CONFIG", str(cfg_path))
    from nervos_brain import pathing
    pathing.config_path.cache_clear()
    pathing.load_project_config.cache_clear()
    runner = _load_script_module("run_discord_bot_pathing_config", "run_discord_bot.py")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    cfg = runner._load_project_cfg()

    assert cfg["discord_bot"]["memory_db"] == "data/discord_bot/memory.db"


def test_discord_runner_resolves_runtime_paths_to_project_root(monkeypatch, tmp_path):
    runner = _load_script_module("run_discord_bot_pathing_runtime", "run_discord_bot.py")
    monkeypatch.chdir(tmp_path)

    resolved = runner.resolve_project_path("data/discord_bot/memory.db")

    assert resolved == project_root() / "data" / "discord_bot" / "memory.db"


class _FakeMemoryService:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.writes: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []

    def list_recent_message_events(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.reads.append(dict(kwargs))
        return list(self.rows)

    def write_message_event(self, **kwargs: Any) -> str:
        self.writes.append(dict(kwargs))
        return f"evt-{len(self.writes)}"


def test_gateway_processes_reply_to_bot_without_mention_and_injects_reply_context():
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(graph_runner=runner, mention_only_in_guild=True, respond_to_bot_replies=True)
    payload = _sample_payload(guild_id="g-1", text="继续说")
    payload["reference"] = {
        "message_id": "m-bot",
        "content": "上一条回答内容",
        "author": {"id": "999", "bot": True},
    }

    row = gateway.process_message_payload(payload, bot_user_id="999")

    assert row["ignored"] is False
    assert captured["user_message"]["reply_to_message_id"] == "m-bot"
    assert "上一条回答内容" in captured["conversation_context"]


def test_gateway_reply_context_does_not_mix_unrelated_recent_context():
    memory = _FakeMemoryService(
        rows=[
            {"role": "user", "content": "Fiber WASM 是怎么实现的？"},
            {"role": "assistant", "content": "WASM 可以放一些轻量验证逻辑。"},
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(
        graph_runner=runner,
        mention_only_in_guild=True,
        respond_to_bot_replies=True,
        memory_service=memory,
    )
    payload = _sample_payload(guild_id="g-1", text="小白版的解释，你说一下")
    payload["reference"] = {
        "message_id": "m-bot",
        "content": "CKB 通常指 Nervos CKB，是 Nervos 生态里的公链基础层。",
        "author": {"id": "999", "bot": True},
    }

    row = gateway.process_message_payload(payload, bot_user_id="999")

    assert row["ignored"] is False
    assert memory.reads == []
    assert captured["recent_messages"] == []
    assert "CKB 通常指 Nervos CKB" in captured["conversation_context"]
    assert "Fiber WASM" not in captured["conversation_context"]


def test_gateway_reply_without_snapshot_does_not_guess_from_recent_context():
    memory = _FakeMemoryService(
        rows=[
            {"role": "user", "content": "Fiber WASM 是怎么实现的？"},
            {"role": "assistant", "content": "WASM 可以放一些轻量验证逻辑。"},
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(
        graph_runner=runner,
        mention_only_in_guild=True,
        respond_to_bot_replies=True,
        memory_service=memory,
    )
    payload = _sample_payload(guild_id="g-1", text="小白版的解释，你说一下")
    payload["reference"] = {
        "message_id": "m-bot",
        "author": {"id": "999", "bot": True},
    }

    row = gateway.process_message_payload(payload, bot_user_id="999")

    assert row["ignored"] is False
    assert memory.reads == []
    assert captured["recent_messages"] == []
    assert "平台没有提供被回复消息内容" in captured["conversation_context"]
    assert "不要从普通历史记录猜测被回复内容" in captured["conversation_context"]
    assert "Fiber WASM" not in captured["conversation_context"]


def test_gateway_ignored_message_does_not_write_memory():
    memory = _FakeMemoryService()
    gateway = DiscordGateway(
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        mention_only_in_guild=True,
        memory_service=memory,
    )

    row = gateway.process_message_payload(_sample_payload(guild_id="g-1", text="hello"), bot_user_id="999")

    assert row["ignored"] is True
    assert memory.writes == []


def test_gateway_writes_memory_and_uses_recent_context_for_followup():
    memory = _FakeMemoryService(rows=[{"role": "assistant", "content": "之前解释过 CCC。"}])
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(
        graph_runner=runner,
        mention_only_in_guild=False,
        memory_service=memory,
        memory_context_limit=7,
    )

    row = gateway.process_message_payload(_sample_payload(guild_id=None, text="继续"), bot_user_id=None)

    assert row["ignored"] is False
    assert len(memory.writes) == 2
    assert memory.writes[0]["role"] == "user"
    assert memory.writes[1]["role"] == "assistant"
    assert memory.reads[0]["limit"] == 7
    assert "之前解释过 CCC" in captured["conversation_context"]


def test_gateway_feedback_command_writes_comment(tmp_path):
    from nervos_brain.tool_runtime.feedback import FeedbackJsonlStore

    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    store.append_answer(
        {
            "request_id": "dc-req",
            "platform": "discord",
            "chat_id": "c-1",
            "user_id": "u-1",
            "trace_summary": "trace",
        }
    )
    gateway = DiscordGateway(graph_runner=lambda state: {}, feedback_store=store, mention_only_in_guild=True)

    row = gateway.process_message_payload(_sample_payload(text="/feedback dc-req 资料不够"), bot_user_id="999")

    assert row["ignored"] is False
    assert row["reason"] == "feedback"
    assert row["send_requests"][0]["payload"]["allowed_mentions"] == {"parse": []}
    comments = [item for item in store.iter_records() if item.get("kind") == "comment"]
    assert comments[0]["comment"] == "资料不够"
    assert comments[0]["platform"] == "discord"


def test_gateway_process_interaction_payload_writes_csat(tmp_path):
    from nervos_brain.tool_runtime.feedback import FeedbackJsonlStore

    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    store.append_answer({"request_id": "dc-req", "platform": "discord", "tool_calls": 2})
    gateway = DiscordGateway(graph_runner=lambda state: {}, feedback_store=store)

    row = gateway.process_interaction_payload(
        {
            "data": {"custom_id": "csat:dc-req:5"},
            "user": {"id": "u-1"},
            "channel_id": "c-1",
            "guild_id": "g-1",
        }
    )

    assert row["ignored"] is False
    assert row["score"] == 5
    ratings = [item for item in store.iter_records() if item.get("kind") == "csat"]
    assert ratings[0]["platform"] == "discord"
    assert ratings[0]["tool_calls"] == 2


def test_gateway_writes_debug_event(tmp_path):
    debug_file = tmp_path / "debug.jsonl"
    gateway = DiscordGateway(
        graph_runner=lambda state: {
            "_final_response": {"request_id": state["request_id"], "text": "ok", "citations": []},
            "_tool_calls_executed": 1,
        },
        mention_only_in_guild=False,
        debug_log_file=debug_file,
    )

    row = gateway.process_message_payload(_sample_payload(guild_id=None, text="hello"), bot_user_id=None)

    assert row["ignored"] is False
    lines = debug_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"platform": "discord"' in lines[0]
    assert '"tool_calls": 1' in lines[0]


def test_gateway_text_attachment_is_added_to_graph_context(monkeypatch):
    import nervos_brain.tool_runtime.discord_bot_runtime as runtime_mod

    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    monkeypatch.setattr(runtime_mod, "_download_attachment", lambda url, max_bytes: b"hello from attachment")
    gateway = DiscordGateway(graph_runner=runner, mention_only_in_guild=False)
    payload = _sample_payload(guild_id=None, text="看附件")
    payload["attachments"] = [
        {
            "id": "a-1",
            "url": "https://example.com/readme.md",
            "filename": "readme.md",
            "content_type": "text/markdown",
            "size": 64,
        }
    ]

    row = gateway.process_message_payload(payload, bot_user_id=None)

    assert row["ignored"] is False
    assert "hello from attachment" in captured["user_message"]["content"]
    assert captured["user_message"]["attachments"][0]["status"] == "ready"


def test_fast_command_sets_one_shot_priority_for_next_discord_request(tmp_path: Path):
    store = FastModeStateStore(tmp_path / "fast.json")
    captured: list[dict[str, Any]] = []

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(state))
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(
        graph_runner=runner,
        mention_only_in_guild=False,
        fast_mode_store=store,
    )

    command_row = gateway.process_message_payload(_sample_payload(text="/fast"), bot_user_id=None)
    first_row = gateway.process_message_payload(_sample_payload(message_id="m-2", text="ckb是什么"), bot_user_id=None)
    second_row = gateway.process_message_payload(_sample_payload(message_id="m-3", text="fiber是什么"), bot_user_id=None)

    assert command_row["reason"] == "fast_command"
    assert command_row["fast_action"] == "enable"
    assert "已开启 fast 模式" in command_row["send_requests"][0]["payload"]["content"]
    assert first_row["ignored"] is False
    assert second_row["ignored"] is False
    assert len(captured) == 2
    assert captured[0]["_llm_service_tier"] == "priority"
    assert "_llm_service_tier" not in captured[1]
    assert store.is_pending(platform="discord", user_id="u-1") is False


def test_fast_command_is_user_scoped_for_discord(tmp_path: Path):
    store = FastModeStateStore(tmp_path / "fast.json")
    captured: list[dict[str, Any]] = []

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(state))
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = DiscordGateway(
        graph_runner=runner,
        mention_only_in_guild=False,
        fast_mode_store=store,
    )

    gateway.process_message_payload(_sample_payload(text="/fast", user_id="u-1"), bot_user_id=None)
    gateway.process_message_payload(_sample_payload(message_id="m-2", text="ckb是什么", user_id="u-2"), bot_user_id=None)
    gateway.process_message_payload(_sample_payload(message_id="m-3", text="ckb是什么", user_id="u-1"), bot_user_id=None)

    assert "_llm_service_tier" not in captured[0]
    assert captured[1]["_llm_service_tier"] == "priority"


def test_fast_off_and_status_do_not_enter_discord_graph(tmp_path: Path):
    store = FastModeStateStore(tmp_path / "fast.json")
    calls: list[dict[str, Any]] = []
    gateway = DiscordGateway(
        graph_runner=lambda state: calls.append(state) or {},
        mention_only_in_guild=False,
        fast_mode_store=store,
    )

    gateway.process_message_payload(_sample_payload(text="/fast"), bot_user_id=None)
    status_row = gateway.process_message_payload(_sample_payload(message_id="m-2", text="/fast status"), bot_user_id=None)
    off_row = gateway.process_message_payload(_sample_payload(message_id="m-3", text="/fast off"), bot_user_id=None)

    assert status_row["reason"] == "fast_command"
    assert off_row["fast_action"] == "disable"
    assert calls == []
    assert "已待命" in status_row["send_requests"][0]["payload"]["content"]
    assert "已取消 fast 模式" in off_row["send_requests"][0]["payload"]["content"]
    assert store.is_pending(platform="discord", user_id="u-1") is False


def test_fast_command_works_as_discord_runtime_command_in_guild(tmp_path: Path):
    store = FastModeStateStore(tmp_path / "fast.json")
    gateway = DiscordGateway(
        graph_runner=lambda state: state,
        mention_only_in_guild=True,
        fast_mode_store=store,
    )

    plain_command = gateway.process_message_payload(_sample_payload(guild_id="g-1", text="/fast"), bot_user_id="999")
    mentioned_command = gateway.process_message_payload(
        _sample_payload(guild_id="g-1", text="<@999> /fast", mention_user_ids=["999"]),
        bot_user_id="999",
    )

    assert plain_command["reason"] == "fast_command"
    assert mentioned_command["reason"] == "fast_command"
    assert store.is_pending(platform="discord", user_id="u-1") is True


def test_discord_progress_locale_strips_bot_mention_and_prefers_message_language():
    payload = _sample_payload(
        text="<@999> How to set up a Fiber node?",
        mention_user_ids=["999"],
    )
    payload["author"]["locale"] = "zh-CN"

    assert _detect_discord_progress_locale(payload, "999") == "en"


class _FakeDiscordChannel:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, content: str, **kwargs: Any) -> None:
        self.sent.append({"content": content, **kwargs})


class _FakeDiscordMessage:
    def __init__(self, *, reply_fails: bool = False) -> None:
        self.channel = _FakeDiscordChannel()
        self.replies: list[dict[str, Any]] = []
        self.reply_fails = reply_fails

    async def reply(self, content: str, **kwargs: Any) -> None:
        if self.reply_fails:
            raise RuntimeError("reply failed")
        self.replies.append({"content": content, **kwargs})


def test_discord_progress_updates_send_visible_replies():
    message = _FakeDiscordMessage()

    asyncio.run(
        _send_discord_progress_updates(
            message=message,
            config=ProgressUpdateConfig(
                enabled=True,
                first_after_s=0.01,
                interval_s=0.01,
                max_updates=2,
                messages=("还在处理 1", "还在处理 2"),
            ),
            locale="en",
        )
    )

    assert [item["content"] for item in message.replies] == [
        "I am still checking sources and organizing the evidence. Please wait a moment.",
        "There is quite a bit of material, so I am verifying the sources and shaping the answer.",
    ]
    assert message.channel.sent == []


def test_discord_progress_updates_send_chinese_visible_replies():
    message = _FakeDiscordMessage()

    asyncio.run(
        _send_discord_progress_updates(
            message=message,
            config=ProgressUpdateConfig(
                enabled=True,
                first_after_s=0.01,
                interval_s=0.01,
                max_updates=1,
                messages=("还在处理",),
            ),
            locale="zh-CN",
        )
    )

    assert [item["content"] for item in message.replies] == ["还在处理"]


def test_discord_progress_updates_fallback_to_channel_send_when_reply_fails():
    message = _FakeDiscordMessage(reply_fails=True)

    asyncio.run(
        _send_discord_progress_updates(
            message=message,
            config=ProgressUpdateConfig(
                enabled=True,
                first_after_s=0.01,
                interval_s=0.01,
                max_updates=1,
                messages=("还在处理",),
            ),
            locale="en",
        )
    )

    assert message.replies == []
    assert [item["content"] for item in message.channel.sent] == [
        "I am still checking sources and organizing the evidence. Please wait a moment."
    ]
