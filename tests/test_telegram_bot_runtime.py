from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nervos_brain.pathing import project_root

from nervos_brain.tool_runtime.telegram_bot_runtime import (
    TelegramBotAPI,
    TelegramBotConfig,
    TelegramBotRuntimeError,
    TelegramPollingGateway,
    TelegramUpdateOffsetStore,
)
from nervos_brain.tool_runtime.fast_mode import FastModeStateStore
from nervos_brain.tool_runtime.feedback import FeedbackJsonlStore
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


def _sample_update(
    *,
    update_id: int = 100,
    chat_id: int = -100123,
    thread_id: int | None = None,
    user_id: int = 42,
    text: str = "hello",
    is_bot: bool = False,
    chat_type: str = "supergroup",
    reply_to_bot: bool = False,
    reply_to_user_id: int | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": 1,
        "date": 1711111111,
        "text": text,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": user_id, "is_bot": is_bot},
    }
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    if reply_to_bot:
        message["reply_to_message"] = {
            "message_id": 99,
            "date": 1711111100,
            "text": "bot previous answer",
            "chat": {"id": chat_id, "type": chat_type},
            "from": {"id": 999, "is_bot": True, "username": "NBCKB_Bot"},
        }
    elif reply_to_user_id is not None:
        message["reply_to_message"] = {
            "message_id": 98,
            "date": 1711111100,
            "text": "someone else",
            "chat": {"id": chat_id, "type": chat_type},
            "from": {"id": reply_to_user_id, "is_bot": False, "username": "alice"},
        }
    return {
        "update_id": update_id,
        "message": message,
    }


class _FakeAPI:
    def __init__(self, updates: list[dict[str, Any]] | None = None) -> None:
        self._updates = list(updates or [])
        self.sent_requests: list[dict[str, Any]] = []
        self.last_send_results: list[dict[str, Any]] = []
        self.callback_requests: list[dict[str, Any]] = []
        self.chat_actions: list[dict[str, Any]] = []
        self.files: dict[str, tuple[str, bytes]] = {}

    def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout_s: int = 25,
        limit: int = 20,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        _ = offset, timeout_s, limit, allowed_updates
        return list(self._updates)

    def send_requests(self, requests_payloads: list[dict[str, Any]]) -> int:
        self.sent_requests.extend(requests_payloads)
        self.last_send_results = [
            {
                "method": str(req.get("method", "sendMessage")),
                "message_id": str(9000 + idx),
                "chat_id": str(req.get("payload", {}).get("chat_id", "")),
            }
            for idx, req in enumerate(requests_payloads)
            if isinstance(req, dict)
        ]
        return len(requests_payloads)

    def send_request(
        self,
        *,
        method: str,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> None:
        _ = timeout_s
        self.callback_requests.append({"method": method, "payload": payload})

    def send_chat_action(
        self,
        *,
        chat_id: str,
        action: str = "typing",
        timeout_s: float = 10.0,
    ) -> None:
        _ = timeout_s
        self.chat_actions.append({"chat_id": chat_id, "action": action})

    def get_file(self, file_id: str) -> dict[str, Any]:
        file_path, _data = self.files[file_id]
        return {"file_id": file_id, "file_path": file_path}

    def download_file(self, file_path: str, *, timeout_s: float = 60.0) -> bytes:
        _ = timeout_s
        for known_path, data in self.files.values():
            if known_path == file_path:
                return data
        raise KeyError(file_path)


class _FlakyTelegramAPI(TelegramBotAPI):
    def __init__(self, *, fail_count: int) -> None:
        super().__init__(TelegramBotConfig(bot_token="token"))
        self.fail_count = fail_count
        self.send_calls: list[dict[str, Any]] = []

    def send_request(
        self,
        *,
        method: str,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        _ = timeout_s
        self.send_calls.append({"method": method, "payload": dict(payload)})
        if len(self.send_calls) <= self.fail_count:
            raise TelegramBotRuntimeError("send failed")
        return {"ok": True}


class _AlwaysFailSendAPI(_FakeAPI):
    def send_requests(self, requests_payloads: list[dict[str, Any]]) -> int:
        self.sent_requests.extend(requests_payloads)
        raise TelegramBotRuntimeError("send failed")


class _FakeMemory:
    def __init__(self, recent: list[dict[str, Any]] | None = None, *, fail_writes: bool = False) -> None:
        self.recent = list(recent or [])
        self.fail_writes = fail_writes
        self.read_calls: list[dict[str, Any]] = []
        self.write_calls: list[dict[str, Any]] = []

    def list_recent_message_events(self, **kwargs):
        self.read_calls.append(dict(kwargs))
        return list(self.recent)

    def write_message_event(self, **kwargs):
        self.write_calls.append(dict(kwargs))
        if self.fail_writes:
            raise RuntimeError("memory write failed")
        return f"ev-{len(self.write_calls)}"


def _sample_callback_update(
    *,
    update_id: int = 200,
    chat_id: int = -100123,
    thread_id: int | None = None,
    user_id: int = 42,
    request_id: str = "tg-req-1",
    score: int = 2,
    data: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": 5,
        "date": 1711111112,
        "text": "bot answer preview",
        "chat": {"id": chat_id, "type": "supergroup"},
    }
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cb-1",
            "from": {"id": user_id, "is_bot": False},
            "message": message,
            "data": data if data is not None else f"csat:{request_id}:{score}",
        },
    }


def test_telegram_bot_config_from_env_requires_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(TelegramBotRuntimeError, match="TELEGRAM_BOT_TOKEN"):
        TelegramBotConfig.from_env()


def test_offset_store_roundtrip(tmp_path: Path):
    store = TelegramUpdateOffsetStore(tmp_path / "offset.txt")
    assert store.load() is None
    store.save(12345)
    assert store.load() == 12345


def test_offset_store_invalid_value_returns_none(tmp_path: Path):
    path = tmp_path / "offset.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-int", encoding="utf-8")
    store = TelegramUpdateOffsetStore(path)
    assert store.load() is None


def test_process_update_builds_fallback_outbound_dry_run():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {
            "_final_response": {
                "request_id": state["request_id"],
                "text": "done [1]",
                "citations": [
                    {
                        "label": "[1]",
                        "url": "https://example.com",
                        "anchor": "a1",
                        "title": "Doc",
                    }
                ],
            }
        }

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=True)
    assert row["ignored"] is False
    assert row["sent_count"] == 1
    assert row["chat_id"] == "-100123"
    assert fake_api.sent_requests == []
    assert fake_api.chat_actions == []
    assert "force_retrieval" not in captured


def test_process_update_attaches_same_user_recent_context_and_writes_memory():
    fake_api = _FakeAPI()
    memory = _FakeMemory(
        recent=[
            {
                "role": "user",
                "content": "我刚才问的是 CKB 是什么",
                "created_ts_ms": 1000,
                "user_id": "42",
                "channel_id": "-100123",
                "thread_id": None,
            },
            {
                "role": "assistant",
                "content": "CKB 是 Nervos 的底层公链。",
                "created_ts_ms": 1100,
                "user_id": "42",
                "channel_id": "-100123",
                "thread_id": None,
            },
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        memory_service=memory,
        memory_context_limit=20,
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot 你看看上文是什么"), dry_run=False)

    assert row["ignored"] is False
    assert memory.read_calls[0]["platform"] == "telegram"
    assert memory.read_calls[0]["user_id"] == "42"
    assert memory.read_calls[0]["channel_id"] == "-100123"
    assert memory.read_calls[0]["limit"] == 20
    assert captured["recent_messages"] == memory.recent
    assert "我刚才问的是 CKB 是什么" in captured["conversation_context"]
    assert len(memory.write_calls) == 2
    assert memory.write_calls[0]["role"] == "user"
    assert memory.write_calls[0]["content"] == "你看看上文是什么"
    assert memory.write_calls[1]["role"] == "assistant"
    assert memory.write_calls[1]["content"] == "ok"
    assert memory.write_calls[1]["created_ts_ms"] >= memory.write_calls[0]["created_ts_ms"]


def test_process_update_skips_recent_context_for_standalone_named_question():
    fake_api = _FakeAPI()
    memory = _FakeMemory(
        recent=[
            {
                "role": "user",
                "content": "那你自己给我写一个完整交易调用代码例子",
                "created_ts_ms": 1000,
                "user_id": "42",
                "channel_id": "-100123",
            },
            {
                "role": "assistant",
                "content": "这里是一段 TypeScript 转账示例代码。",
                "created_ts_ms": 1100,
                "user_id": "42",
                "channel_id": "-100123",
            },
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "CKB 是 Nervos 的 Layer 1。"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        memory_service=memory,
        memory_context_limit=20,
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot ckb是什么"), dry_run=False)

    assert row["ignored"] is False
    assert memory.read_calls == []
    assert captured["recent_messages"] == []
    assert captured["conversation_context"] == ""


def test_process_update_memory_write_failure_does_not_break_reply():
    fake_api = _FakeAPI()
    memory = _FakeMemory(fail_writes=True)

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        memory_service=memory,
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)

    assert row["ignored"] is False
    assert row["sent_count"] == 1
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_process_update_sends_typing_action_before_graph_runner():
    fake_api = _FakeAPI()
    seen_actions_before_runner: list[dict[str, Any]] = []

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        seen_actions_before_runner.extend(fake_api.chat_actions)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)

    assert row["ignored"] is False
    assert seen_actions_before_runner == [{"chat_id": "-100123", "action": "typing"}]
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_process_update_uses_outbound_message_and_sends():
    fake_api = _FakeAPI()

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        context = state["user_message"]["context"]
        return {
            "_outbound_message": {
                "request_id": state["request_id"],
                "context": context,
                "segments": [
                    {
                        "segment_id": f"{state['request_id']}:0",
                        "index": 0,
                        "text": "part 1",
                        "char_count": 6,
                        "citation_labels": [],
                    },
                    {
                        "segment_id": f"{state['request_id']}:1",
                        "index": 1,
                        "text": "part 2",
                        "char_count": 6,
                        "citation_labels": [],
                    },
                ],
                "render_mode": "markdown",
                "append_csat": False,
            }
        }

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)
    assert row["ignored"] is False
    assert row["sent_count"] == 2
    assert len(fake_api.sent_requests) == 2
    assert fake_api.sent_requests[0]["payload"]["reply_to_message_id"] == 1
    assert "reply_to_message_id" not in fake_api.sent_requests[1]["payload"]


def test_group_plain_message_is_ignored_when_not_mentioned():
    fake_api = _FakeAPI()
    memory = _FakeMemory()
    calls: list[dict[str, Any]] = []

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {},
        memory_service=memory,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="昨天你说的"), dry_run=False)

    assert row["ignored"] is True
    assert row["reason"] == "not_mentioned"
    assert calls == []
    assert fake_api.chat_actions == []
    assert fake_api.sent_requests == []
    assert memory.read_calls == []
    assert memory.write_calls == []


def test_group_mention_processes_and_strips_mention():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot ckb是什么"), dry_run=False)

    assert row["ignored"] is False
    assert captured["user_message"]["content"] == "ckb是什么"
    assert fake_api.chat_actions == [{"chat_id": "-100123", "action": "typing"}]
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_group_text_mention_entity_processes_message():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
    )
    update = _sample_update(text="@NBCKB_Bot ckb是什么")
    update["message"]["entities"] = [
        {
            "offset": 0,
            "length": 10,
            "type": "text_mention",
            "user": {"id": 999, "is_bot": True, "username": "NBCKB_Bot"},
        }
    ]

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    assert captured["user_message"]["content"] == "ckb是什么"
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_process_update_downloads_text_document_into_user_message(tmp_path: Path):
    fake_api = _FakeAPI()
    fake_api.files["file_text_1"] = ("documents/notes.md", b"# CKB notes\nCell model basics")
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
        attachment_download_dir=tmp_path,
    )
    update = _sample_update(text="@NBCKB_Bot 总结这个文件")
    message = update["message"]
    message.pop("text")
    message["caption"] = "@NBCKB_Bot 总结这个文件"
    message["document"] = {
        "file_id": "file_text_1",
        "file_unique_id": "uniq_text_1",
        "file_name": "notes.md",
        "mime_type": "text/markdown",
        "file_size": 29,
    }

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    content = captured["user_message"]["content"]
    assert "总结这个文件" in content
    assert "用户上传的文本文件 `notes.md` 内容" in content
    assert "Cell model basics" in content
    attachment = captured["user_message"]["attachments"][0]
    assert attachment["status"] == "ready"
    assert Path(attachment["local_path"]).is_file()


def test_process_update_downloads_image_attachment_for_llm(tmp_path: Path):
    fake_api = _FakeAPI()
    fake_api.files["photo_1"] = ("photos/image.jpg", b"\xff\xd8\xff\xe0fakejpeg")
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
        attachment_download_dir=tmp_path,
    )
    update = _sample_update(text="@NBCKB_Bot 看看这张图")
    message = update["message"]
    message.pop("text")
    message["caption"] = "@NBCKB_Bot 看看这张图"
    message["photo"] = [
        {"file_id": "photo_small", "file_unique_id": "small", "file_size": 3},
        {"file_id": "photo_1", "file_unique_id": "large", "file_size": 12},
    ]

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    attachment = captured["user_message"]["attachments"][0]
    assert attachment["kind"] == "image"
    assert attachment["status"] == "ready"
    assert Path(attachment["local_path"]).is_file()
    assert "用户上传的图片已作为视觉输入传给模型" in captured["user_message"]["content"]


def test_group_reply_to_bot_processes_without_mention():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="继续说", reply_to_bot=True), dry_run=False)

    assert row["ignored"] is False
    assert captured["user_message"]["content"] == "继续说"
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"
    assert fake_api.sent_requests[-1]["payload"]["reply_to_message_id"] == 1


def test_group_reply_to_bot_injects_replied_message_context():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    update = _sample_update(text="你都没给引用凭什么说有", reply_to_bot=True)
    update["message"]["reply_to_message"]["text"] = "有，CCC 相关的基础概念我可以直接聊。"

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    assert captured["user_message"]["reply_to_message_id"] == "99"
    assert "CCC 相关" in captured["user_message"]["reply_to_content"]
    assert "当前消息正在回复这条 assistant 消息" in captured["conversation_context"]
    assert "CCC 相关" in captured["conversation_context"]


def test_group_reply_to_bot_does_not_mix_unrelated_recent_context():
    fake_api = _FakeAPI()
    memory = _FakeMemory(
        recent=[
            {"role": "user", "content": "Fiber WASM 是怎么实现的？"},
            {"role": "assistant", "content": "WASM 可以放一些轻量验证逻辑。"},
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        memory_service=memory,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    update = _sample_update(text="小白版的解释，你说一下", reply_to_bot=True)
    update["message"]["reply_to_message"]["text"] = "CKB 通常指 Nervos CKB，是 Nervos 生态里的公链基础层。"

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    assert memory.read_calls == []
    assert captured["recent_messages"] == []
    assert "CKB 通常指 Nervos CKB" in captured["conversation_context"]
    assert "Fiber WASM" not in captured["conversation_context"]


def test_group_reply_to_bot_missing_snapshot_does_not_guess_from_recent_context():
    fake_api = _FakeAPI()
    memory = _FakeMemory(
        recent=[
            {"role": "user", "content": "Fiber WASM 是怎么实现的？"},
            {"role": "assistant", "content": "WASM 可以放一些轻量验证逻辑。"},
        ]
    )
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        memory_service=memory,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    update = _sample_update(text="小白版的解释，你说一下", reply_to_bot=True)
    del update["message"]["reply_to_message"]["text"]

    row = gateway.process_update(update, dry_run=False)

    assert row["ignored"] is False
    assert memory.read_calls == []
    assert captured["recent_messages"] == []
    assert "平台没有提供被回复消息内容" in captured["conversation_context"]
    assert "不要从普通历史记录猜测被回复内容" in captured["conversation_context"]
    assert "Fiber WASM" not in captured["conversation_context"]


def test_group_reply_to_bot_outbound_override_still_replies_to_current_user_message():
    fake_api = _FakeAPI()

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        context = state["user_message"]["context"]
        return {
            "_outbound_message": {
                "request_id": state["request_id"],
                "context": context,
                "segments": [
                    {
                        "segment_id": f"{state['request_id']}:0",
                        "index": 0,
                        "text": "ok",
                        "char_count": 2,
                        "citation_labels": [],
                    }
                ],
                "render_mode": "markdown",
                "append_csat": False,
                "reply_to_message_id": "99",
            }
        }

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="继续说", reply_to_bot=True), dry_run=False)

    assert row["ignored"] is False
    assert fake_api.sent_requests[-1]["payload"]["reply_to_message_id"] == 1


def test_process_update_sends_visible_progress_for_slow_graph():
    fake_api = _FakeAPI()

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.04)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        progress_update_config=ProgressUpdateConfig(
            enabled=True,
            first_after_s=0.01,
            interval_s=0.01,
            max_updates=2,
            messages=("还在处理 1", "还在处理 2"),
        ),
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot How to set up a Fiber node?"), dry_run=False)

    assert row["ignored"] is False
    progress_requests = [
        item for item in fake_api.callback_requests
        if item["method"] == "sendMessage" and "checking sources" in str(item["payload"].get("text", ""))
    ]
    assert len(progress_requests) >= 1
    assert progress_requests[0]["payload"]["reply_to_message_id"] == 1
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_process_update_sends_chinese_visible_progress_for_chinese_question():
    fake_api = _FakeAPI()

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.04)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        progress_update_config=ProgressUpdateConfig(
            enabled=True,
            first_after_s=0.01,
            interval_s=0.01,
            max_updates=1,
            messages=("还在处理",),
        ),
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot CKB 是什么？"), dry_run=False)

    assert row["ignored"] is False
    progress_requests = [
        item for item in fake_api.callback_requests
        if item["method"] == "sendMessage" and item["payload"].get("text") == "还在处理"
    ]
    assert len(progress_requests) == 1


def test_process_update_does_not_send_visible_progress_for_fast_graph():
    fake_api = _FakeAPI()
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        progress_update_config=ProgressUpdateConfig(
            enabled=True,
            first_after_s=0.05,
            interval_s=0.01,
            max_updates=1,
            messages=("还在处理",),
        ),
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)

    assert row["ignored"] is False
    assert [
        item for item in fake_api.callback_requests
        if item["method"] == "sendMessage" and item["payload"].get("text") == "还在处理"
    ] == []
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_process_update_writes_debug_event(tmp_path: Path):
    fake_api = _FakeAPI()
    debug_file = tmp_path / "debug_events.jsonl"

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "_route_decision": "has_needs",
            "retrieval_policy": "single",
            "_tool_execution_summary": "tools=s1:github_search=ok:2",
            "_tool_execution_trace": [
                {
                    "step_id": "s1",
                    "tool": "qdrant_search",
                    "status": "ok",
                    "evidence_count": 2,
                    "filter_notes": ["mapped_source:docs->github_docs"],
                }
            ],
            "_graph_elapsed_ms": 1234,
            "_node_timings": [{"node": "info_gap_assessor", "elapsed_ms": 100}],
            "_llm_usage_summary": {
                "calls": 2,
                "elapsed_ms": 900,
                "input_tokens": 120,
                "output_tokens": 80,
                "total_tokens": 200,
                "by_node": {"info_gap_assessor": {"calls": 2, "elapsed_ms": 900}},
            },
            "_llm_trace": [
                {
                    "node": "info_gap_assessor",
                    "kind": "business_json",
                    "model": "openai/gpt-5.4-mini",
                    "tier": "low",
                    "elapsed_ms": 900,
                    "usage": {"input_tokens": 120, "output_tokens": 80, "total_tokens": 200},
                }
            ],
            "_ask_user_guard_reason": "ask_user_without_required_info",
            "evidence": [
                {"id": "ev-1", "payload": {"source": "github_docs", "backend": "retrieval"}},
                {"id": "ev-2", "payload": {"source": "github_code", "backend": "retrieval_github_code"}},
            ],
            "_final_response": {
                "request_id": state["request_id"],
                "text": "answer [1]",
                "citations": [{"label": "[1]", "url": "https://example.com"}],
            },
        }

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        debug_log_file=debug_file,
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot ckb是什么"), dry_run=False)

    assert row["ignored"] is False
    rows = [json.loads(line) for line in debug_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    event = rows[0]
    assert event["request_id"] == row["request_id"]
    assert event["route_decision"] == "has_needs"
    assert event["retrieval_policy"] == "single"
    assert event["tool_summary"] == "tools=s1:github_search=ok:2"
    assert event["tool_trace"][0]["filter_notes"] == ["mapped_source:docs->github_docs"]
    assert {"source": "github_docs", "backend": "retrieval", "count": 1} in event["evidence_sources"]
    assert {"source": "github_code", "backend": "retrieval_github_code", "count": 1} in event["evidence_sources"]
    assert event["evidence_count"] == 2
    assert event["citation_count"] == 1
    assert event["outbound_reply_to_message_id"] == "1"
    assert event["first_send_reply_to_message_id"] == "1"
    assert event["sent_message_ids"] == ["9000"]
    assert event["send_results"][0]["message_id"] == "9000"
    assert isinstance(event["graph_elapsed_ms"], int)
    assert event["graph_elapsed_ms"] >= 0
    assert event["node_timings"] == [{"node": "info_gap_assessor", "elapsed_ms": 100}]
    assert event["llm_usage_summary"]["calls"] == 2
    assert event["llm_usage_summary"]["total_tokens"] == 200
    assert event["llm_trace"][0]["model"] == "openai/gpt-5.4-mini"
    assert event["time_budget"]["target_elapsed_ms"] == 30000
    assert event["ask_user_guard_reason"] == "ask_user_without_required_info"


def test_group_reply_to_non_bot_is_ignored_without_mention():
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {},
        bot_user_id="999",
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(
        _sample_update(text="我回复别人", reply_to_user_id=7),
        dry_run=False,
    )

    assert row["ignored"] is True
    assert row["reason"] == "not_mentioned"
    assert calls == []
    assert fake_api.chat_actions == []
    assert fake_api.sent_requests == []


def test_private_chat_processes_without_mention():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(
        _sample_update(chat_id=42, chat_type="private", text="hello"),
        dry_run=False,
    )

    assert row["ignored"] is False
    assert captured["user_message"]["content"] == "hello"
    assert fake_api.sent_requests[-1]["payload"]["text"] == "ok"


def test_group_bot_command_processes_without_mention():
    fake_api = _FakeAPI()
    captured: dict[str, Any] = {}

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="/ask@NBCKB_Bot fiber open channel"), dry_run=False)

    assert row["ignored"] is False
    assert captured["user_message"]["kind"] == "command"
    assert captured["user_message"]["command"] == "/ask"
    assert captured["user_message"]["command_args"] == "fiber open channel"


def test_group_command_for_other_bot_is_ignored():
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {},
        bot_username="NBCKB_Bot",
    )

    row = gateway.process_update(_sample_update(text="/ask@OtherBot fiber open channel"), dry_run=False)

    assert row["ignored"] is True
    assert row["reason"] == "not_mentioned"
    assert calls == []
    assert fake_api.chat_actions == []
    assert fake_api.sent_requests == []


def test_fast_command_sets_one_shot_priority_for_next_telegram_request(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FastModeStateStore(tmp_path / "fast.json")
    captured: list[dict[str, Any]] = []

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(state))
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
        fast_mode_store=store,
    )

    command_row = gateway.process_update(_sample_update(text="/fast@NBCKB_Bot"), dry_run=False)
    first_row = gateway.process_update(_sample_update(update_id=101, text="@NBCKB_Bot ckb是什么"), dry_run=False)
    second_row = gateway.process_update(_sample_update(update_id=102, text="@NBCKB_Bot fiber是什么"), dry_run=False)

    assert command_row["reason"] == "fast_command"
    assert command_row["fast_action"] == "enable"
    assert "已开启 fast 模式" in fake_api.sent_requests[0]["payload"]["text"]
    assert "更快" in fake_api.sent_requests[0]["payload"]["text"]
    assert first_row["ignored"] is False
    assert second_row["ignored"] is False
    assert len(captured) == 2
    assert captured[0]["_llm_service_tier"] == "priority"
    assert "_llm_service_tier" not in captured[1]
    assert store.is_pending(platform="telegram", user_id="42") is False


def test_fast_command_is_user_scoped_for_telegram(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FastModeStateStore(tmp_path / "fast.json")
    captured: list[dict[str, Any]] = []

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(state))
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        bot_username="NBCKB_Bot",
        fast_mode_store=store,
    )

    gateway.process_update(_sample_update(text="/fast@NBCKB_Bot", user_id=42), dry_run=False)
    gateway.process_update(_sample_update(update_id=101, text="@NBCKB_Bot ckb是什么", user_id=7), dry_run=False)
    gateway.process_update(_sample_update(update_id=102, text="@NBCKB_Bot ckb是什么", user_id=42), dry_run=False)

    assert "_llm_service_tier" not in captured[0]
    assert captured[1]["_llm_service_tier"] == "priority"


def test_fast_off_and_status_do_not_enter_telegram_graph(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FastModeStateStore(tmp_path / "fast.json")
    calls: list[dict[str, Any]] = []
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {},
        bot_username="NBCKB_Bot",
        fast_mode_store=store,
    )

    gateway.process_update(_sample_update(text="/fast@NBCKB_Bot"), dry_run=False)
    status_row = gateway.process_update(_sample_update(update_id=101, text="/fast@NBCKB_Bot status"), dry_run=False)
    off_row = gateway.process_update(_sample_update(update_id=102, text="/fast@NBCKB_Bot off"), dry_run=False)

    assert status_row["reason"] == "fast_command"
    assert off_row["fast_action"] == "disable"
    assert calls == []
    assert "已待命" in fake_api.sent_requests[-2]["payload"]["text"]
    assert "已取消 fast 模式" in fake_api.sent_requests[-1]["payload"]["text"]
    assert store.is_pending(platform="telegram", user_id="42") is False


def test_fast_command_for_other_telegram_bot_is_ignored(tmp_path: Path):
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []
    store = FastModeStateStore(tmp_path / "fast.json")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {},
        bot_username="NBCKB_Bot",
        fast_mode_store=store,
    )

    row = gateway.process_update(_sample_update(text="/fast@OtherBot"), dry_run=False)

    assert row["ignored"] is True
    assert row["reason"] == "not_mentioned"
    assert calls == []
    assert fake_api.sent_requests == []
    assert store.is_pending(platform="telegram", user_id="42") is False


def test_process_update_ignores_non_message_update():
    fake_api = _FakeAPI()
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
    )
    row = gateway.process_update({"update_id": 9, "inline_query": {"id": "x"}})
    assert row["ignored"] is True
    assert row["reason"] == "unsupported_update"


def test_process_update_ignores_bot_sender():
    fake_api = _FakeAPI()
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello", is_bot=True))
    assert row["ignored"] is True
    assert row["reason"] == "bot_sender"


def test_allowed_chat_ids_filter():
    fake_api = _FakeAPI()
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        allowed_chat_ids={"-100999"},
    )
    row = gateway.process_update(_sample_update(chat_id=-100123, text="@NBCKB_Bot hello"), dry_run=True)
    assert row["ignored"] is True
    assert row["reason"] == "chat_not_allowed"


def test_allowed_thread_ids_filter_blocks_unlisted_topic():
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        allowed_chat_ids={"-100123"},
        allowed_thread_ids={"224514", "295370"},
    )

    row = gateway.process_update(
        _sample_update(chat_id=-100123, thread_id=1, text="@NBCKB_Bot hello"),
        dry_run=True,
    )

    assert row["ignored"] is True
    assert row["reason"] == "thread_not_allowed"
    assert row["thread_id"] == "1"
    assert calls == []


def test_allowed_thread_ids_filter_allows_configured_topics():
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        allowed_chat_ids={"-100123"},
        allowed_thread_ids={"224514", "295370"},
    )

    dev_row = gateway.process_update(
        _sample_update(chat_id=-100123, thread_id=224514, text="@NBCKB_Bot hello"),
        dry_run=True,
    )
    marketing_row = gateway.process_update(
        _sample_update(update_id=101, chat_id=-100123, thread_id=295370, text="@NBCKB_Bot hello"),
        dry_run=True,
    )

    assert dev_row["ignored"] is False
    assert marketing_row["ignored"] is False
    assert len(calls) == 2
    assert calls[0]["user_message"]["context"]["thread_id"] == "224514"
    assert calls[1]["user_message"]["context"]["thread_id"] == "295370"


def test_allowed_thread_ids_by_chat_only_restricts_configured_chat():
    fake_api = _FakeAPI()
    calls: list[dict[str, Any]] = []
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: calls.append(state) or {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        allowed_chat_ids={"-100123", "-100999"},
        allowed_thread_ids_by_chat={"-100123": {"224514", "295370"}},
    )

    blocked_row = gateway.process_update(
        _sample_update(chat_id=-100123, thread_id=1, text="@NBCKB_Bot hello"),
        dry_run=True,
    )
    allowed_row = gateway.process_update(
        _sample_update(update_id=101, chat_id=-100999, text="@NBCKB_Bot hello"),
        dry_run=True,
    )
    topic_row = gateway.process_update(
        _sample_update(update_id=102, chat_id=-100123, thread_id=224514, text="@NBCKB_Bot hello"),
        dry_run=True,
    )

    assert blocked_row["ignored"] is True
    assert blocked_row["reason"] == "thread_not_allowed"
    assert allowed_row["ignored"] is False
    assert topic_row["ignored"] is False
    assert len(calls) == 2
    assert calls[0]["user_message"]["context"]["channel_id"] == "-100999"
    assert calls[1]["user_message"]["context"]["thread_id"] == "224514"


def test_poll_once_advances_and_persists_offset(tmp_path: Path):
    updates = [
        _sample_update(update_id=201, text="@NBCKB_Bot hello"),
        _sample_update(update_id=202, text="@NBCKB_Bot hi"),
    ]
    fake_api = _FakeAPI(updates=updates)
    store = TelegramUpdateOffsetStore(tmp_path / "tg_offset.txt")

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        offset_store=store,
    )

    rows = gateway.poll_once(dry_run=True, timeout_s=0, limit=10)
    assert len(rows) == 2
    assert gateway.next_offset == 203
    assert store.load() == 203


def test_poll_once_processes_different_users_concurrently(tmp_path: Path):
    updates = [
        _sample_update(update_id=210, chat_id=-100123, user_id=1, text="@NBCKB_Bot first"),
        _sample_update(update_id=211, chat_id=-100123, user_id=2, text="@NBCKB_Bot second"),
    ]
    fake_api = _FakeAPI(updates=updates)
    store = TelegramUpdateOffsetStore(tmp_path / "tg_offset.txt")
    release = threading.Event()
    started: list[int] = []
    lock = threading.Lock()

    def graph_runner(state: dict[str, Any]) -> dict[str, Any]:
        context = state["user_message"]["context"]
        with lock:
            started.append(int(context["user_id"]))
            if len(started) == 2:
                release.set()
        assert release.wait(1.0)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=graph_runner,
        offset_store=store,
        max_worker_threads=2,
    )

    rows = gateway.poll_once(dry_run=True, timeout_s=0, limit=10)

    assert [row["update_id"] for row in rows] == [210, 211]
    assert sorted(started) == [1, 2]
    assert gateway.next_offset == 212
    assert store.load() == 212


def test_poll_once_preserves_same_chat_user_order():
    updates = [
        _sample_update(update_id=220, chat_id=-100123, user_id=7, text="@NBCKB_Bot first"),
        _sample_update(update_id=221, chat_id=-100123, user_id=7, text="@NBCKB_Bot second"),
    ]
    fake_api = _FakeAPI(updates=updates)
    seen: list[str] = []

    def graph_runner(state: dict[str, Any]) -> dict[str, Any]:
        content = str(state["user_message"]["content"])
        if "first" in content:
            time.sleep(0.05)
        seen.append(content)
        return {"_final_response": {"request_id": state["request_id"], "text": "ok"}}

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=graph_runner,
        max_worker_threads=4,
    )

    rows = gateway.poll_once(dry_run=True, timeout_s=0, limit=10)

    assert [row["update_id"] for row in rows] == [220, 221]
    assert seen == ["first", "second"]


def test_send_requests_retries_markdown_failure_as_plain_text():
    api = _FlakyTelegramAPI(fail_count=1)
    sent = api.send_requests(
        [
            {
                "method": "sendMessage",
                "payload": {
                    "chat_id": -100123,
                    "text": "*bad markdown",
                    "parse_mode": "MarkdownV2",
                    "reply_to_message_id": 5,
                },
            }
        ]
    )

    assert sent == 1
    assert len(api.send_calls) == 2
    assert api.send_calls[0]["payload"]["parse_mode"] == "MarkdownV2"
    assert "parse_mode" not in api.send_calls[1]["payload"]
    assert api.send_calls[1]["payload"]["reply_to_message_id"] == 5


def test_send_requests_retries_entities_failure_as_plain_text():
    api = _FlakyTelegramAPI(fail_count=1)
    sent = api.send_requests(
        [
            {
                "method": "sendMessage",
                "payload": {
                    "chat_id": -100123,
                    "text": "bad entities",
                    "entities": [{"type": "bold", "offset": 0, "length": 3}],
                    "reply_to_message_id": 5,
                },
            }
        ]
    )

    assert sent == 1
    assert len(api.send_calls) == 2
    assert "entities" in api.send_calls[0]["payload"]
    assert "entities" not in api.send_calls[1]["payload"]
    assert "parse_mode" not in api.send_calls[1]["payload"]
    assert api.send_calls[1]["payload"]["reply_to_message_id"] == 5


def test_send_requests_retries_reply_failure_as_detached_plain_text():
    api = _FlakyTelegramAPI(fail_count=2)
    sent = api.send_requests(
        [
            {
                "method": "sendMessage",
                "payload": {
                    "chat_id": -100123,
                    "text": "*bad markdown",
                    "parse_mode": "MarkdownV2",
                    "reply_to_message_id": 5,
                },
            }
        ]
    )

    assert sent == 1
    assert len(api.send_calls) == 3
    assert "parse_mode" not in api.send_calls[2]["payload"]
    assert "reply_to_message_id" not in api.send_calls[2]["payload"]


def test_poll_once_advances_offset_when_send_fails(tmp_path: Path):
    fake_api = _AlwaysFailSendAPI([_sample_update(update_id=301, text="@NBCKB_Bot hello")])
    store = TelegramUpdateOffsetStore(tmp_path / "tg_offset.txt")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {"_final_response": {"request_id": state["request_id"], "text": "ok"}},
        offset_store=store,
    )

    rows = gateway.poll_once(dry_run=False, timeout_s=0, limit=10)

    assert rows[0]["reason"] == "process_update_failed"
    assert gateway.next_offset == 302
    assert store.load() == 302


def test_process_update_runner_error_returns_fallback_message():
    fake_api = _FakeAPI()

    def boom(_state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=boom,
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)
    assert row["ignored"] is False
    assert row["sent_count"] == 1
    assert "处理请求时发生错误" in fake_api.sent_requests[0]["payload"]["text"]


def test_process_update_splits_long_response():
    fake_api = _FakeAPI()
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {
            "_final_response": {
                "request_id": state["request_id"],
                "text": "A" * 9000,
                "citations": [],
            }
        },
    )
    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=True)
    assert row["ignored"] is False
    assert row["sent_count"] >= 3


def test_process_update_records_answer_metadata_for_feedback(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")

    def runner(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "_final_response": {
                "request_id": state["request_id"],
                "text": "answer [1]",
                "citations": [{"label": "[1]", "url": "https://example.com"}],
                "trace_summary": "tools=s1:qdrant_search=ok:1",
            },
            "_tool_calls_executed": 1,
            "evidence": [{"id": "e1"}],
        }

    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=runner,
        append_csat=True,
        feedback_store=store,
    )

    row = gateway.process_update(_sample_update(text="@NBCKB_Bot hello"), dry_run=False)

    assert row["ignored"] is False
    records = store.iter_records()
    assert len(records) == 1
    assert records[0]["kind"] == "answer"
    assert records[0]["has_csat"] is True
    assert records[0]["tool_calls"] == 1
    assert records[0]["evidence_count"] == 1
    assert "reply_markup" in fake_api.sent_requests[-1]["payload"]


def test_process_callback_query_writes_feedback_and_badcase(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    store.append_answer(
        {
            "request_id": "tg-req-1",
            "chat_id": "-100123",
            "user_id": "42",
            "trace_summary": "tools=s1:qdrant_search=ok:2",
            "tool_calls": 1,
            "evidence_count": 2,
            "final_text_preview": "joined answer",
        }
    )
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        feedback_store=store,
    )

    row = gateway.process_update(_sample_callback_update(score=2), dry_run=False)

    assert row["ignored"] is False
    assert row["reason"] == "csat"
    assert row["score"] == 2
    assert row["is_bad_case"] is True
    records = store.iter_records()
    assert len(records) == 2
    rating = records[1]
    assert rating["kind"] == "csat"
    assert rating["request_id"] == "tg-req-1"
    assert rating["trace_summary"] == "tools=s1:qdrant_search=ok:2"
    assert rating["tool_calls"] == 1
    assert rating["evidence_count"] == 2
    assert fake_api.callback_requests[0]["method"] == "answerCallbackQuery"


def test_process_callback_query_invalid_payload_is_ignored_safely(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        feedback_store=store,
    )

    row = gateway.process_update(_sample_callback_update(data="bad"), dry_run=False)

    assert row["ignored"] is True
    assert row["reason"] == "invalid_callback"
    assert store.iter_records() == []
    assert fake_api.callback_requests[0]["payload"]["text"] == "Invalid rating."


def test_build_runtime_defaults_to_dynamic_provider_registry(monkeypatch, tmp_path: Path):
    from nervos_brain.graph_engine.provider_registry import ProviderCapabilityRegistry

    runner = _load_script_module("run_telegram_bot_polling_under_test", "run_telegram_bot_polling.py")

    class FakeRetriever:
        backend_names = ["fake"]

    monkeypatch.setattr(runner, "build_configured_retriever", lambda: FakeRetriever())

    runtime, memory_service = runner._build_runtime(model="", memory_db=tmp_path / "memory.db")

    assert isinstance(runtime.provider_registry, ProviderCapabilityRegistry)
    assert runtime.provider_registry.get_profile_for("general", tier="router")["tier"] == "router"
    low = runtime.provider_registry.get_profile_for("planning", tier="low")
    assert low["tier"] == "low"
    assert low["model"] == "openai/gpt-5.6-luna"
    assert low["reasoning_effort"] == "low"
    assert memory_service is not None


def test_build_runtime_model_argument_forces_fixed_registry(monkeypatch, tmp_path: Path):
    runner = _load_script_module("run_telegram_bot_polling_under_test_override", "run_telegram_bot_polling.py")

    class FakeRetriever:
        backend_names = ["fake"]

    monkeypatch.setattr(runner, "build_configured_retriever", lambda: FakeRetriever())

    runtime, _memory_service = runner._build_runtime(
        model="openai/gpt-5.5",
        memory_db=tmp_path / "memory.db",
    )

    profile = runtime.provider_registry.get_profile_for("composing", tier="low")
    assert profile["model"] == "openai/gpt-5.5"
    assert profile["reasoning_effort"] == ""


def test_process_callback_query_allowed_chat_filter_blocks_non_test_group(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        allowed_chat_ids={"-100999"},
        feedback_store=store,
    )

    row = gateway.process_update(_sample_callback_update(chat_id=-100123), dry_run=False)

    assert row["ignored"] is True
    assert row["reason"] == "chat_not_allowed"
    assert store.iter_records() == []
    assert fake_api.callback_requests == []


def test_process_callback_query_allowed_thread_filter_blocks_unlisted_topic(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        allowed_chat_ids={"-100123"},
        allowed_thread_ids={"224514", "295370"},
        feedback_store=store,
    )

    row = gateway.process_update(
        _sample_callback_update(chat_id=-100123, thread_id=1),
        dry_run=False,
    )

    assert row["ignored"] is True
    assert row["reason"] == "thread_not_allowed"
    assert row["thread_id"] == "1"
    assert store.iter_records() == []
    assert fake_api.callback_requests == []


def test_process_callback_query_allowed_thread_by_chat_allows_unconfigured_chat(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        allowed_chat_ids={"-100123", "-100999"},
        allowed_thread_ids_by_chat={"-100123": {"224514", "295370"}},
        feedback_store=store,
    )

    blocked_row = gateway.process_update(
        _sample_callback_update(chat_id=-100123, thread_id=1),
        dry_run=False,
    )
    allowed_row = gateway.process_update(
        _sample_callback_update(update_id=201, chat_id=-100999),
        dry_run=False,
    )

    assert blocked_row["ignored"] is True
    assert blocked_row["reason"] == "thread_not_allowed"
    assert allowed_row["ignored"] is False
    assert allowed_row["reason"] == "csat"
    assert len(store.iter_records()) == 1
    assert len(fake_api.callback_requests) == 1


def test_process_callback_query_dry_run_does_not_write_feedback(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        feedback_store=store,
    )

    row = gateway.process_update(_sample_callback_update(score=5), dry_run=True)

    assert row["ignored"] is False
    assert row["score"] == 5
    assert store.iter_records() == []
    assert fake_api.callback_requests == []


def test_process_callback_query_duplicate_rating_is_idempotent(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        feedback_store=store,
    )

    first = gateway.process_update(_sample_callback_update(score=4), dry_run=False)
    second = gateway.process_update(_sample_callback_update(score=1), dry_run=False)

    assert first["is_duplicate_rating"] is False
    assert second["is_duplicate_rating"] is True
    ratings = [row for row in store.iter_records() if row["kind"] == "csat"]
    assert len(ratings) == 1
    assert ratings[0]["score"] == 4


def test_feedback_command_records_comment(tmp_path: Path):
    fake_api = _FakeAPI()
    store = FeedbackJsonlStore(tmp_path / "feedback.jsonl")
    store.append_answer(
        {
            "request_id": "tg-req-1",
            "trace_summary": "tools=s1:github_search=ok:1",
            "tool_calls": 1,
            "evidence_count": 1,
            "final_text_preview": "answer",
        }
    )
    gateway = TelegramPollingGateway(
        api=fake_api,  # type: ignore[arg-type]
        graph_runner=lambda state: {},
        feedback_store=store,
    )

    row = gateway.process_update(
        _sample_update(text="/feedback tg-req-1 citation missing"),
        dry_run=False,
    )

    assert row["ignored"] is False
    assert row["reason"] == "feedback"
    comments = [record for record in store.iter_records() if record["kind"] == "comment"]
    assert len(comments) == 1
    assert comments[0]["comment"] == "citation missing"
    assert comments[0]["trace_summary"] == "tools=s1:github_search=ok:1"
    assert fake_api.sent_requests[-1]["payload"]["text"] == "Feedback recorded for tg-req-1."


def test_telegram_runner_config_loads_when_cwd_differs(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("telegram_bot:\n  memory_db: data/telegram_bot/memory.db\n", encoding="utf-8")
    monkeypatch.setenv("NERVOS_BRAIN_CONFIG", str(cfg_path))
    from nervos_brain import pathing
    pathing.config_path.cache_clear()
    pathing.load_project_config.cache_clear()
    runner = _load_script_module("run_telegram_bot_pathing_config", "run_telegram_bot_polling.py")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    cfg = runner._load_telegram_bot_cfg()

    assert cfg["memory_db"] == "data/telegram_bot/memory.db"


def test_telegram_runner_resolves_runtime_paths_to_project_root(monkeypatch, tmp_path):
    runner = _load_script_module("run_telegram_bot_pathing_runtime", "run_telegram_bot_polling.py")
    monkeypatch.chdir(tmp_path)

    resolved = runner.resolve_project_path("data/telegram_bot/memory.db")

    assert resolved == project_root() / "data" / "telegram_bot" / "memory.db"
