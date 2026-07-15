from __future__ import annotations

import json
from pathlib import Path

from nervos_brain.tool_runtime.fast_mode import (
    FastModeStateStore,
    fast_status_message,
    parse_fast_command,
)


def test_fast_mode_store_is_one_shot_and_user_scoped(tmp_path: Path):
    store = FastModeStateStore(tmp_path / "fast.json")

    store.enable_next(platform="telegram", user_id="u1")

    assert store.is_pending(platform="telegram", user_id="u1") is True
    assert store.is_pending(platform="telegram", user_id="u2") is False
    assert store.is_pending(platform="discord", user_id="u1") is False
    assert store.consume_next(platform="telegram", user_id="u1") is True
    assert store.consume_next(platform="telegram", user_id="u1") is False


def test_fast_mode_store_persists_without_chat_content(tmp_path: Path):
    path = tmp_path / "fast.json"
    store = FastModeStateStore(path)

    store.enable_next(platform="Discord", user_id="user-1")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw) == ["discord:user-1"]
    assert raw["discord:user-1"]["pending"] is True
    assert "content" not in json.dumps(raw, ensure_ascii=False).lower()
    assert FastModeStateStore(path).consume_next(platform="discord", user_id="user-1") is True


def test_parse_fast_command_variants():
    assert parse_fast_command("/ask") is None
    fast_command = parse_fast_command("/fast")
    assert fast_command.action == "enable"
    assert "更快" in fast_command.message
    assert "faster" in fast_command.message
    assert parse_fast_command("/fast@NBCKB_Bot").action == "enable"
    assert parse_fast_command("/fast", "off").action == "disable"
    assert parse_fast_command("/fast", "取消").action == "disable"
    assert parse_fast_command("/fast", "status").action == "status"
    assert parse_fast_command("/fast", "what is ckb?").action == "invalid"


def test_fast_status_message():
    assert "已待命" in fast_status_message(pending=True)
    assert "更快" in fast_status_message(pending=True)
    assert "未开启" in fast_status_message(pending=False)
    assert "faster" in fast_status_message(pending=False)
