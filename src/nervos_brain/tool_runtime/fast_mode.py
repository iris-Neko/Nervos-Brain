"""One-shot user-side fast mode state for bot runtimes."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FastModeCommand:
    action: str
    message: str


class FastModeStateStore:
    """File-backed pending-fast marker keyed by platform and user id."""

    def __init__(self, path: str | Path = "data/runtime/fast_mode_state.json") -> None:
        self._path = Path(path).expanduser()
        self._lock = threading.Lock()

    def enable_next(self, *, platform: str, user_id: str) -> None:
        key = _state_key(platform=platform, user_id=user_id)
        if not key:
            return
        with self._lock:
            data = self._load_unlocked()
            data[key] = {"pending": True, "updated_ts_ms": _now_ms()}
            self._save_unlocked(data)

    def disable(self, *, platform: str, user_id: str) -> None:
        key = _state_key(platform=platform, user_id=user_id)
        if not key:
            return
        with self._lock:
            data = self._load_unlocked()
            data.pop(key, None)
            self._save_unlocked(data)

    def is_pending(self, *, platform: str, user_id: str) -> bool:
        key = _state_key(platform=platform, user_id=user_id)
        if not key:
            return False
        with self._lock:
            data = self._load_unlocked()
            row = data.get(key)
            return isinstance(row, dict) and bool(row.get("pending", False))

    def consume_next(self, *, platform: str, user_id: str) -> bool:
        key = _state_key(platform=platform, user_id=user_id)
        if not key:
            return False
        with self._lock:
            data = self._load_unlocked()
            row = data.pop(key, None)
            self._save_unlocked(data)
            return isinstance(row, dict) and bool(row.get("pending", False))

    def _load_unlocked(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def parse_fast_command(command: str, args: str = "") -> FastModeCommand | None:
    if _base_command(command) != "/fast":
        return None
    normalized_args = " ".join(str(args or "").strip().lower().split())
    if normalized_args in {"", "on"}:
        return FastModeCommand(
            "enable",
            "已开启 fast 模式，下一次请求会尝试使用更快的回复通道。\n"
            "Fast mode is on. Your next request will try to use a faster response tier.",
        )
    if normalized_args in {"off", "disable", "cancel", "关闭", "取消"}:
        return FastModeCommand("disable", "已取消 fast 模式。\nFast mode is off.")
    if normalized_args in {"status", "状态"}:
        return FastModeCommand("status", "")
    return FastModeCommand(
        "invalid",
        "用法：/fast、/fast off、/fast status。/fast 只加速下一次请求。\n"
        "Usage: /fast, /fast off, /fast status. /fast only speeds up your next request.",
    )


def fast_status_message(*, pending: bool) -> str:
    if pending:
        return (
            "fast 模式已待命：下一次请求会尝试使用更快的回复通道。\n"
            "Fast mode is ready: your next request will try to use a faster response tier."
        )
    return (
        "fast 模式未开启。发送 /fast 可让下一次请求尝试更快回复。\n"
        "Fast mode is off. Send /fast to try a faster response for your next request."
    )


def _base_command(command: str) -> str:
    return str(command or "").strip().split("@", 1)[0].lower()


def _state_key(*, platform: str, user_id: str) -> str:
    platform_text = str(platform or "").strip().lower()
    user_text = str(user_id or "").strip()
    if not platform_text or not user_text:
        return ""
    return f"{platform_text}:{user_text}"


def _now_ms() -> int:
    return int(time.time() * 1000)
