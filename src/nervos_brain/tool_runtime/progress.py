"""User-facing progress update helpers for long-running bot requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DEFAULT_PROGRESS_MESSAGES = (
    "我还在查资料和整理证据，稍等一下。",
    "资料比较多，我正在核对来源和组织回答。",
    "还在生成最终回答，请再等一下。",
)


@dataclass(frozen=True)
class ProgressUpdateConfig:
    """Config for lightweight user-facing progress messages."""

    enabled: bool = True
    first_after_s: float = 30.0
    interval_s: float = 45.0
    max_updates: int = 3
    messages: tuple[str, ...] = _DEFAULT_PROGRESS_MESSAGES

    @classmethod
    def from_mapping(cls, raw: Any) -> "ProgressUpdateConfig":
        if not isinstance(raw, dict):
            return cls()
        enabled = _bool_value(raw.get("enabled", True), True)
        first_after_s = _positive_float(raw.get("first_after_s", 30.0), 30.0)
        interval_s = _positive_float(raw.get("interval_s", 45.0), 45.0)
        max_updates = max(0, min(_int_value(raw.get("max_updates", 3), 3), 10))
        messages = _message_tuple(raw.get("messages"))
        return cls(
            enabled=enabled,
            first_after_s=first_after_s,
            interval_s=interval_s,
            max_updates=max_updates,
            messages=messages,
        )

    def message_for(self, index: int) -> str:
        if not self.messages:
            return "我还在处理这个问题，请稍等一下。"
        if index < len(self.messages):
            return self.messages[index]
        return self.messages[-1]

    @property
    def should_run(self) -> bool:
        return bool(self.enabled and self.max_updates > 0)


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _message_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else _DEFAULT_PROGRESS_MESSAGES
    if isinstance(value, (list, tuple)):
        messages = tuple(str(item).strip() for item in value if str(item).strip())
        return messages or _DEFAULT_PROGRESS_MESSAGES
    return _DEFAULT_PROGRESS_MESSAGES
