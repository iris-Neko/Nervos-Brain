"""User-facing progress update helpers for long-running bot requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nervos_brain.graph_engine.product_policy import load_product_policy

_DEFAULT_PROGRESS_MESSAGES = (
    "I am still checking sources and organizing the evidence. Please wait a moment.",
    "I am verifying the relevant material and preparing the answer.",
    "I am still preparing the final answer. Please wait a little longer.",
)

_DEFAULT_PROGRESS_MESSAGES_EN = (
    "I am still checking sources and organizing the evidence. Please wait a moment.",
    "There is quite a bit of material, so I am verifying the sources and shaping the answer.",
    "I am still preparing the final answer. Please wait a little longer.",
)

_DEFAULT_FALLBACK_MESSAGES = {
    "zh-CN": "我还在处理这个问题，请稍等一下。",
    "en": "I am still working on this. Please wait a moment.",
}


@dataclass(frozen=True)
class ProgressUpdateConfig:
    """Config for lightweight user-facing progress messages."""

    enabled: bool = True
    first_after_s: float = 30.0
    interval_s: float = 45.0
    max_updates: int = 3
    messages: tuple[str, ...] = _DEFAULT_PROGRESS_MESSAGES
    messages_by_locale: dict[str, tuple[str, ...]] | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> "ProgressUpdateConfig":
        if not isinstance(raw, dict):
            return cls()
        enabled = _bool_value(raw.get("enabled", True), True)
        first_after_s = _positive_float(raw.get("first_after_s", 30.0), 30.0)
        interval_s = _positive_float(raw.get("interval_s", 45.0), 45.0)
        max_updates = max(0, min(_int_value(raw.get("max_updates", 3), 3), 10))
        messages = _message_tuple(raw.get("messages"))
        messages_by_locale = _messages_by_locale(raw.get("messages_by_locale"))
        return cls(
            enabled=enabled,
            first_after_s=first_after_s,
            interval_s=interval_s,
            max_updates=max_updates,
            messages=messages,
            messages_by_locale=messages_by_locale,
        )

    def message_for(self, index: int, locale: str | None = None) -> str:
        normalized_locale = _normalize_locale(locale)
        messages = self._messages_for_locale(normalized_locale)
        if not messages:
            return _DEFAULT_FALLBACK_MESSAGES.get(normalized_locale, _DEFAULT_FALLBACK_MESSAGES["en"])
        if index < len(messages):
            return messages[index]
        return messages[-1]

    @property
    def should_run(self) -> bool:
        return bool(self.enabled and self.max_updates > 0)

    def _messages_for_locale(self, locale: str) -> tuple[str, ...]:
        localized = self.messages_by_locale or {}
        if locale in localized:
            return localized[locale]
        prefix = locale.split("-", 1)[0]
        for configured_locale, messages in localized.items():
            if configured_locale.split("-", 1)[0] == prefix:
                return messages
        if locale == "en":
            return _DEFAULT_PROGRESS_MESSAGES_EN
        if prefix == "en":
            return _DEFAULT_PROGRESS_MESSAGES_EN
        return self.messages


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


def _messages_by_locale(value: Any) -> dict[str, tuple[str, ...]] | None:
    if not isinstance(value, dict):
        return None
    messages: dict[str, tuple[str, ...]] = {}
    for raw_locale, raw_messages in value.items():
        locale = _normalize_locale(str(raw_locale))
        parsed = _localized_message_tuple(raw_messages)
        if parsed:
            messages[locale] = parsed
    return messages or None


def _localized_message_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _normalize_locale(locale: str | None) -> str:
    default_locale = load_product_policy().language.default_locale
    return str(locale or "").strip().replace("_", "-") or default_locale
