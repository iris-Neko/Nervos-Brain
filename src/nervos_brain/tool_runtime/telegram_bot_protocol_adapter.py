"""Telegram Bot API protocol adapters (no bot-token network required).

Scope:
  - Convert raw Telegram Bot API updates to MessageEnvelope
  - Build GraphState-like payload from MessageEnvelope
  - Convert OutboundMessage to Telegram sendMessage request payloads
"""

from __future__ import annotations

import time
import re
from typing import Any

from nervos_brain.core_protocols.message_protocols import MessageEnvelope, OutboundMessage
from nervos_brain.graph_engine.product_policy import load_product_policy

from .feedback import build_csat_callback_data
def telegram_update_to_message_envelope(
    update: dict[str, Any],
    *,
    default_locale: str | None = None,
) -> MessageEnvelope:
    """Parse Telegram Bot API update into platform-neutral MessageEnvelope."""
    msg = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
    )
    if not isinstance(msg, dict):
        raise ValueError("unsupported Telegram update: missing message payload")

    chat = msg.get("chat", {})
    sender = msg.get("from", {})
    if not isinstance(chat, dict):
        chat = {}
    if not isinstance(sender, dict):
        sender = {}

    text = _strip_bot_mentions(_extract_text(msg))
    command, command_args = _parse_command(text)

    ts_s = msg.get("date")
    if isinstance(ts_s, int):
        ts_ms = ts_s * 1000
    else:
        ts_ms = int(time.time() * 1000)

    chat_id = str(chat.get("id", ""))
    user_id = str(sender.get("id", "") or chat.get("id", "unknown_user"))

    context: dict[str, Any] = {
        "platform": "telegram",
        "user_id": user_id,
    }

    if chat_id:
        context["channel_id"] = chat_id

    chat_type = str(chat.get("type", "")).lower()
    if chat_type in {"group", "supergroup", "channel"} and chat_id:
        context["guild_id"] = chat_id

    thread_id = msg.get("message_thread_id")
    if isinstance(thread_id, int):
        context["thread_id"] = str(thread_id)

    message: MessageEnvelope = {
        "kind": "command" if command else "message",
        "ts_ms": ts_ms,
        "message_id": str(msg.get("message_id", update.get("update_id", "unknown"))),
        "context": context,  # type: ignore[typeddict-item]
        "content": text,
    }

    reply_to = msg.get("reply_to_message", {})
    if isinstance(reply_to, dict) and reply_to.get("message_id") is not None:
        message["reply_to_message_id"] = str(reply_to["message_id"])
        reply_text = _strip_bot_mentions(_extract_text(reply_to))
        if reply_text:
            message["reply_to_content"] = _truncate_reply_content(reply_text)
        reply_sender = reply_to.get("from", {})
        if isinstance(reply_sender, dict):
            message["reply_to_role"] = "assistant" if bool(reply_sender.get("is_bot")) else "user"

    attachments = _extract_attachments(msg)
    if attachments:
        message["attachments"] = attachments

    if command:
        message["command"] = command
        if command_args:
            message["command_args"] = command_args

    # This is only a weak platform preference. Semantic output language is
    # resolved by TurnInterpreter from the current request.
    locale = sender.get("language_code")
    policy_default = default_locale or load_product_policy().language.default_locale
    message["platform_locale_hint"] = (
        locale.strip() if isinstance(locale, str) and locale.strip() else policy_default
    )

    return message


def message_envelope_to_graph_state(
    message: MessageEnvelope,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal full-graph compatible state dict from MessageEnvelope."""
    context = dict(message.get("context", {}))
    now_req = request_id or f"tg-{int(time.time() * 1000)}"
    policy = load_product_policy()
    locale = policy.language.default_locale

    user_key = {
        "platform": "telegram",
        "user_id": str(context.get("user_id", "unknown_user")),
    }
    state: dict[str, Any] = {
        "request_id": now_req,
        "user_message": message,
        "user_memory_key": user_key,
        "memory_pointers": [],
        "memory_facts": [],
        "info_needs": [],
        "evidence": [],
        "conflicts": [],
        "retry_count": 0,
        "budget": {
            "max_prompt_tokens": 4000,
            "max_evidence_chunks": 10,
            "max_memory_facts": 5,
            "max_tool_calls": 3,
        },
        "route": "graph",
        "locale": locale,
        "platform_locale_hint": str(message.get("platform_locale_hint", "") or ""),
        "_product_policy": policy,
    }

    guild_id = context.get("guild_id")
    channel_id = context.get("channel_id")
    thread_id = context.get("thread_id")
    if guild_id and channel_id:
        state["channel_memory_key"] = {
            "platform": "telegram",
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
        }
    if guild_id and channel_id and thread_id:
        state["thread_key"] = {
            "platform": "telegram",
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
            "thread_id": str(thread_id),
        }

    return state


def outbound_message_to_telegram_requests(
    outbound: OutboundMessage,
) -> list[dict[str, Any]]:
    """Convert OutboundMessage into Telegram Bot API sendMessage payloads.

    These payloads are ready to POST to:
      https://api.telegram.org/bot<token>/sendMessage
    """
    context = outbound.get("context", {})
    chat_id = context.get("channel_id") or context.get("guild_id")
    if not chat_id:
        raise ValueError("OutboundMessage.context missing channel_id/guild_id")

    render_mode = outbound.get("render_mode", "markdown")
    reply_to = outbound.get("reply_to_message_id")
    thread_id = context.get("thread_id")

    requests: list[dict[str, Any]] = []
    segments = outbound.get("segments", [])
    for idx, segment in enumerate(segments):
        payload: dict[str, Any] = {
            "chat_id": _coerce_numeric_or_keep(chat_id),
            "text": str(segment.get("text", "")),
            "disable_web_page_preview": True,
        }
        entities = segment.get("entities")
        if isinstance(entities, list) and entities:
            payload["entities"] = entities
        elif render_mode == "markdown" and segment.get("parse_mode_enabled", True):
            payload["parse_mode"] = "MarkdownV2"
        if idx == 0 and reply_to:
            payload["reply_to_message_id"] = _coerce_numeric_or_keep(reply_to)
        if thread_id is not None:
            payload["message_thread_id"] = _coerce_numeric_or_keep(thread_id)
        if outbound.get("append_csat") and idx == len(segments) - 1:
            payload["reply_markup"] = _build_csat_reply_markup(str(outbound.get("request_id", "")))

        requests.append({"method": "sendMessage", "payload": payload})
    return requests


def _extract_text(msg: dict[str, Any]) -> str:
    text = msg.get("text")
    if isinstance(text, str):
        return text
    caption = msg.get("caption")
    if isinstance(caption, str):
        return caption
    return ""


def _strip_bot_mentions(text: str) -> str:
    cleaned = re.sub(r"@\w+_Bot\b", "", str(text), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _truncate_reply_content(text: str, limit: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _parse_command(text: str) -> tuple[str | None, str | None]:
    raw = text.strip()
    if not raw.startswith("/") or len(raw) < 2:
        return None, None
    parts = raw.split(maxsplit=1)
    command = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    return command, args or None


def _extract_attachments(msg: dict[str, Any]) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []

    photos = msg.get("photo")
    if isinstance(photos, list) and photos:
        largest = photos[-1]
        if isinstance(largest, dict) and largest.get("file_id"):
            attachments.append(
                {
                    "kind": "image",
                    "url": f"tgfile://{largest['file_id']}",
                    "name": str(largest.get("file_unique_id", "photo")),
                    "file_id": str(largest["file_id"]),
                    "file_unique_id": str(largest.get("file_unique_id", "")),
                    "file_size": str(largest.get("file_size", "")),
                }
            )

    for key in ("document", "video", "audio", "voice", "animation"):
        media = msg.get(key)
        if isinstance(media, dict) and media.get("file_id"):
            mime_type = str(media.get("mime_type", "") or "")
            kind = "image" if mime_type.startswith("image/") else "file"
            attachments.append(
                {
                    "kind": kind,
                    "url": f"tgfile://{media['file_id']}",
                    "name": str(media.get("file_name") or media.get("file_unique_id") or key),
                    "file_id": str(media["file_id"]),
                    "file_unique_id": str(media.get("file_unique_id", "")),
                    "mime_type": mime_type,
                    "file_size": str(media.get("file_size", "")),
                }
            )

    entities = msg.get("entities")
    if isinstance(entities, list):
        text = _extract_text(msg)
        for item in entities:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "url":
                continue
            offset = int(item.get("offset", 0))
            length = int(item.get("length", 0))
            if offset < 0 or length <= 0:
                continue
            url = text[offset : offset + length]
            if url:
                attachments.append({"kind": "link", "url": url})

    return attachments


def _coerce_numeric_or_keep(value: Any) -> Any:
    text = str(value)
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


def _build_csat_reply_markup(request_id: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {
                    "text": str(score),
                    "callback_data": build_csat_callback_data(request_id, score),
                }
                for score in range(1, 6)
            ]
        ]
    }
