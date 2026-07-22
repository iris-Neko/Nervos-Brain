"""Discord protocol adapters (no bot-token network required)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from nervos_brain.core_protocols.message_protocols import MessageEnvelope, OutboundMessage
from nervos_brain.graph_engine.product_policy import load_product_policy

def discord_message_to_message_envelope(
    message: dict[str, Any],
    *,
    default_locale: str | None = None,
) -> MessageEnvelope:
    """Parse Discord message payload into platform-neutral MessageEnvelope."""
    if not isinstance(message, dict):
        raise ValueError("unsupported Discord message: payload must be dict")

    author = message.get("author")
    if not isinstance(author, dict):
        author = {}

    content = str(message.get("content", "") or "")
    command, command_args = _parse_command(content)

    context: dict[str, Any] = {
        "platform": "discord",
        "user_id": str(author.get("id", "unknown_user")),
    }

    channel_id = message.get("channel_id")
    guild_id = message.get("guild_id")
    thread_id = message.get("thread_id")
    if channel_id is not None:
        context["channel_id"] = str(channel_id)
    if guild_id is not None:
        context["guild_id"] = str(guild_id)
    if thread_id is not None:
        context["thread_id"] = str(thread_id)

    envelope: MessageEnvelope = {
        "kind": "command" if command else "message",
        "ts_ms": _extract_ts_ms(message),
        "message_id": str(message.get("id", "unknown")),
        "context": context,  # type: ignore[typeddict-item]
        "content": content,
    }

    reference = message.get("reference")
    if isinstance(reference, dict):
        ref_mid = reference.get("message_id")
        if ref_mid is not None:
            envelope["reply_to_message_id"] = str(ref_mid)
        reply_text = _extract_reply_text(reference)
        if reply_text:
            envelope["reply_to_content"] = _truncate_reply_content(reply_text)
        reply_author = reference.get("author")
        if isinstance(reply_author, dict):
            envelope["reply_to_role"] = "assistant" if bool(reply_author.get("bot")) else "user"

    attachments = _extract_attachments(message)
    if attachments:
        envelope["attachments"] = attachments

    if command:
        envelope["command"] = command
        if command_args:
            envelope["command_args"] = command_args

    # This is only a weak platform preference. Semantic output language is
    # resolved by TurnInterpreter from the current request.
    locale = author.get("locale")
    policy_default = default_locale or load_product_policy().language.default_locale
    envelope["platform_locale_hint"] = (
        str(locale).strip() if isinstance(locale, str) and locale.strip() else policy_default
    )
    return envelope


def discord_message_envelope_to_graph_state(
    message: MessageEnvelope,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal full-graph compatible state dict from Discord envelope."""
    context = dict(message.get("context", {}))
    now_req = request_id or f"dc-{int(time.time() * 1000)}"
    policy = load_product_policy()
    locale = policy.language.default_locale

    user_key = {
        "platform": "discord",
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
            "platform": "discord",
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
        }
    if guild_id and channel_id and thread_id:
        state["thread_key"] = {
            "platform": "discord",
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
            "thread_id": str(thread_id),
        }
    return state


def outbound_message_to_discord_requests(outbound: OutboundMessage) -> list[dict[str, Any]]:
    """Convert OutboundMessage to Discord send-message request payloads."""
    context = outbound.get("context", {})
    channel_id = context.get("channel_id")
    if not channel_id:
        raise ValueError("OutboundMessage.context missing channel_id")

    reply_to = outbound.get("reply_to_message_id")
    requests: list[dict[str, Any]] = []
    segments = outbound.get("segments", [])
    for idx, segment in enumerate(segments):
        payload: dict[str, Any] = {
            "channel_id": str(channel_id),
            "content": str(segment.get("text", "")),
            "allowed_mentions": {"parse": []},
        }
        if idx == 0 and reply_to:
            payload["reply_to_message_id"] = str(reply_to)
        if outbound.get("append_csat") and idx == len(segments) - 1:
            payload["components"] = _build_csat_components(str(outbound.get("request_id", "")))
        requests.append({"method": "create_message", "payload": payload})
    return requests


def _extract_reply_text(reference: dict[str, Any]) -> str:
    for key in ("content", "reply_to_content"):
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    resolved = reference.get("resolved")
    if isinstance(resolved, dict):
        value = resolved.get("content")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _truncate_reply_content(text: str, limit: int = 700) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _build_csat_components(request_id: str) -> list[dict[str, Any]]:
    buttons = []
    for score in range(1, 6):
        buttons.append(
            {
                "type": 2,
                "style": 2,
                "label": str(score),
                "custom_id": f"csat:{request_id}:{score}",
            }
        )
    return [{"type": 1, "components": buttons}]


def _parse_command(text: str) -> tuple[str | None, str | None]:
    raw = text.strip()
    if not raw.startswith("/") or len(raw) < 2:
        return None, None
    parts = raw.split(maxsplit=1)
    command = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    return command, args or None


def _extract_ts_ms(message: dict[str, Any]) -> int:
    ts = message.get("timestamp")
    if isinstance(ts, (int, float)):
        return int(float(ts) * 1000)
    if isinstance(ts, str) and ts.strip():
        value = ts.strip().replace("Z", "+00:00")
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except ValueError:
            pass
    return int(time.time() * 1000)


def _extract_attachments(message: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    raw_attachments = message.get("attachments")
    if isinstance(raw_attachments, list):
        for item in raw_attachments:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url:
                continue
            ctype = str(item.get("content_type", "") or "")
            kind = "image" if ctype.startswith("image/") else "file"
            attachment = {
                "kind": kind,
                "url": url,
                "name": str(item.get("filename") or item.get("id") or "file"),
            }
            if item.get("content_type") is not None:
                attachment["mime_type"] = str(item.get("content_type") or "")
            if item.get("size") is not None:
                attachment["file_size"] = str(item.get("size") or "")
            out.append(attachment)

    embeds = message.get("embeds")
    if isinstance(embeds, list):
        for embed in embeds:
            if not isinstance(embed, dict):
                continue
            url = embed.get("url")
            if isinstance(url, str) and url:
                out.append({"kind": "link", "url": url, "name": str(embed.get("title") or "embed")})

    return out
