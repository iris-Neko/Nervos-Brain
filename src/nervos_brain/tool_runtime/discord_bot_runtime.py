"""Discord Bot runtime (token-based online receive/send layer)."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from nervos_brain.logging_system import log_request_context
from nervos_brain.response_normalizer.platform_formatter import format_response_to_outbound

from .discord_bot_protocol_adapter import (
    discord_message_envelope_to_graph_state,
    discord_message_to_message_envelope,
    outbound_message_to_discord_requests,
)
from .fast_mode import FastModeStateStore, fast_status_message, parse_fast_command
from .feedback import FeedbackJsonlStore, parse_csat_callback_data
from .progress import ProgressUpdateConfig

logger = logging.getLogger(__name__)

_MAX_TEXT_ATTACHMENT_CHARS = 20_000
_DEFAULT_ATTACHMENT_MAX_BYTES = 256 * 1024
_TEXT_ATTACHMENT_EXTS = {".txt", ".md", ".markdown", ".log", ".json", ".yaml", ".yml"}
_TEXT_ATTACHMENT_MIME_PREFIXES = ("text/",)
_TEXT_ATTACHMENT_MIME_TYPES = {
    "application/json",
    "application/yaml",
    "application/x-yaml",
}


class DiscordBotRuntimeError(RuntimeError):
    """Domain error for Discord Bot runtime."""


@dataclass(frozen=True)
class DiscordBotConfig:
    """Config for Discord Bot runtime."""

    bot_token: str
    mention_only_in_guild: bool = True

    @classmethod
    def from_env(cls) -> "DiscordBotConfig":
        token = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
        if not token:
            raise DiscordBotRuntimeError("Missing DISCORD_BOT_TOKEN environment variable.")

        mention_raw = (os.getenv("DISCORD_MENTION_ONLY_IN_GUILD") or "true").strip().lower()
        mention_only = mention_raw not in {"0", "false", "no"}
        return cls(bot_token=token, mention_only_in_guild=mention_only)


GraphRunner = Callable[[dict[str, Any]], dict[str, Any]]


class DiscordGateway:
    """Gateway logic that converts Discord payload to graph call and send requests."""

    def __init__(
        self,
        *,
        graph_runner: GraphRunner,
        allowed_channel_ids: set[str] | None = None,
        allowed_guild_ids: set[str] | None = None,
        render_mode: str = "markdown",
        append_csat: bool = False,
        mention_only_in_guild: bool = True,
        respond_to_bot_replies: bool = True,
        feedback_store: FeedbackJsonlStore | None = None,
        debug_log_file: str | Path | None = None,
        memory_service: Any | None = None,
        memory_context_limit: int = 20,
        target_elapsed_ms: int = 30000,
        max_elapsed_ms: int = 90000,
        attachment_max_bytes: int = _DEFAULT_ATTACHMENT_MAX_BYTES,
        fast_mode_store: FastModeStateStore | None = None,
    ) -> None:
        self._graph_runner = graph_runner
        self._allowed_channel_ids = set(allowed_channel_ids or [])
        self._allowed_guild_ids = set(allowed_guild_ids or [])
        self._render_mode = "plain" if render_mode == "plain" else "markdown"
        self._append_csat = append_csat
        self._mention_only_in_guild = mention_only_in_guild
        self._respond_to_bot_replies = bool(respond_to_bot_replies)
        self._feedback_store = feedback_store
        self._debug_log_file = Path(str(debug_log_file)).expanduser() if str(debug_log_file or "").strip() else None
        self._memory_service = memory_service
        self._memory_context_limit = max(1, min(int(memory_context_limit or 20), 100))
        self._target_elapsed_ms = max(0, int(target_elapsed_ms or 0))
        self._max_elapsed_ms = max(0, int(max_elapsed_ms or 0))
        self._attachment_max_bytes = max(1024, int(attachment_max_bytes or _DEFAULT_ATTACHMENT_MAX_BYTES))
        self._fast_mode_store = fast_mode_store or FastModeStateStore()

    def process_message_payload(
        self,
        payload: dict[str, Any],
        *,
        bot_user_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        message_id = str(payload.get("id", "unknown"))
        channel_id = str(payload.get("channel_id", "")) if payload.get("channel_id") is not None else None
        guild_id = str(payload.get("guild_id", "")) if payload.get("guild_id") is not None else None

        author = payload.get("author")
        if not isinstance(author, dict):
            author = {}
        if bool(author.get("bot", False)):
            return _ignored_row(message_id, channel_id, guild_id, "bot_sender")

        if self._allowed_guild_ids and guild_id and guild_id not in self._allowed_guild_ids:
            return _ignored_row(message_id, channel_id, guild_id, "guild_not_allowed")
        if self._allowed_channel_ids and channel_id and channel_id not in self._allowed_channel_ids:
            return _ignored_row(message_id, channel_id, guild_id, "channel_not_allowed")

        content = str(payload.get("content", "") or "")
        is_runtime_command = _is_feedback_command_text(content) or _is_fast_command_text(content)
        if guild_id and self._mention_only_in_guild:
            if not bot_user_id:
                return _ignored_row(message_id, channel_id, guild_id, "missing_bot_user_id")
            mentioned = _is_bot_mentioned(content, bot_user_id, payload)
            bot_reply = self._respond_to_bot_replies and _is_reply_to_bot(payload)
            if not mentioned and not bot_reply and not is_runtime_command:
                return _ignored_row(message_id, channel_id, guild_id, "not_mentioned")
            if mentioned:
                content = _strip_bot_mention(content, bot_user_id)

        if _is_feedback_command_text(content):
            normalized_feedback_payload = dict(payload)
            normalized_feedback_payload["content"] = content
            try:
                envelope = discord_message_to_message_envelope(normalized_feedback_payload)
            except ValueError:
                return _ignored_row(message_id, channel_id, guild_id, "unsupported_message")
            return self._process_feedback_command(envelope=envelope, dry_run=dry_run)

        normalized_payload = dict(payload)
        normalized_payload["content"] = content

        try:
            envelope = discord_message_to_message_envelope(normalized_payload)
        except ValueError:
            return _ignored_row(message_id, channel_id, guild_id, "unsupported_message")

        fast_command = parse_fast_command(
            str(envelope.get("command", "") or ""),
            str(envelope.get("command_args", "") or ""),
        )
        if fast_command is not None:
            return self._process_fast_command(
                envelope=envelope,
                action=fast_command.action,
                message=fast_command.message,
                dry_run=dry_run,
            )

        state = discord_message_envelope_to_graph_state(envelope)
        if self._consume_fast_mode(envelope):
            state["_llm_service_tier"] = "priority"
        request_started_ts_ms = int(time.time() * 1000)
        graph_start = time.perf_counter()
        state["_request_started_ts_ms"] = request_started_ts_ms
        budget = state.setdefault("budget", {})
        if not isinstance(budget, dict):
            budget = {}
            state["budget"] = budget
        if self._target_elapsed_ms > 0:
            budget.setdefault("target_elapsed_ms", self._target_elapsed_ms)
        if self._max_elapsed_ms > 0:
            budget.setdefault("max_elapsed_ms", self._max_elapsed_ms)
        state["render_mode"] = self._render_mode
        state["append_csat"] = self._append_csat
        self._prepare_message_attachments(envelope)
        self._attach_recent_memory_context(state=state, envelope=envelope)
        self._write_memory_event(envelope=envelope, role="user", content=str(envelope.get("content", "") or ""))
        request_id = str(state.get("request_id", "unknown"))

        with log_request_context(request_id):
            try:
                result = self._graph_runner(state)
            except Exception:
                logger.exception(
                    "discord graph_runner failed message_id=%s channel_id=%s guild_id=%s",
                    message_id,
                    channel_id,
                    guild_id,
                )
                result = {
                    "_final_response": {
                        "request_id": str(state.get("request_id", "unknown")),
                        "text": "处理请求时发生错误，请稍后重试。",
                        "citations": [],
                    }
                }
        graph_elapsed_ms = int((time.perf_counter() - graph_start) * 1000)
        result["_graph_elapsed_ms"] = graph_elapsed_ms
        result["_request_started_ts_ms"] = request_started_ts_ms

        outbound = _build_outbound_from_graph_result(
            result=result,
            state=state,
            envelope=envelope,
            render_mode=self._render_mode,
            append_csat=self._append_csat,
        )
        self._write_memory_event(
            envelope=envelope,
            role="assistant",
            content=_outbound_text_preview(outbound, limit=4000),
        )
        send_requests = outbound_message_to_discord_requests(outbound)
        if self._feedback_store is not None and not dry_run:
            self._feedback_store.append_answer(
                _build_answer_feedback_record(
                    result=result,
                    state=state,
                    envelope=envelope,
                    outbound=outbound,
                    sent_count=len(send_requests),
                    has_csat=bool(self._append_csat and send_requests),
                )
            )
        self._write_debug_event(
            envelope=envelope,
            state=state,
            result=result,
            outbound=outbound,
            send_reqs=send_requests,
            sent_count=len(send_requests),
            dry_run=dry_run,
        )
        logger.info(
            "discord processed message_id=%s channel_id=%s guild_id=%s request_id=%s segments=%d",
            message_id,
            channel_id,
            guild_id,
            request_id,
            len(send_requests),
        )
        return {
            "message_id": message_id,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "ignored": False,
            "reason": "",
            "request_id": state.get("request_id"),
            "send_requests": send_requests,
        }

    def _process_fast_command(
        self,
        *,
        envelope: dict[str, Any],
        action: str,
        message: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
        user_id = str(context.get("user_id", "") or "")
        platform = str(context.get("platform", "discord") or "discord")
        if action == "enable":
            if not dry_run:
                self._fast_mode_store.enable_next(platform=platform, user_id=user_id)
            text = message
        elif action == "disable":
            if not dry_run:
                self._fast_mode_store.disable(platform=platform, user_id=user_id)
            text = message
        elif action == "status":
            text = fast_status_message(
                pending=self._fast_mode_store.is_pending(platform=platform, user_id=user_id)
            )
        else:
            text = message
        send_requests = [
            {
                "method": "create_message",
                "payload": {
                    "channel_id": str(context.get("channel_id", "")),
                    "content": text,
                    "allowed_mentions": {"parse": []},
                },
            }
        ]
        return {
            "message_id": envelope.get("message_id"),
            "channel_id": context.get("channel_id"),
            "guild_id": context.get("guild_id"),
            "ignored": False,
            "reason": "fast_command",
            "request_id": None,
            "send_requests": [] if dry_run else send_requests,
            "fast_action": action,
        }

    def _consume_fast_mode(self, envelope: dict[str, Any]) -> bool:
        context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
        return self._fast_mode_store.consume_next(
            platform=str(context.get("platform", "discord") or "discord"),
            user_id=str(context.get("user_id", "") or ""),
        )

    def process_interaction_payload(self, payload: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            return {"ignored": True, "reason": "unsupported_interaction", "request_id": None}
        custom_id = str(data.get("custom_id", "") or "")
        try:
            parsed = parse_csat_callback_data(custom_id)
        except ValueError:
            return {"ignored": True, "reason": "invalid_callback", "request_id": None}

        user = payload.get("user") or payload.get("member", {}).get("user")
        if not isinstance(user, dict):
            user = {}
        message = payload.get("message")
        if not isinstance(message, dict):
            message = {}
        channel_id = str(payload.get("channel_id") or message.get("channel_id") or "")
        guild_id = str(payload.get("guild_id") or message.get("guild_id") or "")
        answer_meta = self._feedback_store.latest_answer(parsed.request_id) if self._feedback_store is not None else None
        answer_fields = _feedback_fields_from_answer(answer_meta)
        record = {
            "request_id": parsed.request_id,
            "platform": "discord",
            "chat_id": channel_id,
            "user_id": str(user.get("id", "")),
            "score": parsed.score,
            "comment": "",
            "created_ts_ms": int(time.time() * 1000),
            "guild_id": guild_id,
            "channel_id": channel_id,
            **answer_fields,
        }
        if self._feedback_store is not None and not dry_run:
            record = self._feedback_store.append(record)
        return {
            "ignored": False,
            "reason": "csat",
            "request_id": parsed.request_id,
            "score": parsed.score,
            "is_bad_case": bool(record.get("is_bad_case", parsed.score <= 3)),
            "is_duplicate_rating": bool(record.get("is_duplicate_rating", False)),
        }

    def _process_feedback_command(self, *, envelope: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        args = str(envelope.get("command_args", "") or "").strip()
        request_id, comment = _parse_feedback_args(args)
        context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
        if not request_id or not comment:
            send_requests = [
                {
                    "method": "create_message",
                    "payload": {
                        "channel_id": str(context.get("channel_id", "")),
                        "content": "用法: /feedback <request_id> <comment>",
                        "allowed_mentions": {"parse": []},
                    },
                }
            ]
            return {
                "message_id": envelope.get("message_id"),
                "channel_id": context.get("channel_id"),
                "guild_id": context.get("guild_id"),
                "ignored": True,
                "reason": "invalid_feedback_command",
                "request_id": request_id,
                "send_requests": send_requests,
            }
        answer_meta = self._feedback_store.latest_answer(request_id) if self._feedback_store is not None else None
        record = {
            "request_id": request_id,
            "platform": "discord",
            "chat_id": str(context.get("channel_id", "")),
            "user_id": str(context.get("user_id", "")),
            "comment": comment,
            "created_ts_ms": int(time.time() * 1000),
            **_context_ids(context),
            **_feedback_fields_from_answer(answer_meta),
        }
        if self._feedback_store is not None and not dry_run:
            self._feedback_store.append_comment(record)
        return {
            "message_id": envelope.get("message_id"),
            "channel_id": context.get("channel_id"),
            "guild_id": context.get("guild_id"),
            "ignored": False,
            "reason": "feedback",
            "request_id": request_id,
            "send_requests": [
                {
                    "method": "create_message",
                    "payload": {
                        "channel_id": str(context.get("channel_id", "")),
                        "content": f"Feedback recorded for {request_id}.",
                        "allowed_mentions": {"parse": []},
                    },
                }
            ],
        }

    def _prepare_message_attachments(self, envelope: dict[str, Any]) -> None:
        attachments = envelope.get("attachments", [])
        if not isinstance(attachments, list) or not attachments:
            return
        text_blocks: list[str] = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            self._prepare_single_attachment(attachment=attachment, text_blocks=text_blocks)
        if text_blocks:
            base = str(envelope.get("content", "") or "").strip()
            envelope["content"] = f"{base}\n\n" + "\n\n".join(text_blocks) if base else "\n\n".join(text_blocks)

    def _prepare_single_attachment(self, *, attachment: dict[str, Any], text_blocks: list[str]) -> None:
        name = _safe_attachment_name(str(attachment.get("name", "") or "discord_file"))
        mime_type = str(attachment.get("mime_type", "") or mimetypes.guess_type(name)[0] or "")
        if not _is_supported_text_attachment(name=name, mime_type=mime_type):
            attachment["status"] = "unsupported"
            return
        declared_size = _int_like(attachment.get("file_size", 0))
        if declared_size and declared_size > self._attachment_max_bytes:
            attachment["status"] = "too_large"
            return
        try:
            data = _download_attachment(str(attachment.get("url", "")), max_bytes=self._attachment_max_bytes)
        except Exception as exc:
            attachment["status"] = "download_failed"
            attachment["error"] = str(exc)[:160]
            return
        if not data:
            attachment["status"] = "empty"
            return
        decoded = _decode_text_attachment(data)
        if decoded is None:
            attachment["status"] = "decode_failed"
            return
        if len(decoded) > _MAX_TEXT_ATTACHMENT_CHARS:
            decoded = decoded[:_MAX_TEXT_ATTACHMENT_CHARS].rstrip() + "\n...[文件内容已截断]"
            attachment["truncated"] = "true"
        attachment["status"] = "ready"
        attachment["mime_type"] = mime_type
        attachment["file_size"] = str(len(data))
        text_blocks.append(f"用户上传的文本文件 `{name}` 内容：\n```text\n{decoded}\n```")

    def _attach_recent_memory_context(self, *, state: dict[str, Any], envelope: dict[str, Any]) -> None:
        reply_context = _reply_context_from_envelope(envelope)
        if envelope.get("reply_to_message_id"):
            state["recent_messages"] = []
            state["conversation_context"] = reply_context
            return

        svc = self._memory_service
        if svc is None or not hasattr(svc, "list_recent_message_events"):
            if reply_context:
                state["recent_messages"] = []
                state["conversation_context"] = reply_context
            return
        context = envelope.get("context", {})
        if not isinstance(context, dict):
            return
        user_id = str(context.get("user_id", "") or "")
        if not user_id:
            return
        if not _should_attach_recent_context(envelope):
            state["recent_messages"] = []
            state["conversation_context"] = ""
            return
        try:
            rows = svc.list_recent_message_events(
                platform="discord",
                user_id=user_id,
                guild_id=_optional_str(context.get("guild_id")),
                channel_id=_optional_str(context.get("channel_id")),
                thread_id=_optional_str(context.get("thread_id")),
                limit=self._memory_context_limit,
            )
        except Exception as exc:
            logger.debug("discord recent memory read failed user_id=%s error=%s", user_id, exc)
            return
        if isinstance(rows, list):
            state["recent_messages"] = rows
            recent_context = _format_recent_messages(rows)
            if reply_context and recent_context:
                state["conversation_context"] = f"{reply_context}\n{recent_context}"
            else:
                state["conversation_context"] = reply_context or recent_context

    def _write_memory_event(self, *, envelope: dict[str, Any], role: str, content: str) -> None:
        svc = self._memory_service
        text = str(content or "").strip()
        if svc is None or not text:
            return
        context = envelope.get("context", {})
        if not isinstance(context, dict):
            return
        user_id = str(context.get("user_id", "") or "")
        if not user_id:
            return
        try:
            svc.write_message_event(
                platform="discord",
                user_id=user_id,
                guild_id=_optional_str(context.get("guild_id")),
                channel_id=_optional_str(context.get("channel_id")),
                thread_id=_optional_str(context.get("thread_id")),
                role=role,
                content=text,
                created_ts_ms=_memory_event_ts_ms(envelope=envelope, role=role),
            )
        except Exception as exc:
            logger.debug("discord memory write failed role=%s user_id=%s error=%s", role, user_id, exc)

    def _write_debug_event(
        self,
        *,
        envelope: dict[str, Any],
        state: dict[str, Any],
        result: dict[str, Any],
        outbound: dict[str, Any],
        send_reqs: list[dict[str, Any]],
        sent_count: int,
        dry_run: bool,
    ) -> None:
        path = self._debug_log_file
        if path is None:
            return
        try:
            response = result.get("_final_response") if isinstance(result.get("_final_response"), dict) else {}
            context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
            event = {
                "created_ts_ms": int(time.time() * 1000),
                "platform": "discord",
                "request_id": str(state.get("request_id") or response.get("request_id") or ""),
                "guild_id": str(context.get("guild_id", "")),
                "channel_id": str(context.get("channel_id", "")),
                "thread_id": str(context.get("thread_id", "")),
                "user_id": str(context.get("user_id", "")),
                "message_id": str(envelope.get("message_id", "")),
                "incoming_reply_to_message_id": str(envelope.get("reply_to_message_id", "")),
                "reply_to_role": str(envelope.get("reply_to_role", "")),
                "reply_to_content_preview": _preview_text(envelope.get("reply_to_content", ""), limit=500),
                "conversation_context_preview": _preview_text(state.get("conversation_context", ""), limit=900),
                "recent_messages_preview": _debug_recent_messages_preview(state.get("recent_messages")),
                "content_preview": str(envelope.get("content", ""))[:240],
                "route_decision": str(result.get("_route_decision") or state.get("_route_decision") or ""),
                "llm_service_tier": str(result.get("_llm_service_tier") or state.get("_llm_service_tier") or ""),
                "retrieval_policy": str(result.get("retrieval_policy") or state.get("retrieval_policy") or ""),
                "reflection_decision": str(result.get("reflection_decision") or state.get("reflection_decision") or ""),
                "tool_summary": str(result.get("_tool_execution_summary") or state.get("_tool_execution_summary") or ""),
                "tool_calls": _infer_tool_calls(result=result, state=state),
                "evidence_count": _infer_evidence_count(result=result, state=state, response=response),
                "citation_count": len(response.get("citations", [])) if isinstance(response.get("citations"), list) else 0,
                "graph_elapsed_ms": _int_like(result.get("_graph_elapsed_ms", 0)),
                "node_timings": _debug_node_timings(result.get("_node_timings") or state.get("_node_timings")),
                "llm_usage_summary": _debug_llm_usage_summary(result.get("_llm_usage_summary") or state.get("_llm_usage_summary")),
                "final_text_preview": _outbound_text_preview(outbound, limit=600),
                "sent_count": int(sent_count),
                "dry_run": bool(dry_run),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug("discord debug event write failed: %s", exc)


class DiscordBotRuntime:
    """Online Discord runtime backed by discord.py event loop."""

    def __init__(
        self,
        *,
        config: DiscordBotConfig,
        gateway: DiscordGateway,
        max_worker_threads: int = 4,
        progress_update_config: ProgressUpdateConfig | None = None,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._max_worker_threads = max(1, min(int(max_worker_threads or 1), 32))
        self._progress_update_config = progress_update_config or ProgressUpdateConfig(enabled=False)
        self._channel_locks: dict[str, asyncio.Lock] = {}

    def run(self) -> None:
        try:
            import discord
        except ImportError as exc:
            raise DiscordBotRuntimeError("Missing dependency: discord.py. Install with `pip install discord.py`.") from exc

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.dm_messages = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            logger.info("Discord bot online: user=%s id=%s", client.user, client.user.id if client.user else "unknown")

        @client.event
        async def on_message(message):
            if client.user is None:
                return
            payload = _discord_message_to_payload(message)
            channel_id = str(payload.get("channel_id") or "unknown")
            lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())
            progress_task: asyncio.Task | None = None
            async with lock:
                progress_task = asyncio.create_task(
                    _send_discord_progress_updates(
                        message=message,
                        config=self._progress_update_config,
                    )
                )
                try:
                    row = await _run_gateway_in_executor(
                        gateway=self._gateway,
                        payload=payload,
                        bot_user_id=str(client.user.id),
                        executor=executor,
                    )
                finally:
                    await _stop_discord_progress_task(progress_task)
            if row.get("ignored") and not row.get("send_requests"):
                logger.info("[skip] message_id=%s channel_id=%s guild_id=%s reason=%s", row.get("message_id"), row.get("channel_id"), row.get("guild_id"), row.get("reason"))
                return
            send_requests = row.get("send_requests", [])
            sent = await _send_discord_requests(message, send_requests)
            logger.info("[ok] message_id=%s channel_id=%s guild_id=%s request_id=%s sent=%s", row.get("message_id"), row.get("channel_id"), row.get("guild_id"), row.get("request_id"), sent)

        @client.event
        async def on_interaction(interaction):
            payload = _discord_interaction_to_payload(interaction)
            row = self._gateway.process_interaction_payload(payload)
            try:
                if hasattr(interaction, "response") and not interaction.response.is_done():
                    text = "Rating already recorded." if row.get("is_duplicate_rating") else f"Thanks. Rating recorded: {row.get('score')}/5"
                    await interaction.response.send_message(text, ephemeral=True)
            except Exception:
                logger.debug("discord interaction response failed", exc_info=True)

        executor = ThreadPoolExecutor(max_workers=self._max_worker_threads, thread_name_prefix="discord-graph-worker")
        try:
            client.run(self._config.bot_token)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


async def _run_gateway_in_executor(*, gateway: DiscordGateway, payload: dict[str, Any], bot_user_id: str, executor: ThreadPoolExecutor) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: gateway.process_message_payload(payload, bot_user_id=bot_user_id))


async def _send_discord_progress_updates(*, message: Any, config: ProgressUpdateConfig) -> None:
    if not config.should_run:
        return
    try:
        await asyncio.sleep(config.first_after_s)
        for idx in range(config.max_updates):
            await _send_discord_progress_message(message=message, text=config.message_for(idx))
            if idx >= config.max_updates - 1:
                return
            await asyncio.sleep(config.interval_s)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("discord progress update loop failed", exc_info=True)


async def _stop_discord_progress_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("discord progress task failed while stopping", exc_info=True)


async def _send_discord_progress_message(*, message: Any, text: str) -> None:
    allowed_mentions = _build_discord_allowed_mentions({"allowed_mentions": {"parse": []}})
    try:
        await message.reply(text, allowed_mentions=allowed_mentions)
    except Exception:
        logger.debug("discord progress reply failed; retrying detached message", exc_info=True)
        try:
            await message.channel.send(text, allowed_mentions=allowed_mentions)
        except Exception:
            logger.debug("discord progress detached send failed", exc_info=True)


def _is_bot_mentioned(content: str, bot_user_id: str, payload: dict[str, Any]) -> bool:
    if re.search(rf"<@!?{re.escape(bot_user_id)}>", content):
        return True
    mention_ids = payload.get("mention_user_ids")
    if isinstance(mention_ids, list):
        return bot_user_id in {str(v) for v in mention_ids}
    return False


def _strip_bot_mention(content: str, bot_user_id: str) -> str:
    cleaned = re.sub(rf"<@!?{re.escape(bot_user_id)}>", "", content)
    return cleaned.strip()


def _is_reply_to_bot(payload: dict[str, Any]) -> bool:
    ref = payload.get("reference")
    if not isinstance(ref, dict):
        return False
    author = ref.get("author")
    return isinstance(author, dict) and bool(author.get("bot"))


def _discord_message_to_payload(message: Any) -> dict[str, Any]:
    channel_id = str(getattr(message.channel, "id", ""))
    guild_id = str(getattr(message.guild, "id", "")) if getattr(message, "guild", None) else None
    thread_id: str | None = None
    parent = getattr(message.channel, "parent", None)
    if parent is not None and getattr(parent, "id", None) is not None:
        thread_id = channel_id
        channel_id = str(parent.id)
    payload: dict[str, Any] = {
        "id": str(getattr(message, "id", "unknown")),
        "content": str(getattr(message, "content", "") or ""),
        "timestamp": getattr(message, "created_at", None).isoformat() if getattr(message, "created_at", None) else None,
        "author": {"id": str(getattr(message.author, "id", "unknown_user")), "bot": bool(getattr(message.author, "bot", False)), "locale": getattr(message.author, "locale", None)},
        "channel_id": channel_id,
        "guild_id": guild_id,
        "thread_id": thread_id,
        "mention_user_ids": [str(getattr(u, "id", "")) for u in getattr(message, "mentions", [])],
        "attachments": [
            {"id": str(getattr(a, "id", "")), "url": str(getattr(a, "url", "")), "filename": str(getattr(a, "filename", "")), "content_type": str(getattr(a, "content_type", "")), "size": str(getattr(a, "size", ""))}
            for a in getattr(message, "attachments", [])
        ],
        "embeds": [{"url": str(getattr(e, "url", "")) if getattr(e, "url", None) else "", "title": str(getattr(e, "title", ""))} for e in getattr(message, "embeds", [])],
    }
    reference = getattr(message, "reference", None)
    if reference is not None and getattr(reference, "message_id", None) is not None:
        ref: dict[str, Any] = {"message_id": str(reference.message_id)}
        resolved = getattr(reference, "resolved", None)
        if resolved is not None:
            ref["content"] = str(getattr(resolved, "content", "") or "")
            ref_author = getattr(resolved, "author", None)
            ref["author"] = {"id": str(getattr(ref_author, "id", "")), "bot": bool(getattr(ref_author, "bot", False))}
        payload["reference"] = ref
    return payload


def _discord_interaction_to_payload(interaction: Any) -> dict[str, Any]:
    user = getattr(interaction, "user", None)
    message = getattr(interaction, "message", None)
    data = getattr(interaction, "data", {})
    return {
        "id": str(getattr(interaction, "id", "")),
        "channel_id": str(getattr(getattr(interaction, "channel", None), "id", "") or getattr(interaction, "channel_id", "") or ""),
        "guild_id": str(getattr(getattr(interaction, "guild", None), "id", "") or getattr(interaction, "guild_id", "") or ""),
        "user": {"id": str(getattr(user, "id", "")), "bot": bool(getattr(user, "bot", False))},
        "data": data if isinstance(data, dict) else {},
        "message": {"id": str(getattr(message, "id", "")), "content": str(getattr(message, "content", "") or "")} if message is not None else {},
    }


def _build_outbound_from_graph_result(*, result: dict[str, Any], state: dict[str, Any], envelope: dict[str, Any], render_mode: str, append_csat: bool) -> dict[str, Any]:
    outbound = result.get("_outbound_message")
    if isinstance(outbound, dict) and isinstance(outbound.get("segments"), list):
        if envelope.get("message_id"):
            outbound["reply_to_message_id"] = str(envelope.get("message_id"))
        return outbound
    response = result.get("_final_response")
    if not isinstance(response, dict):
        response = {}
    response.setdefault("request_id", str(state.get("request_id", "unknown")))
    response.setdefault("text", "")
    response.setdefault("citations", [])
    return format_response_to_outbound(response=response, context=envelope["context"], render_mode="plain" if render_mode == "plain" else "markdown", append_csat=bool(append_csat), reply_to_message_id=str(envelope.get("message_id") or "") or None)


async def _send_discord_requests(message: Any, send_requests: list[dict[str, Any]]) -> int:
    sent = 0
    for req in send_requests:
        payload = req.get("payload")
        if not isinstance(payload, dict):
            continue
        content = str(payload.get("content", ""))
        reply_to = payload.get("reply_to_message_id")
        components = payload.get("components")
        allowed_mentions = _build_discord_allowed_mentions(payload)
        view = _build_discord_view(components)
        try:
            if reply_to:
                await message.reply(content, allowed_mentions=allowed_mentions, view=view)
            else:
                await message.channel.send(content, allowed_mentions=allowed_mentions, view=view)
        except Exception:
            logger.warning("Discord rich/reply send failed; retrying plain detached message.", exc_info=True)
            plain = _plain_discord_text(content)
            await message.channel.send(plain, allowed_mentions=allowed_mentions)
        sent += 1
    return sent


def _build_discord_allowed_mentions(payload: dict[str, Any]) -> Any:
    try:
        import discord
        raw = payload.get("allowed_mentions")
        if isinstance(raw, dict) and not raw.get("parse"):
            return discord.AllowedMentions.none()
        return discord.AllowedMentions.none()
    except Exception:
        return None


def _build_discord_view(components: Any) -> Any:
    if not isinstance(components, list) or not components:
        return None
    try:
        import discord
        view = discord.ui.View(timeout=None)
        for row in components:
            if not isinstance(row, dict):
                continue
            for item in row.get("components", []):
                if not isinstance(item, dict) or item.get("type") != 2:
                    continue
                button = discord.ui.Button(label=str(item.get("label", "")), style=discord.ButtonStyle.secondary, custom_id=str(item.get("custom_id", "")))
                view.add_item(button)
        return view
    except Exception:
        return None


def _plain_discord_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    return cleaned[:2000]


def _ignored_row(message_id: str, channel_id: str | None, guild_id: str | None, reason: str) -> dict[str, Any]:
    return {"message_id": message_id, "channel_id": channel_id, "guild_id": guild_id, "ignored": True, "reason": reason, "request_id": None, "send_requests": []}


def _is_feedback_command_text(text: str) -> bool:
    raw = str(text or "").strip().lower()
    return raw.startswith("/feedback")


def _is_fast_command_text(text: str) -> bool:
    raw = str(text or "").strip().lower()
    return raw == "/fast" or raw.startswith("/fast ")


def _parse_feedback_args(args: str) -> tuple[str, str]:
    text = str(args or "").strip()
    if not text:
        return "", ""
    parts = text.split(maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (parts[0].strip(), "")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_like(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_attachment_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "file")).strip("._")
    return cleaned or "file"


def _is_supported_text_attachment(*, name: str, mime_type: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix in _TEXT_ATTACHMENT_EXTS:
        return True
    mt = mime_type.lower().strip()
    return mt.startswith(_TEXT_ATTACHMENT_MIME_PREFIXES) or mt in _TEXT_ATTACHMENT_MIME_TYPES


def _download_attachment(url: str, *, max_bytes: int) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise ValueError("unsupported attachment url")
    req = Request(url, headers={"User-Agent": "nervos-brain-discord-bot"})
    with urlopen(req, timeout=15) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("attachment too large")
    return data


def _decode_text_attachment(data: bytes) -> str | None:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = data.decode(encoding).replace("\x00", "")
            return text.replace("```", "`\u200b``").strip()
        except UnicodeDecodeError:
            continue
    return None


def _format_recent_messages(rows: list[Any], limit_chars: int = 1800) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "message") or "message")
        content = re.sub(r"\s+", " ", str(row.get("content", "") or "")).strip()
        if not content:
            continue
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        lines.append(f"{role}: {content}")
    text = "\n".join(lines).strip()
    return text[-limit_chars:].lstrip() if len(text) > limit_chars else text


def _reply_context_from_envelope(envelope: dict[str, Any]) -> str:
    reply_text = re.sub(r"\s+", " ", str(envelope.get("reply_to_content", "") or "")).strip()
    reply_id = str(envelope.get("reply_to_message_id", "") or "").strip()
    if not reply_text:
        if reply_id:
            role = str(envelope.get("reply_to_role", "") or "").strip().lower()
            label = "assistant" if role in {"assistant", "bot"} else "user" if role == "user" else "replied_message"
            return (
                f"当前消息正在回复一条 {label} 消息（message_id={reply_id}），"
                "但平台没有提供被回复消息内容。不要从普通历史记录猜测被回复内容；"
                "如果当前短追问无法独立理解，应请用户补充或重发被回复内容。"
            )
        return ""
    if len(reply_text) > 700:
        reply_text = reply_text[:700].rstrip() + "..."
    role = str(envelope.get("reply_to_role", "") or "").strip().lower()
    label = "assistant" if role in {"assistant", "bot"} else "user" if role == "user" else "replied_message"
    return f"当前消息正在回复这条 {label} 消息: {reply_text}"


def _preview_text(value: Any, *, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def _debug_recent_messages_preview(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "role": str(item.get("role", "")),
                "content_preview": _preview_text(item.get("content", ""), limit=180),
            }
        )
    return rows


def _normalized_message_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _asks_for_recent_context(text: str) -> bool:
    normalized = _normalized_message_text(text)
    markers = ("上文", "上下文", "刚才", "刚刚", "之前", "前面", "上面", "继续", "接着", "延续", "它", "这个", "那个", "换成", "再来", "再写")
    return bool(normalized) and any(marker in normalized for marker in markers)


def _is_standalone_named_question(text: str) -> bool:
    normalized = _normalized_message_text(text)
    if not normalized or any(marker in normalized for marker in ("上文", "上下文", "刚才", "之前", "继续", "接着")):
        return False
    subjects = ("ckb", "nervos", "fiber", "ccc", "rgb++", "spore", "molecule", "cell", "utxo")
    intents = ("是什么", "什么是", "怎么", "如何", "为什么", "区别", "原理", "流程", "开发", "教程", "代码", "示例", "例子")
    return any(subject in normalized for subject in subjects) and any(intent in normalized for intent in intents)


def _looks_like_short_followup(text: str) -> bool:
    normalized = _normalized_message_text(text)
    if not normalized or _is_standalone_named_question(text):
        return False
    if normalized in {"你是谁", "help", "/help", "帮助"}:
        return False
    return len(normalized) <= 48


def _should_attach_recent_context(envelope: dict[str, Any]) -> bool:
    if envelope.get("reply_to_message_id"):
        return True
    text = str(envelope.get("content", "") or "")
    if _is_standalone_named_question(text):
        return False
    return _asks_for_recent_context(text) or _looks_like_short_followup(text)


def _memory_event_ts_ms(*, envelope: dict[str, Any], role: str) -> int:
    if role == "user":
        return _int_like(envelope.get("ts_ms"), int(time.time() * 1000))
    return int(time.time() * 1000)


def _context_ids(context: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("guild_id", "channel_id", "thread_id"):
        value = context.get(key)
        if value is not None and str(value).strip():
            result[key] = str(value)
    return result


def _feedback_fields_from_answer(answer: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {"trace_summary": "", "tool_calls": 0, "evidence_count": 0, "final_text_preview": ""}
    return {"trace_summary": str(answer.get("trace_summary", "") or ""), "tool_calls": _int_like(answer.get("tool_calls")), "evidence_count": _int_like(answer.get("evidence_count")), "final_text_preview": str(answer.get("final_text_preview", "") or "")}


def _build_answer_feedback_record(*, result: dict[str, Any], state: dict[str, Any], envelope: dict[str, Any], outbound: dict[str, Any], sent_count: int, has_csat: bool) -> dict[str, Any]:
    response = result.get("_final_response") if isinstance(result.get("_final_response"), dict) else {}
    context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
    request_id = str(state.get("request_id") or response.get("request_id") or "unknown")
    return {"request_id": request_id, "platform": "discord", "chat_id": str(context.get("channel_id") or ""), "user_id": str(context.get("user_id", "")), "created_ts_ms": int(time.time() * 1000), "trace_summary": str(response.get("trace_summary") or result.get("_tool_execution_summary") or state.get("_tool_execution_summary") or ""), "tool_calls": _infer_tool_calls(result=result, state=state), "evidence_count": _infer_evidence_count(result=result, state=state, response=response), "final_text_preview": _outbound_text_preview(outbound), "sent_count": int(sent_count), "has_csat": bool(has_csat), "answer_char_count": sum(int(seg.get("char_count", len(str(seg.get("text", ""))))) for seg in outbound.get("segments", []) if isinstance(seg, dict)), **_context_ids(context)}


def _infer_tool_calls(*, result: dict[str, Any], state: dict[str, Any]) -> int:
    for value in (result.get("_tool_calls_executed"), state.get("_tool_calls_executed")):
        if value is not None:
            return _int_like(value)
    trace = result.get("_tool_execution_trace") or state.get("_tool_execution_trace")
    return len(trace) if isinstance(trace, list) else 0


def _infer_evidence_count(*, result: dict[str, Any], state: dict[str, Any], response: dict[str, Any]) -> int:
    for value in (result.get("evidence_count"), state.get("evidence_count")):
        if value is not None:
            return _int_like(value)
    evidence = result.get("evidence") or state.get("evidence")
    if isinstance(evidence, list):
        return len(evidence)
    citations = response.get("citations")
    return len(citations) if isinstance(citations, list) else 0


def _outbound_text_preview(outbound: dict[str, Any], limit: int = 300) -> str:
    chunks = [str(seg.get("text", "") or "") for seg in outbound.get("segments", []) if isinstance(seg, dict)]
    return "\n".join(chunks).strip()[:limit]


def _debug_node_timings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, dict):
            rows.append({"node": str(item.get("node", "")), "elapsed_ms": _int_like(item.get("elapsed_ms", 0))})
    return rows


def _debug_llm_usage_summary(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
