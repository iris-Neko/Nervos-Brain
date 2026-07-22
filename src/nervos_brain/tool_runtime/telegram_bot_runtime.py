"""Telegram Bot polling runtime (token-based online receive/send layer).

Scope:
  - Telegram Bot API client (getMe/getUpdates/send*)
  - Update offset persistence
  - Online pipeline:
      Telegram update -> MessageEnvelope -> GraphState -> Outbound -> send requests
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from nervos_brain.logging_system import log_request_context
from nervos_brain.response_normalizer.platform_formatter import format_response_to_outbound

from .telegram_bot_protocol_adapter import (
    message_envelope_to_graph_state,
    outbound_message_to_telegram_requests,
    telegram_update_to_message_envelope,
)
from .fast_mode import FastModeStateStore, fast_status_message, parse_fast_command
from .feedback import FeedbackJsonlStore, parse_csat_callback_data
from .progress import ProgressUpdateConfig

logger = logging.getLogger(__name__)

_TEXT_ATTACHMENT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".csv", ".tsv", ".log", ".ini", ".cfg", ".conf", ".env", ".py", ".js",
    ".jsx", ".ts", ".tsx", ".rs", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash",
    ".zsh", ".fish", ".sql", ".html", ".css", ".xml", ".sol", ".move",
}
_TEXT_ATTACHMENT_MIME_PREFIXES = ("text/",)
_TEXT_ATTACHMENT_MIME_TYPES = {
    "application/json",
    "application/jsonl",
    "application/x-ndjson",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/xml",
    "application/javascript",
    "application/typescript",
    "application/x-sh",
}
_IMAGE_ATTACHMENT_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_MAX_TEXT_ATTACHMENT_BYTES = 256 * 1024
_MAX_TEXT_ATTACHMENT_CHARS = 16000
_MAX_IMAGE_ATTACHMENT_BYTES = 20 * 1024 * 1024


class TelegramBotRuntimeError(RuntimeError):
    """Domain error for token-based Telegram Bot runtime."""


@dataclass(frozen=True)
class TelegramBotConfig:
    """Config for Telegram Bot API calls."""

    bot_token: str
    api_base: str = "https://api.telegram.org"

    @classmethod
    def from_env(cls) -> "TelegramBotConfig":
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            raise TelegramBotRuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")

        api_base = (os.getenv("TELEGRAM_BOT_API_BASE") or "https://api.telegram.org").strip()
        if not api_base:
            api_base = "https://api.telegram.org"
        return cls(bot_token=token, api_base=api_base.rstrip("/"))


class TelegramBotAPI:
    """Small sync client over Telegram Bot API."""

    def __init__(
        self,
        config: TelegramBotConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self._cfg = config
        self._session = session or requests.Session()
        self._call_lock = threading.Lock()
        self.last_send_results: list[dict[str, Any]] = []

    @property
    def api_base(self) -> str:
        return self._cfg.api_base

    def _method_url(self, method: str) -> str:
        return f"{self._cfg.api_base}/bot{self._cfg.bot_token}/{method}"

    def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> Any:
        try:
            with self._call_lock:
                resp = self._session.post(
                    self._method_url(method),
                    json=payload or {},
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                body = resp.json()
        except requests.RequestException as exc:
            detail = _redact_telegram_error(str(exc), self._cfg.bot_token)
            raise TelegramBotRuntimeError(
                f"Telegram Bot API request failed for `{method}`: {detail}"
            ) from None
        except ValueError as exc:
            raise TelegramBotRuntimeError(
                f"Telegram Bot API returned non-JSON response for `{method}`."
            ) from None

        if not isinstance(body, dict):
            raise TelegramBotRuntimeError(
                f"Telegram Bot API malformed response for `{method}`: expected object."
            )
        if not bool(body.get("ok", False)):
            description = str(body.get("description", "unknown error"))
            raise TelegramBotRuntimeError(
                f"Telegram Bot API `{method}` failed: {description}"
            )
        return body.get("result")

    def get_me(self) -> dict[str, Any]:
        result = self.call("getMe")
        if isinstance(result, dict):
            return result
        raise TelegramBotRuntimeError("Telegram Bot API `getMe` returned invalid shape.")

    def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout_s: int = 25,
        limit: int = 20,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": max(0, int(timeout_s)),
            "limit": max(1, min(int(limit), 100)),
            "allowed_updates": allowed_updates
            or ["message", "edited_message", "channel_post", "edited_channel_post", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = int(offset)

        result = self.call(
            "getUpdates",
            payload=payload,
            timeout_s=max(10.0, float(timeout_s) + 10.0),
        )
        if not isinstance(result, list):
            raise TelegramBotRuntimeError(
                "Telegram Bot API `getUpdates` returned invalid shape."
            )
        return [row for row in result if isinstance(row, dict)]

    def send_request(
        self,
        *,
        method: str,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> Any:
        return self.call(method=method, payload=payload, timeout_s=timeout_s)

    def get_file(self, file_id: str) -> dict[str, Any]:
        result = self.call("getFile", {"file_id": str(file_id)})
        if isinstance(result, dict):
            return result
        raise TelegramBotRuntimeError("Telegram Bot API `getFile` returned invalid shape.")

    def download_file(self, file_path: str, *, timeout_s: float = 60.0) -> bytes:
        url = f"{self._cfg.api_base}/file/bot{self._cfg.bot_token}/{str(file_path).lstrip('/')}"
        try:
            with self._call_lock:
                resp = self._session.get(url, timeout=timeout_s)
                resp.raise_for_status()
                return resp.content
        except requests.RequestException as exc:
            detail = _redact_telegram_error(str(exc), self._cfg.bot_token)
            raise TelegramBotRuntimeError(f"Telegram file download failed: {detail}") from None

    def send_chat_action(
        self,
        *,
        chat_id: str,
        action: str = "typing",
        timeout_s: float = 10.0,
    ) -> Any:
        return self.send_request(
            method="sendChatAction",
            payload={"chat_id": chat_id, "action": action},
            timeout_s=timeout_s,
        )

    def send_requests(self, requests_payloads: list[dict[str, Any]]) -> int:
        sent = 0
        self.last_send_results = []
        for req in requests_payloads:
            method = str(req.get("method", "sendMessage"))
            payload = req.get("payload")
            if not isinstance(payload, dict):
                raise TelegramBotRuntimeError("Invalid send request payload: expected object.")
            result = self._send_request_with_plain_fallback(method=method, payload=payload)
            self.last_send_results.append(_sent_result_row(method=method, result=result))
            sent += 1
        return sent

    def _send_request_with_plain_fallback(
        self,
        *,
        method: str,
        payload: dict[str, Any],
    ) -> Any:
        try:
            return self.send_request(method=method, payload=payload)
        except TelegramBotRuntimeError as first_exc:
            if method != "sendMessage":
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("parse_mode", None)
            fallback_payload.pop("entities", None)
            if fallback_payload != payload:
                try:
                    logger.warning("Telegram sendMessage failed with rich formatting; retrying as plain text.")
                    return self.send_request(method=method, payload=fallback_payload)
                except TelegramBotRuntimeError:
                    pass

            detached_payload = dict(fallback_payload)
            detached_payload.pop("reply_to_message_id", None)
            if detached_payload != fallback_payload:
                logger.warning("Telegram sendMessage failed as reply; retrying detached plain text.")
                return self.send_request(method=method, payload=detached_payload)
            raise first_exc


@dataclass(frozen=True)
class TelegramUpdateOffsetStore:
    """File-backed checkpoint for Telegram `update_id + 1` offset."""

    path: Path

    def load(self) -> int | None:
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            value = int(raw)
            return value if value >= 0 else None
        except Exception:
            return None

    def save(self, offset: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(int(offset)), encoding="utf-8")


GraphRunner = Callable[[dict[str, Any]], dict[str, Any]]


class _TelegramTypingProgress:
    """Keep Telegram's built-in typing indicator alive while the graph runs."""

    def __init__(
        self,
        *,
        api: Any,
        chat_id: str | None,
        enabled: bool,
        interval_s: float = 4.0,
    ) -> None:
        self._api = api
        self._chat_id = str(chat_id or "").strip()
        self._enabled = bool(enabled and self._chat_id)
        self._interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._send_once()
        self._thread = threading.Thread(
            target=self._run,
            name="telegram-typing-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._send_once()

    def _send_once(self) -> None:
        try:
            if hasattr(self._api, "send_chat_action"):
                self._api.send_chat_action(chat_id=self._chat_id, action="typing", timeout_s=10.0)
            else:
                self._api.send_request(
                    method="sendChatAction",
                    payload={"chat_id": self._chat_id, "action": "typing"},
                    timeout_s=10.0,
                )
        except Exception as exc:
            logger.debug("telegram typing indicator failed chat_id=%s error=%s", self._chat_id, exc)


class _TelegramMessageProgress:
    """Send occasional visible progress messages while the graph runs."""

    def __init__(
        self,
        *,
        api: Any,
        chat_id: str | None,
        envelope: dict[str, Any],
        config: ProgressUpdateConfig,
        locale: str,
        enabled: bool,
    ) -> None:
        self._api = api
        self._chat_id = str(chat_id or "").strip()
        self._envelope = envelope
        self._config = config
        self._locale = str(locale or "zh-CN")
        self._enabled = bool(enabled and self._chat_id and config.should_run)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="telegram-message-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        if self._stop.wait(self._config.first_after_s):
            return
        for idx in range(self._config.max_updates):
            self._send_once(self._config.message_for(idx, locale=self._locale))
            if idx >= self._config.max_updates - 1:
                return
            if self._stop.wait(self._config.interval_s):
                return

    def _send_once(self, text: str) -> None:
        try:
            req = _send_text_request(
                chat_id=self._chat_id,
                text=text,
                envelope=self._envelope,
                reply_to_message_id=str(self._envelope.get("message_id") or "") or None,
            )
            self._api.send_request(
                method=str(req.get("method", "sendMessage")),
                payload=req["payload"],
                timeout_s=10.0,
            )
        except Exception as exc:
            logger.debug("telegram progress message failed chat_id=%s error=%s", self._chat_id, exc)


class TelegramPollingGateway:
    """Token-based polling gateway that runs graph and sends replies."""

    def __init__(
        self,
        *,
        api: TelegramBotAPI,
        graph_runner: GraphRunner,
        offset_store: TelegramUpdateOffsetStore | None = None,
        render_mode: str = "markdown",
        append_csat: bool = False,
        allowed_chat_ids: set[str] | None = None,
        allowed_thread_ids: set[str] | None = None,
        allowed_thread_ids_by_chat: dict[str, set[str]] | None = None,
        feedback_store: FeedbackJsonlStore | None = None,
        debug_log_file: str | Path | None = None,
        memory_service: Any | None = None,
        memory_context_limit: int = 20,
        mention_only_in_group: bool = True,
        respond_to_bot_replies: bool = True,
        bot_user_id: str | None = None,
        bot_username: str | None = None,
        chat_id_command_enabled: bool = False,
        chat_id_command_allowed_user_ids: set[str] | None = None,
        chat_id_command_private_response: bool = True,
        target_elapsed_ms: int = 30000,
        max_elapsed_ms: int = 90000,
        progress_update_config: ProgressUpdateConfig | None = None,
        fast_mode_store: FastModeStateStore | None = None,
        max_worker_threads: int = 1,
        attachment_download_dir: str | Path = "data/telegram_bot/attachments",
    ) -> None:
        self._api = api
        self._graph_runner = graph_runner
        self._offset_store = offset_store
        self._next_offset = offset_store.load() if offset_store else None
        self._render_mode = "plain" if render_mode == "plain" else "markdown"
        self._append_csat = append_csat
        self._allowed_chat_ids = set(allowed_chat_ids or [])
        self._allowed_thread_ids = set(allowed_thread_ids or [])
        self._allowed_thread_ids_by_chat = {
            str(chat_id): {str(thread_id) for thread_id in thread_ids}
            for chat_id, thread_ids in (allowed_thread_ids_by_chat or {}).items()
        }
        self._feedback_store = feedback_store
        debug_path = str(debug_log_file or "").strip()
        self._debug_log_file = Path(debug_path).expanduser() if debug_path else None
        self._memory_service = memory_service
        self._memory_context_limit = max(1, min(int(memory_context_limit or 20), 100))
        self._mention_only_in_group = bool(mention_only_in_group)
        self._respond_to_bot_replies = bool(respond_to_bot_replies)
        self._bot_user_id = str(bot_user_id or "").strip()
        self._bot_username = str(bot_username or "").lstrip("@").strip()
        self._chat_id_command_enabled = bool(chat_id_command_enabled)
        self._chat_id_command_allowed_user_ids = {
            str(user_id).strip()
            for user_id in (chat_id_command_allowed_user_ids or set())
            if str(user_id).strip()
        }
        self._chat_id_command_private_response = bool(chat_id_command_private_response)
        self._target_elapsed_ms = max(0, int(target_elapsed_ms or 0))
        self._max_elapsed_ms = max(0, int(max_elapsed_ms or 0))
        self._progress_update_config = progress_update_config or ProgressUpdateConfig(enabled=False)
        self._fast_mode_store = fast_mode_store or FastModeStateStore()
        self._max_worker_threads = max(1, min(int(max_worker_threads or 1), 32))
        self._attachment_download_dir = Path(attachment_download_dir).expanduser()
        self._debug_write_lock = threading.Lock()
        self._feedback_lock = threading.Lock()

    @property
    def next_offset(self) -> int | None:
        return self._next_offset

    def set_next_offset(self, offset: int) -> None:
        self._next_offset = max(0, int(offset))
        if self._offset_store:
            self._offset_store.save(self._next_offset)

    def poll_once(
        self,
        *,
        timeout_s: int = 25,
        limit: int = 20,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        updates = self._api.get_updates(
            offset=self._next_offset,
            timeout_s=timeout_s,
            limit=limit,
        )
        if not updates:
            logger.debug("telegram poll_once: no updates")
            return []

        rows: list[dict[str, Any]] = []
        max_seen: int | None = None
        indexed_updates: list[tuple[int, dict[str, Any]]] = []
        for idx, update in enumerate(updates):
            uid = _extract_update_id(update)
            if uid is not None:
                max_seen = uid if max_seen is None else max(max_seen, uid)
            indexed_updates.append((idx, update))

        if self._max_worker_threads <= 1 or len(indexed_updates) <= 1:
            rows = [self._process_update_safely(update, dry_run=dry_run) for _, update in indexed_updates]
        else:
            rows_by_index: list[dict[str, Any] | None] = [None] * len(indexed_updates)
            groups = _group_updates_by_conversation(indexed_updates)
            worker_count = min(self._max_worker_threads, len(groups))
            logger.debug(
                "telegram poll_once concurrent processing updates=%d groups=%d workers=%d",
                len(indexed_updates),
                len(groups),
                worker_count,
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="telegram-update-worker",
            ) as pool:
                futures = [
                    pool.submit(self._process_update_group, group, dry_run=dry_run)
                    for group in groups
                ]
                for future in as_completed(futures):
                    for idx, row in future.result():
                        rows_by_index[idx] = row
            rows = [row for row in rows_by_index if row is not None]

        if max_seen is not None:
            self._next_offset = max_seen + 1
            if self._offset_store:
                self._offset_store.save(self._next_offset)
        logger.debug("telegram poll_once: updates=%d next_offset=%s", len(rows), self._next_offset)
        return rows

    def _process_update_group(
        self,
        group: list[tuple[int, dict[str, Any]]],
        *,
        dry_run: bool,
    ) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        for idx, update in group:
            rows.append((idx, self._process_update_safely(update, dry_run=dry_run)))
        return rows

    def _process_update_safely(
        self,
        update: dict[str, Any],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        uid = _extract_update_id(update)
        try:
            return self.process_update(update, dry_run=dry_run)
        except Exception as exc:
            logger.exception(
                "telegram process_update failed; advancing offset to avoid duplicate LLM calls "
                "update_id=%s error=%s",
                uid,
                exc,
            )
            return {
                "update_id": uid,
                "chat_id": _extract_chat_id_from_update(update),
                "ignored": False,
                "reason": "process_update_failed",
                "request_id": None,
                "sent_count": 0,
                "error": str(exc),
            }

    def process_update(
        self,
        update: dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        update_id = _extract_update_id(update)
        raw_chat_id = _extract_chat_id_from_update(update)
        callback = _extract_callback_query(update)
        if isinstance(callback, dict):
            return self._process_callback_query(
                update_id=update_id,
                callback=callback,
                raw_chat_id=raw_chat_id,
                dry_run=dry_run,
            )

        msg = _extract_message_payload(update)
        if isinstance(msg, dict):
            sender = msg.get("from")
            if isinstance(sender, dict) and bool(sender.get("is_bot")):
                return {
                    "update_id": update_id,
                    "chat_id": raw_chat_id,
                    "ignored": True,
                    "reason": "bot_sender",
                    "request_id": None,
                    "sent_count": 0,
                }

        try:
            envelope = telegram_update_to_message_envelope(update)
        except ValueError:
            return {
                "update_id": update_id,
                "chat_id": raw_chat_id,
                "ignored": True,
                "reason": "unsupported_update",
                "request_id": None,
                "sent_count": 0,
            }

        chat_id_command = _is_chat_id_command(
            msg,
            envelope=envelope,
            bot_username=self._bot_username,
        )
        if chat_id_command and not self._is_chat_id_command_authorized(envelope):
            return {
                "update_id": update_id,
                "chat_id": raw_chat_id,
                "ignored": True,
                "reason": (
                    "chat_id_command_disabled"
                    if not self._chat_id_command_enabled
                    else "chat_id_command_not_allowed"
                ),
                "request_id": None,
                "sent_count": 0,
            }
        chat_id = (
            str(envelope.get("context", {}).get("channel_id") or "")
            or str(envelope.get("context", {}).get("guild_id") or "")
        )
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids and not chat_id_command:
            return {
                "update_id": update_id,
                "chat_id": chat_id or raw_chat_id,
                "ignored": True,
                "reason": "chat_not_allowed",
                "request_id": None,
                "sent_count": 0,
            }

        thread_id = _thread_id_from_envelope(envelope)
        if not chat_id_command and not self._is_thread_allowed(chat_id=chat_id or raw_chat_id, thread_id=thread_id):
            return {
                "update_id": update_id,
                "chat_id": chat_id or raw_chat_id,
                "thread_id": thread_id,
                "ignored": True,
                "reason": "thread_not_allowed",
                "request_id": None,
                "sent_count": 0,
            }

        if chat_id_command:
            return self._process_chat_id_command(
                update_id=update_id,
                envelope=envelope,
                message=msg,
                chat_id=chat_id or raw_chat_id,
                dry_run=dry_run,
            )

        if _is_feedback_command(envelope):
            return self._process_feedback_command(
                update_id=update_id,
                envelope=envelope,
                chat_id=chat_id or raw_chat_id,
                dry_run=dry_run,
            )

        if not self._should_process_message(update=update, envelope=envelope):
            logger.debug(
                "telegram ignored not mentioned update_id=%s chat_id=%s",
                update_id,
                chat_id or raw_chat_id,
            )
            return {
                "update_id": update_id,
                "chat_id": chat_id or raw_chat_id,
                "ignored": True,
                "reason": "not_mentioned",
                "request_id": None,
                "sent_count": 0,
            }

        fast_command = parse_fast_command(
            str(envelope.get("command", "") or ""),
            str(envelope.get("command_args", "") or ""),
        )
        if fast_command is not None:
            return self._process_fast_command(
                update_id=update_id,
                envelope=envelope,
                chat_id=chat_id or raw_chat_id,
                action=fast_command.action,
                message=fast_command.message,
                dry_run=dry_run,
            )

        state = message_envelope_to_graph_state(envelope)
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
        self._prepare_message_attachments(envelope=envelope, request_id=str(state.get("request_id", "unknown")))
        self._attach_recent_memory_context(state=state, envelope=envelope)
        self._write_memory_event(envelope=envelope, role="user", content=str(envelope.get("content", "") or ""))
        request_id = str(state.get("request_id", "unknown"))

        with log_request_context(request_id):
            typing_progress = _TelegramTypingProgress(
                api=self._api,
                chat_id=chat_id or raw_chat_id,
                enabled=not dry_run,
            )
            message_progress = _TelegramMessageProgress(
                api=self._api,
                chat_id=chat_id or raw_chat_id,
                envelope=envelope,
                config=self._progress_update_config,
                locale=str(state.get("locale") or envelope.get("locale_hint") or "zh-CN"),
                enabled=not dry_run,
            )
            typing_progress.start()
            message_progress.start()
            try:
                result = self._graph_runner(state)
            except Exception:
                logger.exception(
                    "telegram graph_runner failed chat_id=%s update_id=%s",
                    chat_id or raw_chat_id,
                    update_id,
                )
                result = {
                    "_final_response": {
                        "request_id": str(state.get("request_id", "unknown")),
                        "text": "处理请求时发生错误，请稍后重试。",
                        "citations": [],
                    }
                }
            finally:
                message_progress.stop()
                typing_progress.stop()
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
        send_reqs = outbound_message_to_telegram_requests(outbound)
        sent_count = 0
        send_results: list[dict[str, Any]] = []
        if not dry_run:
            sent_count = self._api.send_requests(send_reqs)
            send_results = _api_last_send_results(self._api)
            if self._feedback_store is not None:
                with self._feedback_lock:
                    self._feedback_store.append_answer(
                        _build_answer_feedback_record(
                            result=result,
                            state=state,
                            envelope=envelope,
                            outbound=outbound,
                            sent_count=sent_count,
                            has_csat=bool(self._append_csat and send_reqs),
                        )
                    )
        self._write_debug_event(
            update=update,
            update_id=update_id,
            chat_id=chat_id or raw_chat_id,
            envelope=envelope,
            state=state,
            result=result,
            outbound=outbound,
            send_reqs=send_reqs,
            send_results=send_results,
            sent_count=sent_count if not dry_run else len(send_reqs),
            dry_run=dry_run,
        )
        logger.info(
            "telegram processed update_id=%s chat_id=%s request_id=%s sent=%d dry_run=%s",
            update_id,
            chat_id or raw_chat_id,
            request_id,
            sent_count if not dry_run else len(send_reqs),
            bool(dry_run),
        )

        return {
            "update_id": update_id,
            "chat_id": chat_id or raw_chat_id,
            "ignored": False,
            "reason": "",
            "request_id": state.get("request_id"),
            "sent_count": sent_count if not dry_run else len(send_reqs),
        }

    def _process_fast_command(
        self,
        *,
        update_id: int | None,
        envelope: dict[str, Any],
        chat_id: str | None,
        action: str,
        message: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
        user_id = str(context.get("user_id", "") or "")
        platform = str(context.get("platform", "telegram") or "telegram")
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

        if not dry_run:
            self._api.send_requests(
                [_send_text_request(chat_id=chat_id, text=text, envelope=envelope)]
            )
        return {
            "update_id": update_id,
            "chat_id": chat_id,
            "ignored": False,
            "reason": "fast_command",
            "request_id": None,
            "sent_count": 0 if dry_run else 1,
            "fast_action": action,
        }

    def _process_chat_id_command(
        self,
        *,
        update_id: int | None,
        envelope: dict[str, Any],
        message: dict[str, Any],
        chat_id: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        chat = message.get("chat")
        chat = chat if isinstance(chat, dict) else {}
        chat_type = str(chat.get("type", "") or "unknown")
        chat_title = _telegram_chat_display_name(chat)
        thread_id = _thread_id_from_envelope(envelope)
        context = envelope.get("context", {})
        user_id = str(context.get("user_id", "") or "").strip() if isinstance(context, dict) else ""
        response_chat_id = user_id if self._chat_id_command_private_response else chat_id
        lines = [
            f"Chat title: {chat_title}",
            f"Chat type: {chat_type}",
            f"Chat ID: {chat_id or 'unknown'}",
        ]
        if thread_id:
            lines.append(f"Topic ID: {thread_id}")
        text = "\n".join(lines)
        if not dry_run:
            self._api.send_requests(
                [
                    _send_text_request(
                        chat_id=response_chat_id,
                        text=text,
                        envelope=envelope,
                        reply_to_message_id=(
                            None
                            if self._chat_id_command_private_response
                            else str(envelope.get("message_id", "")) or None
                        ),
                        include_thread_id=not self._chat_id_command_private_response,
                    )
                ]
            )
        return {
            "update_id": update_id,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "chat_title": chat_title,
            "response_chat_id": response_chat_id,
            "response_mode": "private" if self._chat_id_command_private_response else "same_chat",
            "ignored": False,
            "reason": "chat_id_command",
            "request_id": None,
            "sent_count": 0 if dry_run else 1,
        }

    def _is_chat_id_command_authorized(self, envelope: dict[str, Any]) -> bool:
        if not self._chat_id_command_enabled:
            return False
        context = envelope.get("context", {})
        if not isinstance(context, dict):
            return False
        user_id = str(context.get("user_id", "") or "").strip()
        return bool(user_id and user_id in self._chat_id_command_allowed_user_ids)

    def _consume_fast_mode(self, envelope: dict[str, Any]) -> bool:
        context = envelope.get("context", {}) if isinstance(envelope.get("context"), dict) else {}
        return self._fast_mode_store.consume_next(
            platform=str(context.get("platform", "telegram") or "telegram"),
            user_id=str(context.get("user_id", "") or ""),
        )

    def _prepare_message_attachments(self, *, envelope: dict[str, Any], request_id: str) -> None:
        attachments = envelope.get("attachments", [])
        if not isinstance(attachments, list) or not attachments:
            return

        text_blocks: list[str] = []
        image_names: list[str] = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            self._prepare_single_attachment(
                attachment=attachment,
                request_id=request_id,
                text_blocks=text_blocks,
                image_names=image_names,
            )

        content = str(envelope.get("content", "") or "").strip()
        if not content and (text_blocks or image_names):
            content = "Read the uploaded files or images and answer based on their contents."
        if text_blocks:
            content = (content + "\n\n" if content else "") + "\n\n".join(text_blocks)
        if image_names:
            image_note = "The user's uploaded images are provided as visual input: " + ", ".join(image_names)
            content = (content + "\n\n" if content else "") + image_note
        envelope["content"] = content

    def _prepare_single_attachment(
        self,
        *,
        attachment: dict[str, Any],
        request_id: str,
        text_blocks: list[str],
        image_names: list[str],
    ) -> None:
        file_id = _attachment_file_id(attachment)
        if not file_id:
            return

        name = _safe_attachment_name(str(attachment.get("name", "") or "telegram_file"))
        mime_type = str(attachment.get("mime_type", "") or mimetypes.guess_type(name)[0] or "")
        is_image = _is_supported_image_attachment(attachment, name=name, mime_type=mime_type)
        is_text = _is_supported_text_attachment(attachment, name=name, mime_type=mime_type)
        if not is_image and not is_text:
            attachment["status"] = "unsupported"
            return

        declared_size = _int_like(attachment.get("file_size", 0))
        size_limit = _MAX_IMAGE_ATTACHMENT_BYTES if is_image else _MAX_TEXT_ATTACHMENT_BYTES
        if declared_size > size_limit:
            attachment["status"] = "too_large"
            return

        try:
            meta = self._api.get_file(file_id)
            file_path = str(meta.get("file_path", "") or "")
            if not file_path:
                attachment["status"] = "missing_file_path"
                return
            data = self._api.download_file(file_path)
        except Exception as exc:
            attachment["status"] = "download_failed"
            attachment["error"] = str(exc)[:160]
            logger.debug("telegram attachment download failed name=%s error=%s", name, exc)
            return

        if not data or len(data) > size_limit:
            attachment["status"] = "too_large" if data else "empty"
            return

        suffix = Path(file_path).suffix or Path(name).suffix or (".jpg" if is_image else ".txt")
        local_dir = self._attachment_download_dir / _safe_attachment_name(request_id)
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"{len(list(local_dir.iterdir())) + 1:02d}_{Path(name).stem[:48]}{suffix}"
            local_path.write_bytes(data)
        except OSError as exc:
            attachment["status"] = "save_failed"
            attachment["error"] = str(exc)[:160]
            return

        attachment["local_path"] = str(local_path)
        attachment["mime_type"] = mime_type or mimetypes.guess_type(local_path.name)[0] or ""
        attachment["file_size"] = str(len(data))
        attachment["status"] = "ready"

        if is_image:
            attachment["kind"] = "image"
            image_names.append(name)
            return

        decoded = _decode_text_attachment(data)
        if decoded is None:
            attachment["status"] = "decode_failed"
            return
        if len(decoded) > _MAX_TEXT_ATTACHMENT_CHARS:
            decoded = decoded[:_MAX_TEXT_ATTACHMENT_CHARS].rstrip() + "\n...[file content truncated]"
            attachment["truncated"] = "true"
        text_blocks.append(f"Uploaded text file `{name}` contents:\n```text\n{decoded}\n```")

    def _attach_recent_memory_context(
        self,
        *,
        state: dict[str, Any],
        envelope: dict[str, Any],
    ) -> None:
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
        platform = str(context.get("platform", "telegram") or "telegram")
        user_id = str(context.get("user_id", "") or "")
        if not user_id:
            return
        if not _should_attach_recent_context(envelope):
            state["recent_messages"] = []
            state["conversation_context"] = ""
            return
        try:
            rows = svc.list_recent_message_events(
                platform=platform,
                user_id=user_id,
                guild_id=_optional_str(context.get("guild_id")),
                channel_id=_optional_str(context.get("channel_id")),
                thread_id=_optional_str(context.get("thread_id")),
                limit=self._memory_context_limit,
            )
        except Exception as exc:
            logger.debug("telegram recent memory read failed user_id=%s error=%s", user_id, exc)
            return
        if isinstance(rows, list):
            state["recent_messages"] = rows
            recent_context = _format_recent_messages(rows)
            if reply_context and recent_context:
                state["conversation_context"] = f"{reply_context}\n{recent_context}"
            else:
                state["conversation_context"] = reply_context or recent_context

    def _write_memory_event(
        self,
        *,
        envelope: dict[str, Any],
        role: str,
        content: str,
    ) -> None:
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
                platform=str(context.get("platform", "telegram") or "telegram"),
                user_id=user_id,
                guild_id=_optional_str(context.get("guild_id")),
                channel_id=_optional_str(context.get("channel_id")),
                thread_id=_optional_str(context.get("thread_id")),
                role=role,
                content=text,
                created_ts_ms=_memory_event_ts_ms(envelope=envelope, role=role),
            )
        except Exception as exc:
            logger.debug(
                "telegram memory write failed role=%s user_id=%s error=%s",
                role,
                user_id,
                exc,
            )

    def _process_callback_query(
        self,
        *,
        update_id: int | None,
        callback: dict[str, Any],
        raw_chat_id: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        callback_id = str(callback.get("id", ""))
        msg = callback.get("message")
        chat_id = _extract_chat_id_from_message(msg) if isinstance(msg, dict) else raw_chat_id
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            return {
                "update_id": update_id,
                "chat_id": chat_id,
                "ignored": True,
                "reason": "chat_not_allowed",
                "request_id": None,
                "sent_count": 0,
            }

        context = _feedback_context_from_callback(callback)
        thread_id = str(context.get("thread_id", "") or "")
        if not self._is_thread_allowed(chat_id=chat_id or raw_chat_id, thread_id=thread_id):
            return {
                "update_id": update_id,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "ignored": True,
                "reason": "thread_not_allowed",
                "request_id": None,
                "sent_count": 0,
            }

        data = str(callback.get("data", ""))
        try:
            parsed = parse_csat_callback_data(data)
        except ValueError:
            if not dry_run and callback_id:
                self._api.send_request(
                    method="answerCallbackQuery",
                    payload={"callback_query_id": callback_id, "text": "Invalid rating."},
                )
            return {
                "update_id": update_id,
                "chat_id": chat_id,
                "ignored": True,
                "reason": "invalid_callback",
                "request_id": None,
                "sent_count": 0,
            }

        sender = callback.get("from")
        if not isinstance(sender, dict):
            sender = {}
        answer_meta = None
        if self._feedback_store is not None:
            with self._feedback_lock:
                answer_meta = self._feedback_store.latest_answer(parsed.request_id)
        answer_fields = _feedback_fields_from_answer(answer_meta)
        preview = answer_fields.get("final_text_preview") or _message_text_preview(msg)
        record = {
            "request_id": parsed.request_id,
            "platform": "telegram",
            "chat_id": chat_id or "",
            "user_id": str(sender.get("id", "")),
            "score": parsed.score,
            "comment": "",
            "created_ts_ms": int(time.time() * 1000),
            **answer_fields,
            "final_text_preview": preview,
            **context,
        }
        if self._feedback_store is not None and not dry_run:
            with self._feedback_lock:
                record = self._feedback_store.append(record)

        if not dry_run and callback_id:
            duplicate = bool(record.get("is_duplicate_rating", False))
            self._api.send_request(
                method="answerCallbackQuery",
                payload={
                    "callback_query_id": callback_id,
                    "text": "Rating already recorded."
                    if duplicate
                    else f"Thanks. Rating recorded: {parsed.score}/5",
                    "show_alert": False,
                },
            )

        logger.info(
            "telegram csat update_id=%s chat_id=%s request_id=%s score=%s bad_case=%s dry_run=%s",
            update_id,
            chat_id,
            parsed.request_id,
            parsed.score,
            bool(record.get("is_bad_case", parsed.score <= 3)),
            bool(dry_run),
        )
        return {
            "update_id": update_id,
            "chat_id": chat_id,
            "ignored": False,
            "reason": "csat",
            "request_id": parsed.request_id,
            "sent_count": 0,
            "score": parsed.score,
            "is_bad_case": bool(record.get("is_bad_case", parsed.score <= 3)),
            "is_duplicate_rating": bool(record.get("is_duplicate_rating", False)),
        }

    def _is_thread_allowed(self, *, chat_id: str | None, thread_id: str) -> bool:
        chat_key = str(chat_id or "")
        if chat_key in self._allowed_thread_ids_by_chat:
            return str(thread_id) in self._allowed_thread_ids_by_chat[chat_key]
        if self._allowed_thread_ids:
            return str(thread_id) in self._allowed_thread_ids
        return True

    def _process_feedback_command(
        self,
        *,
        update_id: int | None,
        envelope: dict[str, Any],
        chat_id: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        args = str(envelope.get("command_args", "") or "").strip()
        request_id, comment = _parse_feedback_args(args)
        sender_context = envelope.get("context", {})
        if not isinstance(sender_context, dict):
            sender_context = {}

        if not request_id or not comment:
            text = "用法: /feedback <request_id> <comment>"
            if not dry_run:
                self._api.send_requests(
                    [_send_text_request(chat_id=chat_id, text=text, envelope=envelope)]
                )
            return {
                "update_id": update_id,
                "chat_id": chat_id,
                "ignored": True,
                "reason": "invalid_feedback_command",
                "request_id": request_id,
                "sent_count": 0 if dry_run else 1,
            }

        answer_meta = (
            self._feedback_store.latest_answer(request_id)
            if self._feedback_store is not None
            else None
        )
        record = {
            "request_id": request_id,
            "platform": "telegram",
            "chat_id": chat_id or "",
            "user_id": str(sender_context.get("user_id", "")),
            "comment": comment,
            "created_ts_ms": int(time.time() * 1000),
            **_context_ids(sender_context),
            **_feedback_fields_from_answer(answer_meta),
        }
        if self._feedback_store is not None and not dry_run:
            self._feedback_store.append_comment(record)

        text = f"Feedback recorded for {request_id}."
        if not dry_run:
            self._api.send_requests(
                [_send_text_request(chat_id=chat_id, text=text, envelope=envelope)]
            )
        return {
            "update_id": update_id,
            "chat_id": chat_id,
            "ignored": False,
            "reason": "feedback",
            "request_id": request_id,
            "sent_count": 0 if dry_run else 1,
        }

    def _write_debug_event(
        self,
        *,
        update: dict[str, Any],
        update_id: int | None,
        chat_id: str | None,
        envelope: dict[str, Any],
        state: dict[str, Any],
        result: dict[str, Any],
        outbound: dict[str, Any],
        send_reqs: list[dict[str, Any]],
        send_results: list[dict[str, Any]],
        sent_count: int,
        dry_run: bool,
    ) -> None:
        path = self._debug_log_file
        if path is None:
            return
        try:
            response = result.get("_final_response")
            if not isinstance(response, dict):
                response = {}
            context = envelope.get("context", {})
            if not isinstance(context, dict):
                context = {}
            event = {
                "created_ts_ms": int(time.time() * 1000),
                "update_id": update_id,
                "chat_id": str(chat_id or ""),
                "request_id": str(state.get("request_id") or response.get("request_id") or ""),
                "user_id": str(context.get("user_id", "")),
                "message_id": str(envelope.get("message_id", "")),
                "incoming_reply_to_message_id": str(envelope.get("reply_to_message_id", "")),
                "reply_to_role": str(envelope.get("reply_to_role", "")),
                "reply_to_content_preview": _preview_text(envelope.get("reply_to_content", ""), limit=500),
                "conversation_context_preview": _preview_text(state.get("conversation_context", ""), limit=900),
                "recent_messages_preview": _debug_recent_messages_preview(state.get("recent_messages")),
                "outbound_reply_to_message_id": str(outbound.get("reply_to_message_id", "")),
                "first_send_reply_to_message_id": _first_send_reply_to_message_id(send_reqs),
                "sent_message_ids": _sent_message_ids(send_results),
                "send_results": send_results[:10],
                "content_preview": _message_text_preview(_extract_message_payload(update), limit=240),
                "normalized_content": str(envelope.get("content", ""))[:240],
                "route_decision": str(result.get("_route_decision") or state.get("_route_decision") or ""),
                "llm_service_tier": str(result.get("_llm_service_tier") or state.get("_llm_service_tier") or ""),
                "retrieval_policy": str(result.get("retrieval_policy") or state.get("retrieval_policy") or ""),
                "reflection_decision": str(result.get("reflection_decision") or state.get("reflection_decision") or ""),
                "reflection_reasoning": str(result.get("reflection_reasoning") or state.get("reflection_reasoning") or "")[:500],
                "info_needs": _debug_info_needs(result.get("info_needs") or state.get("info_needs")),
                "tool_summary": str(result.get("_tool_execution_summary") or state.get("_tool_execution_summary") or ""),
                "tool_trace": _debug_tool_trace(result.get("_tool_execution_trace") or state.get("_tool_execution_trace")),
                "tool_calls": _infer_tool_calls(result=result, state=state),
                "evidence_count": _infer_evidence_count(result=result, state=state, response=response),
                "evidence_sources": _debug_evidence_sources(result.get("evidence") or state.get("evidence")),
                "citation_count": len(response.get("citations", [])) if isinstance(response.get("citations"), list) else 0,
                "graph_elapsed_ms": _int_like(result.get("_graph_elapsed_ms", 0)),
                "node_timings": _debug_node_timings(result.get("_node_timings") or state.get("_node_timings")),
                "llm_usage_summary": _debug_llm_usage_summary(result.get("_llm_usage_summary") or state.get("_llm_usage_summary")),
                "llm_trace": _debug_llm_trace(result.get("_llm_trace") or state.get("_llm_trace")),
                "time_budget": _debug_time_budget(result=result, state=state),
                "final_text_preview": _outbound_text_preview(outbound, limit=600),
                "sent_count": int(sent_count),
                "dry_run": bool(dry_run),
                "terminal_insufficient_evidence": bool(result.get("_terminal_insufficient_evidence") or state.get("_terminal_insufficient_evidence")),
                "direct_answer": bool(result.get("_direct_answer") or state.get("_direct_answer")),
                "ask_user_guard_reason": str(
                    result.get("_ask_user_guard_reason")
                    or response.get("ask_user_guard_reason")
                    or state.get("_ask_user_guard_reason")
                    or ""
                ),
            }
            with self._debug_write_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug("telegram debug event write failed: %s", exc)


    def _should_process_message(
        self,
        *,
        update: dict[str, Any],
        envelope: dict[str, Any],
    ) -> bool:
        msg = _extract_message_payload(update)
        if not isinstance(msg, dict):
            return False
        if not _is_group_chat(msg):
            return True
        if not self._mention_only_in_group:
            return True
        if _is_bot_command_for_this_bot(
            msg,
            envelope=envelope,
            bot_username=self._bot_username,
        ):
            return True
        if _has_bot_mention(msg, bot_username=self._bot_username):
            return True
        if self._respond_to_bot_replies and _is_reply_to_this_bot(
            msg,
            bot_user_id=self._bot_user_id,
            bot_username=self._bot_username,
        ):
            return True
        return False


def _extract_update_id(update: dict[str, Any]) -> int | None:
    raw = update.get("update_id")
    if isinstance(raw, int):
        return raw
    return None


def _redact_telegram_error(text: str, bot_token: str) -> str:
    redacted = str(text)
    token = str(bot_token or "").strip()
    if token:
        redacted = redacted.replace(token, "<telegram-bot-token>")
    redacted = re.sub(r"/bot[^/\s]+/", "/bot<telegram-bot-token>/", redacted)
    return redacted


def _extract_message_payload(update: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        value = update.get(key)
        if isinstance(value, dict):
            return value
    return None


def _extract_callback_query(update: dict[str, Any]) -> dict[str, Any] | None:
    value = update.get("callback_query")
    return value if isinstance(value, dict) else None


def _extract_chat_id_from_update(update: dict[str, Any]) -> str | None:
    msg = _extract_message_payload(update)
    if msg is None:
        callback = _extract_callback_query(update)
        if isinstance(callback, dict) and isinstance(callback.get("message"), dict):
            msg = callback["message"]
    if not isinstance(msg, dict):
        return None
    return _extract_chat_id_from_message(msg)


def _extract_chat_id_from_message(msg: dict[str, Any]) -> str | None:
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return None
    value = chat.get("id")
    if value is None:
        return None
    return str(value)


def _conversation_key_from_update(update: dict[str, Any]) -> str:
    chat_id = _extract_chat_id_from_update(update)
    user_id = _extract_user_id_from_update(update)
    if chat_id is None:
        callback = _extract_callback_query(update)
        if isinstance(callback, dict):
            callback_id = str(callback.get("id", "") or "")
            return f"callback:{callback_id or _extract_update_id(update) or 'unknown'}"
        return f"update:{_extract_update_id(update) or 'unknown'}"
    return f"chat:{chat_id}:user:{user_id or 'unknown'}"


def _extract_user_id_from_update(update: dict[str, Any]) -> str | None:
    msg = _extract_message_payload(update)
    if isinstance(msg, dict):
        sender = msg.get("from") or msg.get("sender_chat")
        if isinstance(sender, dict) and sender.get("id") is not None:
            return str(sender["id"])
    callback = _extract_callback_query(update)
    if isinstance(callback, dict):
        sender = callback.get("from")
        if isinstance(sender, dict) and sender.get("id") is not None:
            return str(sender["id"])
    return None


def _group_updates_by_conversation(
    indexed_updates: list[tuple[int, dict[str, Any]]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    groups_by_key: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    key_order: list[str] = []
    for item in indexed_updates:
        key = _conversation_key_from_update(item[1])
        if key not in groups_by_key:
            groups_by_key[key] = []
            key_order.append(key)
        groups_by_key[key].append(item)
    return [groups_by_key[key] for key in key_order]


def _message_text_preview(msg: Any, limit: int = 300) -> str:
    if not isinstance(msg, dict):
        return ""
    text = msg.get("text")
    if not isinstance(text, str):
        text = str(msg.get("caption", "") or "")
    return text[:limit]


def _message_text(msg: dict[str, Any]) -> str:
    text = msg.get("text")
    if isinstance(text, str):
        return text
    caption = msg.get("caption")
    if isinstance(caption, str):
        return caption
    return ""


def _telegram_chat_display_name(chat: dict[str, Any]) -> str:
    title = str(chat.get("title", "") or "").strip()
    if title:
        return _single_line_preview(title, limit=200)

    username = str(chat.get("username", "") or "").strip().lstrip("@")
    if username:
        return f"@{username}"

    name = " ".join(
        part
        for part in (
            str(chat.get("first_name", "") or "").strip(),
            str(chat.get("last_name", "") or "").strip(),
        )
        if part
    )
    return _single_line_preview(name, limit=200) if name else "Unnamed chat"


def _single_line_preview(value: str, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _is_group_chat(msg: dict[str, Any]) -> bool:
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return False
    return str(chat.get("type", "")).lower() in {"group", "supergroup", "channel"}


def _raw_command_target(text: str) -> tuple[str, str | None] | None:
    match = re.match(r"^/([A-Za-z0-9_]+)(?:@([A-Za-z0-9_]+))?(?:\s|$)", str(text or "").strip())
    if not match:
        return None
    return "/" + match.group(1).lower(), match.group(2)


def _is_bot_command_for_this_bot(
    msg: dict[str, Any],
    *,
    envelope: dict[str, Any],
    bot_username: str,
) -> bool:
    command = str(envelope.get("command", "") or "").split("@", 1)[0].lower()
    known_commands = {"/ask", "/help", "/start", "/feedback", "/fast", "/chatid"}
    if command not in known_commands:
        return False
    raw = _raw_command_target(_message_text(msg))
    if raw is None:
        return True
    _raw_command, target = raw
    if not target:
        return True
    if not bot_username:
        return True
    return target.lower() == bot_username.lower()


def _is_chat_id_command(
    msg: dict[str, Any] | None,
    *,
    envelope: dict[str, Any],
    bot_username: str,
) -> bool:
    if not isinstance(msg, dict):
        return False
    command = str(envelope.get("command", "") or "").split("@", 1)[0].lower()
    return command == "/chatid" and _is_bot_command_for_this_bot(
        msg,
        envelope=envelope,
        bot_username=bot_username,
    )


def _has_bot_mention(msg: dict[str, Any], *, bot_username: str) -> bool:
    text = _message_text(msg)
    if _has_bot_mention_entity(msg, bot_username=bot_username):
        return True
    if not text:
        return False
    if bot_username:
        return re.search(rf"@{re.escape(bot_username)}\b", text, flags=re.IGNORECASE) is not None
    return re.search(r"@\w+_Bot\b", text, flags=re.IGNORECASE) is not None


def _has_bot_mention_entity(msg: dict[str, Any], *, bot_username: str) -> bool:
    text = _message_text(msg)
    entities = msg.get("entities")
    if not isinstance(entities, list):
        entities = msg.get("caption_entities")
    if not isinstance(entities, list):
        return False

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("type", "") or "")
        if entity_type == "text_mention":
            user = entity.get("user")
            if isinstance(user, dict):
                username = str(user.get("username", "") or "").lstrip("@")
                if bot_username and username.lower() == bot_username.lower():
                    return True
                if bool(user.get("is_bot")) and not bot_username:
                    return True
        if entity_type != "mention":
            continue
        try:
            offset = int(entity.get("offset", 0))
            length = int(entity.get("length", 0))
        except (TypeError, ValueError):
            continue
        mention = text[offset: offset + length].lstrip("@")
        if bot_username:
            if mention.lower() == bot_username.lower():
                return True
        elif mention.lower().endswith("_bot"):
            return True
    return False


def _is_reply_to_this_bot(
    msg: dict[str, Any],
    *,
    bot_user_id: str,
    bot_username: str,
) -> bool:
    reply = msg.get("reply_to_message")
    if not isinstance(reply, dict):
        return False
    sender = reply.get("from")
    if not isinstance(sender, dict):
        return False
    if bot_user_id and str(sender.get("id", "") or "") == bot_user_id:
        return True
    username = str(sender.get("username", "") or "").lstrip("@")
    if bot_username and username.lower() == bot_username.lower():
        return True
    return False


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _attachment_file_id(attachment: dict[str, Any]) -> str:
    file_id = str(attachment.get("file_id", "") or "").strip()
    if file_id:
        return file_id
    url = str(attachment.get("url", "") or "")
    if url.startswith("tgfile://"):
        return url.removeprefix("tgfile://").strip()
    return ""


def _safe_attachment_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned[:96] or "telegram_file"


def _is_supported_image_attachment(attachment: dict[str, Any], *, name: str, mime_type: str) -> bool:
    suffix = Path(name).suffix.lower()
    kind = str(attachment.get("kind", "") or "")
    if mime_type.startswith("image/"):
        return suffix != ".pdf"
    return kind == "image" or suffix in _IMAGE_ATTACHMENT_EXTS


def _is_supported_text_attachment(attachment: dict[str, Any], *, name: str, mime_type: str) -> bool:
    if str(attachment.get("kind", "") or "") == "image":
        return False
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return False
    if suffix in _TEXT_ATTACHMENT_EXTS:
        return True
    if any(mime_type.startswith(prefix) for prefix in _TEXT_ATTACHMENT_MIME_PREFIXES):
        return True
    return mime_type in _TEXT_ATTACHMENT_MIME_TYPES


def _decode_text_attachment(data: bytes) -> str | None:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = data.decode(encoding)
            text = text.replace("\x00", "")
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
    if len(text) > limit_chars:
        text = text[-limit_chars:].lstrip()
    return text


def _reply_context_from_envelope(envelope: dict[str, Any]) -> str:
    reply_text = re.sub(r"\s+", " ", str(envelope.get("reply_to_content", "") or "")).strip()
    reply_id = str(envelope.get("reply_to_message_id", "") or "").strip()
    if not reply_text:
        if reply_id:
            role = str(envelope.get("reply_to_role", "") or "").strip().lower()
            if role in {"assistant", "bot"}:
                label = "assistant"
            elif role == "user":
                label = "user"
            else:
                label = "replied_message"
            return (
                f"Current message replies to a {label} message (message_id={reply_id}), "
                "but the platform did not provide the replied message content. Do not guess it "
                "from ordinary history; if this short follow-up cannot be understood on its own, "
                "ask the user to provide or resend the replied content."
            )
        return ""
    if len(reply_text) > 700:
        reply_text = reply_text[:700].rstrip() + "..."
    role = str(envelope.get("reply_to_role", "") or "").strip().lower()
    if role in {"assistant", "bot"}:
        label = "assistant"
    elif role == "user":
        label = "user"
    else:
        label = "replied_message"
    return f"Current message replies to this {label} message: {reply_text}"


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
    if not normalized:
        return False
    markers = (
        "上文",
        "上下文",
        "刚才",
        "刚刚",
        "之前",
        "前面",
        "上面",
        "继续",
        "接着",
        "延续",
        "它",
        "这个",
        "那个",
        "那",
        "同样",
        "换成",
        "再来",
        "再写",
        "我刚才",
        "你刚才",
    )
    return any(marker in normalized for marker in markers)


def _is_standalone_named_question(text: str) -> bool:
    normalized = _normalized_message_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in ("上文", "上下文", "刚才", "之前", "继续", "接着")):
        return False
    subjects = (
        "ckb",
        "nervos",
        "fiber",
        "ccc",
        "rgb++",
        "spore",
        "molecule",
        "cell",
        "utxo",
        "commonknowledgebase",
    )
    intents = (
        "是什么",
        "什么是",
        "怎么",
        "如何",
        "为什么",
        "区别",
        "原理",
        "流程",
        "构建",
        "开发",
        "教程",
        "介绍",
        "解释",
        "应用",
        "代码",
        "示例",
        "例子",
    )
    return any(subject in normalized for subject in subjects) and any(
        intent in normalized for intent in intents
    )


def _looks_like_short_followup(text: str) -> bool:
    normalized = _normalized_message_text(text)
    if not normalized:
        return False
    if _is_standalone_named_question(text):
        return False
    if normalized in {"你是谁", "help", "/help", "帮助"}:
        return False
    return len(normalized) <= 48


def _should_attach_recent_context(envelope: dict[str, Any]) -> bool:
    """Only inject same-user history when the current turn actually depends on it."""
    if envelope.get("reply_to_message_id"):
        return True
    text = str(envelope.get("content", "") or "")
    if _is_standalone_named_question(text):
        return False
    return _asks_for_recent_context(text) or _looks_like_short_followup(text)


def _memory_event_ts_ms(*, envelope: dict[str, Any], role: str) -> int:
    if role == "user":
        try:
            return int(envelope.get("ts_ms") or time.time() * 1000)
        except (TypeError, ValueError):
            return int(time.time() * 1000)
    return int(time.time() * 1000)


def _is_feedback_command(envelope: dict[str, Any]) -> bool:
    command = str(envelope.get("command", "") or "").strip().lower()
    return command.split("@", 1)[0] == "/feedback"


def _parse_feedback_args(args: str) -> tuple[str, str]:
    text = str(args or "").strip()
    if not text:
        return "", ""
    request_id, _, comment = text.partition(" ")
    return request_id.strip(), comment.strip()


def _feedback_context_from_callback(callback: dict[str, Any]) -> dict[str, str]:
    msg = callback.get("message")
    if not isinstance(msg, dict):
        return {}
    context: dict[str, str] = {}
    chat = msg.get("chat")
    if isinstance(chat, dict):
        chat_id = chat.get("id")
        chat_type = str(chat.get("type", "")).lower()
        if chat_id is not None:
            context["channel_id"] = str(chat_id)
            if chat_type in {"group", "supergroup", "channel"}:
                context["guild_id"] = str(chat_id)
    thread_id = msg.get("message_thread_id")
    if thread_id is not None:
        context["thread_id"] = str(thread_id)
    return context


def _thread_id_from_envelope(envelope: dict[str, Any]) -> str:
    context = envelope.get("context", {})
    if isinstance(context, dict):
        value = context.get("thread_id")
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _first_send_reply_to_message_id(send_reqs: list[dict[str, Any]]) -> str:
    for req in send_reqs:
        if not isinstance(req, dict):
            continue
        payload = req.get("payload")
        if isinstance(payload, dict) and payload.get("reply_to_message_id") is not None:
            return str(payload.get("reply_to_message_id"))
    return ""


def _sent_result_row(*, method: str, result: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"method": str(method)}
    if isinstance(result, dict):
        if result.get("message_id") is not None:
            row["message_id"] = str(result.get("message_id"))
        chat = result.get("chat")
        if isinstance(chat, dict) and chat.get("id") is not None:
            row["chat_id"] = str(chat.get("id"))
        if result.get("message_thread_id") is not None:
            row["message_thread_id"] = str(result.get("message_thread_id"))
        if result.get("date") is not None:
            row["date"] = str(result.get("date"))
    return row


def _api_last_send_results(api: Any) -> list[dict[str, Any]]:
    rows = getattr(api, "last_send_results", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _sent_message_ids(send_results: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in send_results:
        if not isinstance(row, dict):
            continue
        message_id = str(row.get("message_id", "") or "").strip()
        if message_id:
            ids.append(message_id)
    return ids


def _debug_info_needs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "kind": str(item.get("kind", "")),
                "question": str(item.get("question", ""))[:240],
                "required": bool(item.get("required", False)),
            }
        )
    return rows


def _debug_tool_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[-20:]:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {
            "step_id": str(item.get("step_id", "")),
            "tool": str(item.get("tool", "")),
            "status": str(item.get("status", "")),
            "evidence_count": _int_like(item.get("evidence_count", 0)),
        }
        for key in ("error_code", "fallback_reason"):
            value_text = str(item.get(key, "") or "").strip()
            if value_text:
                row[key] = value_text[:160]
        filter_notes = item.get("filter_notes")
        if isinstance(filter_notes, list):
            row["filter_notes"] = [str(note)[:160] for note in filter_notes[:8]]
        rows.append(row)
    return rows


def _debug_evidence_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    counts: dict[tuple[str, str], int] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        source = str(payload.get("source") or item.get("source") or "")
        backend = str(payload.get("backend") or "")
        key = (source, backend)
        counts[key] = counts.get(key, 0) + 1
    rows = [
        {"source": source, "backend": backend, "count": count}
        for (source, backend), count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    ]
    return rows[:12]


def _debug_node_timings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[-40:]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "node": str(item.get("node", "")),
                "elapsed_ms": _int_like(item.get("elapsed_ms", 0)),
            }
        )
    return rows


def _debug_llm_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[-40:]:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        rows.append(
            {
                "node": str(item.get("node", "")),
                "kind": str(item.get("kind", "")),
                "model": str(item.get("model", "")),
                "tier": str(item.get("tier", "")),
                "reasoning_effort": str(item.get("reasoning_effort", "")),
                "service_tier": str(item.get("service_tier", "")),
                "json_mode": bool(item.get("json_mode", False)),
                "max_tokens": _int_like(item.get("max_tokens", 0)),
                "elapsed_ms": _int_like(item.get("elapsed_ms", 0)),
                "usage": {
                    "input_tokens": _int_like(usage.get("input_tokens", 0)),
                    "output_tokens": _int_like(usage.get("output_tokens", 0)),
                    "total_tokens": _int_like(usage.get("total_tokens", 0)),
                },
            }
        )
    return rows


def _debug_llm_usage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        "calls": _int_like(value.get("calls", 0)),
        "elapsed_ms": _int_like(value.get("elapsed_ms", 0)),
        "input_tokens": _int_like(value.get("input_tokens", 0)),
        "output_tokens": _int_like(value.get("output_tokens", 0)),
        "total_tokens": _int_like(value.get("total_tokens", 0)),
        "by_node": {},
    }
    by_node = value.get("by_node", {})
    if isinstance(by_node, dict):
        result["by_node"] = {
            str(node): {
                "calls": _int_like(row.get("calls", 0)) if isinstance(row, dict) else 0,
                "elapsed_ms": _int_like(row.get("elapsed_ms", 0)) if isinstance(row, dict) else 0,
                "input_tokens": _int_like(row.get("input_tokens", 0)) if isinstance(row, dict) else 0,
                "output_tokens": _int_like(row.get("output_tokens", 0)) if isinstance(row, dict) else 0,
                "total_tokens": _int_like(row.get("total_tokens", 0)) if isinstance(row, dict) else 0,
            }
            for node, row in by_node.items()
        }
    return result


def _debug_time_budget(*, result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        budget = {}
    started_ms = _int_like(result.get("_request_started_ts_ms") or state.get("_request_started_ts_ms"), 0)
    elapsed_ms = _int_like(result.get("_graph_elapsed_ms", 0))
    if elapsed_ms <= 0 and started_ms > 0:
        elapsed_ms = max(0, int(time.time() * 1000) - started_ms)
    target_ms = _int_like(budget.get("target_elapsed_ms", 0))
    max_ms = _int_like(budget.get("max_elapsed_ms", 0))
    return {
        "elapsed_ms": elapsed_ms,
        "target_elapsed_ms": target_ms,
        "max_elapsed_ms": max_ms,
        "remaining_target_ms": max(0, target_ms - elapsed_ms) if target_ms > 0 else 0,
        "remaining_max_ms": max(0, max_ms - elapsed_ms) if max_ms > 0 else 0,
    }


def _int_like(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _context_ids(context: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("guild_id", "channel_id", "thread_id"):
        value = context.get(key)
        if value is not None and str(value).strip():
            result[key] = str(value)
    return result


def _feedback_fields_from_answer(answer: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {
            "trace_summary": "",
            "tool_calls": 0,
            "evidence_count": 0,
            "final_text_preview": "",
        }
    return {
        "trace_summary": str(answer.get("trace_summary", "") or ""),
        "tool_calls": _int_or_zero(answer.get("tool_calls")),
        "evidence_count": _int_or_zero(answer.get("evidence_count")),
        "final_text_preview": str(answer.get("final_text_preview", "") or ""),
    }


def _build_answer_feedback_record(
    *,
    result: dict[str, Any],
    state: dict[str, Any],
    envelope: dict[str, Any],
    outbound: dict[str, Any],
    sent_count: int,
    has_csat: bool,
) -> dict[str, Any]:
    response = result.get("_final_response")
    if not isinstance(response, dict):
        response = {}
    context = envelope.get("context", {})
    if not isinstance(context, dict):
        context = {}
    request_id = str(state.get("request_id") or response.get("request_id") or "unknown")
    return {
        "request_id": request_id,
        "platform": "telegram",
        "chat_id": str(context.get("channel_id") or context.get("guild_id") or ""),
        "user_id": str(context.get("user_id", "")),
        "created_ts_ms": int(time.time() * 1000),
        "trace_summary": str(
            response.get("trace_summary")
            or result.get("_tool_execution_summary")
            or state.get("_tool_execution_summary")
            or ""
        ),
        "tool_calls": _infer_tool_calls(result=result, state=state),
        "evidence_count": _infer_evidence_count(result=result, state=state, response=response),
        "final_text_preview": _outbound_text_preview(outbound),
        "sent_count": int(sent_count),
        "has_csat": bool(has_csat),
        "answer_char_count": sum(
            int(seg.get("char_count", len(str(seg.get("text", "")))))
            for seg in outbound.get("segments", [])
            if isinstance(seg, dict)
        ),
        **_context_ids(context),
    }


def _infer_tool_calls(*, result: dict[str, Any], state: dict[str, Any]) -> int:
    for value in (
        result.get("_tool_calls_executed"),
        state.get("_tool_calls_executed"),
    ):
        if value is not None:
            return _int_or_zero(value)
    trace = result.get("_tool_execution_trace") or state.get("_tool_execution_trace")
    if isinstance(trace, list):
        return len(trace)
    return 0


def _infer_evidence_count(
    *,
    result: dict[str, Any],
    state: dict[str, Any],
    response: dict[str, Any],
) -> int:
    for value in (
        result.get("evidence_count"),
        state.get("evidence_count"),
    ):
        if value is not None:
            return _int_or_zero(value)
    evidence = result.get("evidence") or state.get("evidence")
    if isinstance(evidence, list):
        return len(evidence)
    citations = response.get("citations")
    if isinstance(citations, list):
        return len(citations)
    return 0


def _outbound_text_preview(outbound: dict[str, Any], limit: int = 300) -> str:
    chunks = []
    for seg in outbound.get("segments", []):
        if isinstance(seg, dict):
            chunks.append(str(seg.get("text", "") or ""))
    return "\n".join(chunks).strip()[:limit]


def _send_text_request(
    *,
    chat_id: str | None,
    text: str,
    envelope: dict[str, Any],
    reply_to_message_id: str | None = None,
    include_thread_id: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chat_id": _coerce_numeric_or_keep(chat_id or ""),
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = _coerce_numeric_or_keep(reply_to_message_id)
    context = envelope.get("context", {})
    if include_thread_id and isinstance(context, dict) and context.get("thread_id") is not None:
        payload["message_thread_id"] = _coerce_numeric_or_keep(context["thread_id"])
    return {"method": "sendMessage", "payload": payload}


def _coerce_numeric_or_keep(value: Any) -> Any:
    text = str(value)
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_outbound_from_graph_result(
    *,
    result: dict[str, Any],
    state: dict[str, Any],
    envelope: dict[str, Any],
    render_mode: str,
    append_csat: bool,
) -> dict[str, Any]:
    outbound = result.get("_outbound_message")
    if isinstance(outbound, dict) and isinstance(outbound.get("segments"), list):
        # Telegram replies should point to the user's current message. When a
        # user triggers the bot by replying to a previous bot answer, the
        # envelope also contains reply_to_message_id, but using that would make
        # Telegram render the bot as replying to itself.
        if envelope.get("message_id"):
            outbound["reply_to_message_id"] = str(envelope.get("message_id"))
        return outbound

    response = result.get("_final_response")
    if not isinstance(response, dict):
        response = {}
    response.setdefault("request_id", str(state.get("request_id", "unknown")))
    response.setdefault("text", "")
    response.setdefault("citations", [])

    out = format_response_to_outbound(
        response=response,
        context=envelope["context"],  # type: ignore[arg-type]
        render_mode="plain" if render_mode == "plain" else "markdown",  # type: ignore[arg-type]
        append_csat=bool(append_csat),
        reply_to_message_id=str(envelope.get("message_id") or "") or None,
    )
    if out.get("segments"):
        return out

    req_id = str(response.get("request_id", "unknown"))
    out["segments"] = [
        {
            "segment_id": f"{req_id}:0",
            "index": 0,
            "text": "(empty response)",
            "char_count": len("(empty response)"),
            "citation_labels": [],
        }
    ]
    return out
