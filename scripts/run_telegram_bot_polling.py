#!/usr/bin/env python3
"""Run token-based Telegram Bot polling gateway.

This is the online receive/send layer:
  Telegram Bot API getUpdates
    -> MessageEnvelope
    -> Full Graph (RAG + LLM)
    -> OutboundMessage
    -> Telegram Bot API sendMessage
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nervos_brain.graph_engine.full_graph import (  # noqa: E402
    FullGraphRuntime,
    build_full_graph,
    invoke_full_graph,
)
from nervos_brain.graph_engine.provider_registry import ProviderCapabilityRegistry  # noqa: E402
from nervos_brain.memory import (  # noqa: E402
    MemoryService,
    build_session_factory,
    init_memory_schema,
)
from nervos_brain.logging_system import setup_logging  # noqa: E402
from nervos_brain.pathing import load_project_config, resolve_project_path  # noqa: E402
from nervos_brain.retrieval import (  # noqa: E402
    build_configured_retriever,
)
from nervos_brain.tool_runtime.telegram_bot_runtime import (  # noqa: E402
    TelegramBotAPI,
    TelegramBotConfig,
    TelegramBotRuntimeError,
    TelegramPollingGateway,
    TelegramUpdateOffsetStore,
)
from nervos_brain.tool_runtime.fast_mode import FastModeStateStore  # noqa: E402
from nervos_brain.tool_runtime.feedback import FeedbackJsonlStore  # noqa: E402
from nervos_brain.tool_runtime.progress import ProgressUpdateConfig  # noqa: E402

logger = logging.getLogger("nervos_brain.telegram_bot")


class _FixedModelRegistry:
    """Pin all graph tasks to one model name (from config or CLI)."""

    def __init__(self, model: str) -> None:
        self._model = model

    def get_model_for(
        self,
        task_type: str,
        *,
        require_json: bool = False,
        max_cost: str = "high",
    ) -> str:
        _ = task_type, require_json, max_cost
        return self._model

    def get_profile_for(
        self,
        task_type: str,
        *,
        tier: str | None = None,
        require_json: bool = False,
        max_cost: str = "high",
    ) -> dict[str, Any]:
        _ = task_type, tier, require_json, max_cost
        return {
            "tier": "override",
            "model": self._model,
            "reasoning_effort": "",
            "verbosity": "",
            "max_tokens": 0,
        }


def _load_project_cfg() -> dict[str, Any]:
    try:
        return load_project_config()
    except Exception:
        logger.exception("Failed to load project config")
        return {}


def _load_telegram_bot_cfg(project_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = project_cfg if project_cfg is not None else _load_project_cfg()
    section = cfg.get("telegram_bot", {}) if isinstance(cfg, dict) else {}
    return dict(section) if isinstance(section, dict) else {}


def _load_progress_update_cfg(project_cfg: dict[str, Any]) -> ProgressUpdateConfig:
    section = project_cfg.get("progress_updates", {})
    return ProgressUpdateConfig.from_mapping(section if isinstance(section, dict) else {})


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cfg_str_list(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key, [])
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _build_runtime(*, model: str, memory_db: Path) -> tuple[FullGraphRuntime, MemoryService]:
    retriever = build_configured_retriever()
    backend_names = getattr(retriever, "backend_names", None)
    if backend_names:
        logger.info("Retrieval backends loaded: %s", backend_names)
    else:
        cfg = getattr(retriever, "_cfg", None)
        logger.info(
            "Retrieval backend loaded: %s",
            getattr(cfg, "collection_name", "retrieval"),
        )

    memory_db.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite+pysqlite:///{memory_db}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    init_memory_schema(engine)
    memory_service = MemoryService(build_session_factory(engine))

    provider_registry = _FixedModelRegistry(model) if model else ProviderCapabilityRegistry()
    if model:
        logger.info("LLM model override enabled for all graph nodes: %s", model)
    else:
        profiles = getattr(provider_registry, "list_profiles", lambda: {})()
        logger.info("Dynamic LLM profiles loaded: %s", profiles)

    runtime = FullGraphRuntime(
        multi_retriever=retriever,
        memory_service=memory_service,
        provider_registry=provider_registry,
        provider_max_cost="high",
    )
    return runtime, memory_service


def _print_update_result(row: dict) -> None:
    uid = row.get("update_id")
    chat_id = row.get("chat_id")
    if row.get("ignored"):
        reason = row.get("reason", "ignored")
        logger.info("[skip] update_id=%s chat_id=%s reason=%s", uid, chat_id, reason)
        return
    rid = row.get("request_id", "unknown")
    sent = int(row.get("sent_count", 0))
    logger.info("[ok] update_id=%s chat_id=%s request_id=%s sent=%d", uid, chat_id, rid, sent)


def _str_list_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _parse_id_set(values: list[str]) -> set[str]:
    allowed: set[str] = set()
    for raw in values:
        for part in raw.split(","):
            value = part.strip()
            if value:
                allowed.add(value)
    return allowed


def _parse_allowed_chat_ids(values: list[str]) -> set[str]:
    return _parse_id_set(values)


def _parse_allowed_thread_ids(values: list[str]) -> set[str]:
    return _parse_id_set(values)


def _parse_allowed_thread_ids_by_chat(value: Any) -> dict[str, set[str]]:
    if not isinstance(value, dict):
        return {}
    allowed: dict[str, set[str]] = {}
    for raw_chat_id, raw_thread_ids in value.items():
        chat_id = str(raw_chat_id).strip()
        if not chat_id:
            continue
        thread_ids = _parse_allowed_thread_ids(_str_list_from_value(raw_thread_ids))
        if thread_ids:
            allowed[chat_id] = thread_ids
    return allowed


def main() -> int:
    project_cfg = _load_project_cfg()
    bot_cfg = _load_telegram_bot_cfg(project_cfg)
    progress_update_cfg = _load_progress_update_cfg(project_cfg)

    parser = argparse.ArgumentParser(description="Run Telegram Bot polling gateway.")
    parser.add_argument(
        "--bot-token",
        default="",
        help=(
            "Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN env, "
            "then telegram_bot.bot_token in config.yaml."
        ),
    )
    parser.add_argument(
        "--api-base",
        default=str(bot_cfg.get("api_base", "")),
        help="Telegram Bot API base URL. Defaults to TELEGRAM_BOT_API_BASE env or official endpoint.",
    )
    parser.add_argument(
        "--offset-file",
        default=str(bot_cfg.get("offset_file", "data/telegram_bot/offset.txt")),
        help="Persisted getUpdates offset path.",
    )
    parser.add_argument(
        "--memory-db",
        default=str(bot_cfg.get("memory_db", "data/telegram_bot/memory.db")),
        help="SQLite path for memory service.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Debug override for all full-graph tasks. Omit to use dynamic node-level router.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=_cfg_int(bot_cfg, "poll_timeout", 25),
        help="getUpdates timeout seconds.",
    )
    parser.add_argument(
        "--poll-limit",
        type=int,
        default=_cfg_int(bot_cfg, "poll_limit", 20),
        help="getUpdates batch size (1-100).",
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=_cfg_float(bot_cfg, "idle_sleep", 0.8),
        help="Sleep seconds after empty poll or error.",
    )
    parser.add_argument(
        "--allowed-chat-id",
        action="append",
        default=_cfg_str_list(bot_cfg, "allowed_chat_ids"),
        help="Allow only these chat IDs (repeatable, comma supported).",
    )
    parser.add_argument(
        "--allowed-thread-id",
        action="append",
        default=_cfg_str_list(bot_cfg, "allowed_thread_ids"),
        help="Allow only these Telegram topic/thread IDs inside allowed chats (repeatable, comma supported).",
    )
    parser.add_argument(
        "--render-mode",
        choices=["markdown", "plain"],
        default="markdown",
        help="Outbound render mode.",
    )
    parser.add_argument(
        "--append-csat",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "append_csat", False),
        help="Whether to attach CSAT inline keyboard to outbound messages.",
    )
    parser.add_argument(
        "--feedback-file",
        default=str(bot_cfg.get("feedback_file", "data/telegram_bot/feedback.jsonl")),
        help="JSONL path for answer metadata, CSAT, comments, and BadCase records.",
    )
    parser.add_argument(
        "--debug-log-file",
        default=str(bot_cfg.get("debug_log_file", "data/telegram_bot/debug_events.jsonl")),
        help="JSONL path for per-update Telegram debug events.",
    )
    parser.add_argument(
        "--fast-mode-state-file",
        default=str(bot_cfg.get("fast_mode_state_file", "data/runtime/fast_mode_state.json")),
        help="JSON path for one-shot per-user /fast state.",
    )
    parser.add_argument(
        "--memory-context-limit",
        type=int,
        default=_cfg_int(bot_cfg, "memory_context_limit", 20),
        help="Recent same-user same-chat message events injected into graph state.",
    )
    parser.add_argument(
        "--max-worker-threads",
        type=int,
        default=_cfg_int(bot_cfg, "max_worker_threads", 4),
        help="Concurrent Telegram update worker threads; same chat/user updates stay ordered.",
    )
    parser.add_argument(
        "--target-elapsed-ms",
        type=int,
        default=_cfg_int(bot_cfg, "target_elapsed_ms", 30000),
        help="Soft target latency budget passed to graph nodes for LLM planning.",
    )
    parser.add_argument(
        "--max-elapsed-ms",
        type=int,
        default=_cfg_int(bot_cfg, "max_elapsed_ms", 90000),
        help="Soft maximum latency budget passed to graph nodes for LLM planning.",
    )
    parser.add_argument(
        "--mention-only-in-group",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "mention_only_in_group", True),
        help="In Telegram groups, only respond to @bot, bot replies, or bot commands.",
    )
    parser.add_argument(
        "--respond-to-bot-replies",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "respond_to_bot_replies", True),
        help="In Telegram groups, respond when a user replies to a bot message.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline but do not send Telegram messages.",
    )
    parser.add_argument(
        "--drop-pending-on-start",
        action="store_true",
        help="Skip pending backlog updates when process starts.",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="For debug: stop after N polls (0 = infinite).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug log level.",
    )
    args = parser.parse_args()

    setup_info = setup_logging(
        service_name="telegram_bot_polling",
        debug=bool(args.debug),
    )

    cfg_token = str(bot_cfg.get("bot_token", "") or "").strip()
    env_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    env_api_base = (os.getenv("TELEGRAM_BOT_API_BASE") or "https://api.telegram.org").strip()
    bot_token = args.bot_token.strip() or env_token or cfg_token
    if not bot_token:
        raise TelegramBotRuntimeError(
            "Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN, pass --bot-token, "
            "or set telegram_bot.bot_token in config.yaml."
        )
    api_base = (args.api_base.strip() or env_api_base or "https://api.telegram.org").rstrip("/")
    cfg = TelegramBotConfig(bot_token=bot_token, api_base=api_base)

    model = args.model.strip()
    runtime, memory_service = _build_runtime(
        model=model,
        memory_db=resolve_project_path(args.memory_db),
    )
    graph = build_full_graph()

    api = TelegramBotAPI(cfg)
    me = api.get_me()
    username = me.get("username")
    logger.info(
        "Telegram bot online: id=%s username=@%s model_mode=%s dry_run=%s log_file=%s",
        me.get("id"),
        username or "unknown",
        model or "dynamic-router",
        args.dry_run,
        setup_info.get("log_file", ""),
    )

    offset_store = TelegramUpdateOffsetStore(resolve_project_path(args.offset_file))
    allowed_chat_ids = _parse_allowed_chat_ids(args.allowed_chat_id)
    allowed_thread_ids = _parse_allowed_thread_ids(args.allowed_thread_id)
    allowed_thread_ids_by_chat = _parse_allowed_thread_ids_by_chat(
        bot_cfg.get("allowed_thread_ids_by_chat", {})
    )
    feedback_file = resolve_project_path(args.feedback_file)
    debug_log_file = resolve_project_path(args.debug_log_file) if str(args.debug_log_file).strip() else None
    fast_mode_state_file = resolve_project_path(args.fast_mode_state_file)
    gateway = TelegramPollingGateway(
        api=api,
        graph_runner=lambda state: invoke_full_graph(
            state,
            runtime=runtime,
            compiled_graph=graph,
        ),
        offset_store=offset_store,
        render_mode=args.render_mode,
        append_csat=args.append_csat,
        allowed_chat_ids=allowed_chat_ids,
        allowed_thread_ids=allowed_thread_ids,
        allowed_thread_ids_by_chat=allowed_thread_ids_by_chat,
        feedback_store=FeedbackJsonlStore(feedback_file),
        debug_log_file=debug_log_file,
        memory_service=memory_service,
        memory_context_limit=max(1, int(args.memory_context_limit)),
        mention_only_in_group=bool(args.mention_only_in_group),
        respond_to_bot_replies=bool(args.respond_to_bot_replies),
        bot_user_id=str(me.get("id", "") or ""),
        bot_username=str(username or ""),
        target_elapsed_ms=max(0, int(args.target_elapsed_ms)),
        max_elapsed_ms=max(0, int(args.max_elapsed_ms)),
        progress_update_config=progress_update_cfg,
        fast_mode_store=FastModeStateStore(fast_mode_state_file),
        max_worker_threads=max(1, int(args.max_worker_threads)),
    )
    logger.info(
        "Telegram beta controls: allowed_chat_ids=%s allowed_thread_ids=%s allowed_thread_ids_by_chat=%s append_csat=%s mention_only_in_group=%s respond_to_bot_replies=%s feedback_file=%s debug_log_file=%s",
        sorted(allowed_chat_ids) if allowed_chat_ids else "ALL",
        sorted(allowed_thread_ids) if allowed_thread_ids else "ALL",
        {
            chat_id: sorted(thread_ids)
            for chat_id, thread_ids in sorted(allowed_thread_ids_by_chat.items())
        }
        or "ALL",
        bool(args.append_csat),
        bool(args.mention_only_in_group),
        bool(args.respond_to_bot_replies),
        feedback_file,
        debug_log_file or "disabled",
    )

    if args.drop_pending_on_start:
        updates = api.get_updates(
            offset=gateway.next_offset,
            timeout_s=0,
            limit=100,
        )
        if updates:
            ids = [
                int(row["update_id"])
                for row in updates
                if isinstance(row.get("update_id"), int)
            ]
            if ids:
                latest = max(ids)
                gateway.set_next_offset(latest + 1)
                logger.info("Dropped pending updates. next_offset=%s", gateway.next_offset)
            else:
                logger.warning("Pending updates contained no valid update_id; offset unchanged.")
        else:
            logger.info("No pending updates to drop.")

    poll_count = 0
    while True:
        try:
            rows = gateway.poll_once(
                timeout_s=max(0, args.poll_timeout),
                limit=max(1, min(args.poll_limit, 100)),
                dry_run=bool(args.dry_run),
            )
            if rows:
                for row in rows:
                    _print_update_result(row)
            else:
                time.sleep(max(0.1, args.idle_sleep))
        except KeyboardInterrupt:
            logger.info("Interrupted. exit.")
            return 0
        except TelegramBotRuntimeError as exc:
            logger.warning("[error] %s", exc)
            time.sleep(max(0.1, args.idle_sleep))

        poll_count += 1
        if args.max_polls > 0 and poll_count >= args.max_polls:
            logger.info("Reached max polls: %d. exit.", args.max_polls)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
