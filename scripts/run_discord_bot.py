#!/usr/bin/env python3
"""Run Discord bot gateway."""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
from nervos_brain.logging_system import setup_logging  # noqa: E402
from nervos_brain.memory import (  # noqa: E402
    MemoryService,
    build_session_factory,
    init_memory_schema,
)
from nervos_brain.pathing import load_project_config, resolve_project_path  # noqa: E402
from nervos_brain.retrieval import build_configured_retriever  # noqa: E402
from nervos_brain.tool_runtime.discord_bot_runtime import (  # noqa: E402
    DiscordBotConfig,
    DiscordBotRuntime,
    DiscordBotRuntimeError,
    DiscordGateway,
)
from nervos_brain.tool_runtime.fast_mode import FastModeStateStore  # noqa: E402
from nervos_brain.tool_runtime.feedback import FeedbackJsonlStore  # noqa: E402
from nervos_brain.tool_runtime.progress import ProgressUpdateConfig  # noqa: E402

logger = logging.getLogger("nervos_brain.discord_bot")


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


def _load_discord_bot_cfg(project_cfg: dict[str, Any]) -> dict[str, Any]:
    section = project_cfg.get("discord_bot", {})
    if isinstance(section, dict):
        return section
    return {}


def _load_progress_update_cfg(project_cfg: dict[str, Any]) -> ProgressUpdateConfig:
    section = project_cfg.get("progress_updates", {})
    return ProgressUpdateConfig.from_mapping(section if isinstance(section, dict) else {})


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


def _parse_id_set(values: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in values:
        for part in raw.split(","):
            value = part.strip()
            if value:
                out.add(value)
    return out


def _cfg_str_list(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key, [])
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    project_cfg = _load_project_cfg()
    bot_cfg = _load_discord_bot_cfg(project_cfg)
    progress_update_cfg = _load_progress_update_cfg(project_cfg)
    default_model = ""

    parser = argparse.ArgumentParser(description="Run Discord Bot gateway.")
    parser.add_argument(
        "--bot-token",
        default="",
        help="Discord bot token. Defaults to DISCORD_BOT_TOKEN env.",
    )
    parser.add_argument(
        "--memory-db",
        default=str(bot_cfg.get("memory_db", "data/discord_bot/memory.db")),
        help="SQLite path for memory service.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Debug override for all full-graph tasks. Omit to use dynamic node-level router.",
    )
    parser.add_argument(
        "--allowed-channel-id",
        action="append",
        default=_cfg_str_list(bot_cfg, "allowed_channel_ids"),
        help="Allow only these channel IDs (repeatable, comma supported).",
    )
    parser.add_argument(
        "--allowed-guild-id",
        action="append",
        default=_cfg_str_list(bot_cfg, "allowed_guild_ids"),
        help="Allow only these guild IDs (repeatable, comma supported).",
    )
    parser.add_argument(
        "--render-mode",
        choices=["markdown", "plain"],
        default=str(bot_cfg.get("render_mode", "markdown") or "markdown"),
        help="Outbound render mode.",
    )
    parser.add_argument(
        "--append-csat",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "append_csat", False),
        help="Whether to attach CSAT buttons to the last outbound message segment.",
    )
    parser.add_argument(
        "--feedback-file",
        default=str(bot_cfg.get("feedback_file", "data/discord_bot/feedback.jsonl")),
        help="JSONL path for answer metadata, CSAT, comments, and BadCase records.",
    )
    parser.add_argument(
        "--debug-log-file",
        default=str(bot_cfg.get("debug_log_file", "data/discord_bot/debug_events.jsonl")),
        help="JSONL path for per-message Discord debug events.",
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
        help="Recent same-user same-channel message events injected into graph state.",
    )
    parser.add_argument(
        "--max-worker-threads",
        type=int,
        default=_cfg_int(bot_cfg, "max_worker_threads", 4),
        help="Concurrent Discord graph worker threads; same channel messages stay ordered.",
    )
    parser.add_argument(
        "--mention-only-in-guild",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "mention_only_in_guild", True),
        help="In guilds, only respond when bot is mentioned.",
    )
    parser.add_argument(
        "--respond-to-bot-replies",
        action=argparse.BooleanOptionalAction,
        default=_cfg_bool(bot_cfg, "respond_to_bot_replies", True),
        help="In guilds, respond when a user replies to a bot message.",
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
        "--attachment-max-bytes",
        type=int,
        default=_cfg_int(bot_cfg, "attachment_max_bytes", 262144),
        help="Maximum Discord text attachment bytes to download into graph context.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug log level.",
    )
    args = parser.parse_args()

    setup_info = setup_logging(
        service_name="discord_bot",
        debug=bool(args.debug),
    )

    token = args.bot_token.strip() or (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise DiscordBotRuntimeError(
            "Missing Discord bot token. Set DISCORD_BOT_TOKEN or pass --bot-token."
        )

    model = str(args.model).strip()
    logger.info(
        "Discord bot starting: model_mode=%s mention_only_in_guild=%s render_mode=%s log_file=%s",
        model or "dynamic-router",
        bool(args.mention_only_in_guild),
        args.render_mode,
        setup_info.get("log_file", ""),
    )
    runtime, memory_service = _build_runtime(
        model=model,
        memory_db=resolve_project_path(args.memory_db),
    )
    graph = build_full_graph()
    feedback_file = resolve_project_path(args.feedback_file)
    debug_log_file = resolve_project_path(args.debug_log_file) if str(args.debug_log_file).strip() else None
    fast_mode_state_file = resolve_project_path(args.fast_mode_state_file)
    allowed_channel_ids = _parse_id_set(args.allowed_channel_id)
    allowed_guild_ids = _parse_id_set(args.allowed_guild_id)

    gateway = DiscordGateway(
        graph_runner=lambda state: invoke_full_graph(
            state,
            runtime=runtime,
            compiled_graph=graph,
        ),
        allowed_channel_ids=allowed_channel_ids,
        allowed_guild_ids=allowed_guild_ids,
        render_mode=args.render_mode,
        append_csat=args.append_csat,
        mention_only_in_guild=bool(args.mention_only_in_guild),
        respond_to_bot_replies=bool(args.respond_to_bot_replies),
        feedback_store=FeedbackJsonlStore(feedback_file),
        debug_log_file=debug_log_file,
        memory_service=memory_service,
        memory_context_limit=max(1, int(args.memory_context_limit)),
        target_elapsed_ms=max(0, int(args.target_elapsed_ms)),
        max_elapsed_ms=max(0, int(args.max_elapsed_ms)),
        attachment_max_bytes=max(1024, int(args.attachment_max_bytes)),
        fast_mode_store=FastModeStateStore(fast_mode_state_file),
    )
    bot_runtime = DiscordBotRuntime(
        config=DiscordBotConfig(
            bot_token=token,
            mention_only_in_guild=bool(args.mention_only_in_guild),
        ),
        gateway=gateway,
        max_worker_threads=max(1, int(args.max_worker_threads)),
        progress_update_config=progress_update_cfg,
    )
    logger.info(
        "Discord beta controls: allowed_guild_ids=%s allowed_channel_ids=%s append_csat=%s mention_only_in_guild=%s respond_to_bot_replies=%s feedback_file=%s debug_log_file=%s",
        sorted(allowed_guild_ids) if allowed_guild_ids else "ALL",
        sorted(allowed_channel_ids) if allowed_channel_ids else "ALL",
        bool(args.append_csat),
        bool(args.mention_only_in_guild),
        bool(args.respond_to_bot_replies),
        feedback_file,
        debug_log_file or "disabled",
    )
    bot_runtime.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
