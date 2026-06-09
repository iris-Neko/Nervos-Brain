"""M6-T1/T2/T3: 工具白名单、参数 schema、校验函数。"""

from __future__ import annotations

from typing import Any

TOOL_WHITELIST: set[str] = {
    "qdrant_search",
    "discourse_query",
    "github_search",
    "memory_fetch",
}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "qdrant_search": {
        "type": "object",
        "required": ["query", "filters"],
        "properties": {
            "query": {"type": "string", "maxLength": 512},
            "filters": {"type": "object"},
            "regex_queries": {"type": "array"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "additionalProperties": False,
    },
    "discourse_query": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "maxLength": 512},
            "category": {"type": "string"},
            "time_range": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "additionalProperties": False,
    },
    "github_search": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "maxLength": 512},
            "repo": {"type": "string"},
            "path": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "additionalProperties": False,
    },
    "memory_fetch": {
        "type": "object",
        "required": ["namespace", "platform"],
        "properties": {
            "namespace": {"type": "string", "enum": ["user", "channel"]},
            "platform": {"type": "string"},
            "user_id": {"type": "string"},
            "guild_id": {"type": "string"},
            "channel_id": {"type": "string"},
        },
        "additionalProperties": False,
    },
}


def validate_tool_args(tool: str, args: dict[str, Any]) -> list[str]:
    """校验工具名 + 参数。返回错误列表，空列表表示通过。"""
    errors: list[str] = []

    if tool not in TOOL_WHITELIST:
        errors.append(f"tool '{tool}' not in whitelist {TOOL_WHITELIST}")
        return errors

    schema = TOOL_SCHEMAS.get(tool)
    if schema is None:
        return errors

    required = schema.get("required", [])
    for key in required:
        if key not in args:
            errors.append(f"missing required arg '{key}' for tool '{tool}'")

    props = schema.get("properties", {})
    for key, value in args.items():
        # 运行时依赖通过私有参数注入，不参与用户侧 schema 校验。
        if key.startswith("_"):
            continue
        if key not in props:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected arg '{key}' for tool '{tool}'")
            continue

        prop_schema = props[key]
        expected_type = prop_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"arg '{key}' must be string")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"arg '{key}' must be integer")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"arg '{key}' must be object")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"arg '{key}' must be array")

        if expected_type == "string":
            max_len = prop_schema.get("maxLength")
            if max_len and isinstance(value, str) and len(value) > max_len:
                errors.append(f"arg '{key}' exceeds maxLength {max_len}")

        if expected_type == "integer" and isinstance(value, int):
            minimum = prop_schema.get("minimum")
            maximum = prop_schema.get("maximum")
            if minimum is not None and value < minimum:
                errors.append(f"arg '{key}' below minimum {minimum}")
            if maximum is not None and value > maximum:
                errors.append(f"arg '{key}' above maximum {maximum}")

        allowed = prop_schema.get("enum")
        if allowed is not None and value not in allowed:
            errors.append(f"arg '{key}' must be one of {allowed}")

    return errors
