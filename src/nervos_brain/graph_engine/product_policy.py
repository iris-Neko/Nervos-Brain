"""Centralized product, locale, model-profile and message policy.

Natural-language decisions are made by the Turn Interpreter.  This module only
holds product configuration and deterministic validation/defaulting so that
Telegram, Discord, and Graph nodes use one policy source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nervos_brain.pathing import load_project_config


_DEFAULT_MESSAGES: dict[str, dict[str, Any]] = {
    "en": {
        "insufficient_evidence": "I could not find enough reliable evidence to answer this accurately yet.",
        "generation_failed": "I could not prepare the answer this time. Please try again.",
        "direct_generation_failed": "I could not prepare a direct answer this time. Please try again.",
        "clarify_conflict": "Which of these conflicting options should I use for your request?",
        "clarify_uncertainty": "What specific detail should I use to narrow this request?",
        "policy_refusal": (
            "I cannot provide specific price predictions or buy/sell decisions. "
            "I can help with neutral facts, mechanisms, risks, or public data sources."
        ),
        "references_heading": "References",
        "source_label": "Source",
        "progress_messages": [
            "I am still checking sources and organizing the evidence. Please wait a moment.",
            "I am verifying the relevant material and preparing the answer.",
            "I am still preparing the final answer. Please wait a little longer.",
        ],
    },
    "zh-CN": {
        "insufficient_evidence": "当前没有检索到足够可靠的证据，暂时无法准确回答。",
        "generation_failed": "这次暂时没能生成回答，请稍后重试。",
        "direct_generation_failed": "这次暂时没能生成直接回答，请稍后重试。",
        "clarify_conflict": "这些资料存在冲突。你希望我按哪一个选项继续？",
        "clarify_uncertainty": "你希望我根据哪个具体条件来缩小这个问题？",
        "policy_refusal": "我不能提供具体价格预测或买卖决策，但可以解释中立事实、机制、风险或公开数据来源。",
        "references_heading": "参考来源",
        "source_label": "来源",
        "progress_messages": [
            "我还在查资料和整理证据，请稍等。",
            "我正在核对相关资料并组织回答。",
            "还在生成最终回答，请再等一下。",
        ],
    },
}


@dataclass(frozen=True)
class IdentityPolicy:
    name: str = "Nervos Brain"
    aliases: tuple[str, ...] = ("NB",)
    description: str = "An assistant for information, questions, and task completion."


@dataclass(frozen=True)
class LanguagePolicy:
    supported_locales: tuple[str, ...] = ("en", "zh-CN")
    default_locale: str = "en"
    ambiguous_locale: str = "en"
    use_confirmed_user_preference: bool = True
    unsupported_locale_behavior: str = "default_with_notice"


@dataclass(frozen=True)
class ContextPolicy:
    recent_message_limit: int = 20
    max_context_chars: int = 1800


@dataclass(frozen=True)
class ModelProfilePolicy:
    turn_interpreter: str = "low"
    turn_interpreter_fallback: str = "medium"
    response_compliance: str = "medium"


@dataclass(frozen=True)
class CompliancePolicy:
    max_replacements: int = 1


@dataclass(frozen=True)
class ProductPolicy:
    identity: IdentityPolicy = field(default_factory=IdentityPolicy)
    language: LanguagePolicy = field(default_factory=LanguagePolicy)
    context: ContextPolicy = field(default_factory=ContextPolicy)
    model_profiles: ModelProfilePolicy = field(default_factory=ModelProfilePolicy)
    compliance: CompliancePolicy = field(default_factory=CompliancePolicy)
    messages: dict[str, dict[str, Any]] = field(default_factory=lambda: _copy_messages(_DEFAULT_MESSAGES))

    def resolve_locale(self, locale: str | None) -> str:
        candidate = str(locale or "").strip().replace("_", "-")
        if candidate in self.language.supported_locales:
            return candidate
        prefix = candidate.split("-", 1)[0]
        for supported in self.language.supported_locales:
            if supported.split("-", 1)[0] == prefix and prefix:
                return supported
        return self.language.default_locale

    def message(self, key: str, locale: str | None = None, default: str = "") -> str:
        selected = self.resolve_locale(locale)
        localized = self.messages.get(selected, {})
        value = localized.get(key)
        if isinstance(value, str) and value.strip():
            return value
        english = self.messages.get("en", {})
        value = english.get(key, default)
        return str(value) if value is not None else default

    def progress_messages(self, locale: str | None = None) -> list[str]:
        selected = self.resolve_locale(locale)
        value = self.messages.get(selected, {}).get("progress_messages", [])
        if not isinstance(value, list) or not all(str(item).strip() for item in value):
            value = self.messages.get("en", {}).get("progress_messages", [])
        return [str(item) for item in value]

    def render_prompt_block(self) -> str:
        aliases = ", ".join(self.identity.aliases) or "(none)"
        locales = ", ".join(self.language.supported_locales)
        return (
            "Configured product policy:\n"
            f"- Assistant name: {self.identity.name}\n"
            f"- Assistant aliases: {aliases}\n"
            f"- Assistant description: {self.identity.description}\n"
            f"- Supported output locales: {locales}\n"
            f"- Default locale for ambiguous input: {self.language.ambiguous_locale}\n"
            "- The configured product identity has priority over underlying implementation details."
        )


def _copy_messages(value: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for locale, messages in value.items():
        copied[locale] = {
            key: list(item) if isinstance(item, list) else item
            for key, item in messages.items()
        }
    return copied


def _section(raw: Any, key: str) -> dict[str, Any]:
    value = raw.get(key, {}) if isinstance(raw, dict) else {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    items = tuple(str(item).strip() for item in value if str(item).strip())
    return items or default


def _profile_name(value: Any, default: str, configured: set[str]) -> str:
    candidate = str(value or default).strip() or default
    if not configured or candidate in configured:
        return candidate
    if default in configured:
        return default
    return sorted(configured)[0]


def load_product_policy(raw_config: Mapping[str, Any] | None = None) -> ProductPolicy:
    """Load and validate one centralized product policy."""
    config = dict(raw_config) if isinstance(raw_config, Mapping) else load_project_config()
    raw = _section(config, "agent_policy")
    identity_raw = _section(raw, "identity")
    language_raw = _section(raw, "language")
    context_raw = _section(raw, "context")
    profiles_raw = _section(raw, "model_profiles")
    compliance_raw = _section(raw, "compliance")

    identity = IdentityPolicy(
        name=str(identity_raw.get("name", IdentityPolicy.name) or IdentityPolicy.name).strip(),
        aliases=_string_tuple(identity_raw.get("aliases"), IdentityPolicy.aliases),
        description=str(identity_raw.get("description", IdentityPolicy.description) or IdentityPolicy.description).strip(),
    )
    supported = _string_tuple(language_raw.get("supported_locales"), LanguagePolicy.supported_locales)
    default_locale = str(language_raw.get("default_locale", supported[0]) or supported[0]).strip()
    ambiguous_locale = str(language_raw.get("ambiguous_locale", default_locale) or default_locale).strip()
    if default_locale not in supported:
        default_locale = supported[0]
    if ambiguous_locale not in supported:
        ambiguous_locale = default_locale
    language = LanguagePolicy(
        supported_locales=supported,
        default_locale=default_locale,
        ambiguous_locale=ambiguous_locale,
        use_confirmed_user_preference=bool(language_raw.get("use_confirmed_user_preference", True)),
        unsupported_locale_behavior=str(
            language_raw.get("unsupported_locale_behavior", "default_with_notice")
            or "default_with_notice"
        ),
    )
    context = ContextPolicy(
        recent_message_limit=_positive_int(context_raw.get("recent_message_limit"), 20),
        max_context_chars=_positive_int(context_raw.get("max_context_chars"), 1800),
    )
    configured_profiles = set(_section(config, "llm_profiles"))
    model_profiles = ModelProfilePolicy(
        turn_interpreter=_profile_name(profiles_raw.get("turn_interpreter"), "low", configured_profiles),
        turn_interpreter_fallback=_profile_name(
            profiles_raw.get("turn_interpreter_fallback"), "medium", configured_profiles
        ),
        response_compliance=_profile_name(
            profiles_raw.get("response_compliance"), "medium", configured_profiles
        ),
    )
    compliance = CompliancePolicy(
        max_replacements=_positive_int(compliance_raw.get("max_replacements"), 1),
    )

    message_raw = _section(config, "message_catalog")
    messages = _copy_messages(_DEFAULT_MESSAGES)
    for locale, values in message_raw.items():
        if not isinstance(values, dict):
            continue
        messages.setdefault(str(locale), {}).update(values)
    return ProductPolicy(
        identity=identity,
        language=language,
        context=context,
        model_profiles=model_profiles,
        compliance=compliance,
        messages=messages,
    )


def policy_from_state(state: Mapping[str, Any] | None = None) -> ProductPolicy:
    """Return an injected policy or the centralized project policy."""
    if isinstance(state, Mapping):
        value = state.get("_product_policy")
        if isinstance(value, ProductPolicy):
            return value
        if isinstance(value, Mapping):
            return load_product_policy(value)
    return load_product_policy()
