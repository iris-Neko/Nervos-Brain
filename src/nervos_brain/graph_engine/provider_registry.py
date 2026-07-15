"""M7-T9: ProviderCapabilityRegistry — 模型能力注册与选择。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Literal

from nervos_brain.pathing import load_project_config


TaskType = Literal[
    "planning",
    "grading",
    "reflection",
    "composing",
    "self_check",
    "general",
]
ProfileTier = Literal["router", "low", "medium", "high"]
ModelCostTier = Literal["low", "medium", "high"]


@dataclass
class ModelCapability:
    provider: str
    model: str
    supports_native_tool_call: bool = False
    supports_json_schema: bool = False
    max_context_tokens: int = 128_000
    cost_tier: ModelCostTier = "low"
    preferred_tasks: list[TaskType] = field(default_factory=lambda: ["general"])


@dataclass(frozen=True)
class ModelProfile:
    tier: ProfileTier
    model: str
    reasoning_effort: str = "low"
    verbosity: str = "low"
    max_tokens: int = 2048


_DEFAULT_REGISTRY: list[ModelCapability] = [
    ModelCapability(
        provider="openai",
        model="gpt-4o-mini",
        supports_native_tool_call=True,
        supports_json_schema=True,
        max_context_tokens=128_000,
        cost_tier="low",
        preferred_tasks=["planning", "grading", "reflection", "self_check", "general"],
    ),
    ModelCapability(
        provider="openai",
        model="gpt-4o",
        supports_native_tool_call=True,
        supports_json_schema=True,
        max_context_tokens=128_000,
        cost_tier="high",
        preferred_tasks=["composing"],
    ),
    ModelCapability(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        supports_native_tool_call=True,
        supports_json_schema=True,
        max_context_tokens=200_000,
        cost_tier="medium",
        preferred_tasks=["composing", "reflection", "self_check"],
    ),
]


_DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "router": ModelProfile(
        tier="router",
        model="openai/gpt-5.6-luna",
        reasoning_effort="low",
        verbosity="low",
        max_tokens=512,
    ),
    "low": ModelProfile(
        tier="low",
        model="openai/gpt-5.6-luna",
        reasoning_effort="low",
        verbosity="low",
        max_tokens=2048,
    ),
    "medium": ModelProfile(
        tier="medium",
        model="openai/gpt-5.6-luna",
        reasoning_effort="medium",
        verbosity="low",
        max_tokens=2048,
    ),
    "high": ModelProfile(
        tier="high",
        model="openai/gpt-5.6-luna",
        reasoning_effort="high",
        verbosity="low",
        max_tokens=4096,
    ),
}


def _load_profiles_from_config() -> dict[str, ModelProfile]:
    raw = load_project_config().get("llm_profiles", {})
    if not isinstance(raw, dict):
        return dict(_DEFAULT_PROFILES)

    profiles = dict(_DEFAULT_PROFILES)
    for tier in ("router", "low", "medium", "high"):
        section = raw.get(tier, {})
        if not isinstance(section, dict):
            continue
        base = profiles[tier]
        try:
            max_tokens = int(section.get("max_tokens", base.max_tokens))
        except (TypeError, ValueError):
            max_tokens = base.max_tokens
        profiles[tier] = ModelProfile(
            tier=tier,
            model=str(section.get("model", base.model) or base.model),
            reasoning_effort=str(
                section.get("reasoning_effort", base.reasoning_effort)
                or base.reasoning_effort
            ),
            verbosity=str(section.get("verbosity", base.verbosity) or base.verbosity),
            max_tokens=max_tokens,
        )
    return profiles


class ProviderCapabilityRegistry:
    """按能力匹配选模型。"""

    def __init__(
        self,
        models: list[ModelCapability] | None = None,
        profiles: dict[str, ModelProfile] | None = None,
    ) -> None:
        self._models = models or list(_DEFAULT_REGISTRY)
        if profiles is None:
            self._profiles = _load_profiles_from_config()
        else:
            merged_profiles = dict(_DEFAULT_PROFILES)
            merged_profiles.update(profiles)
            self._profiles = merged_profiles

    def get_model_for(
        self,
        task_type: TaskType,
        *,
        require_json: bool = False,
        max_cost: ModelCostTier = "high",
    ) -> str:
        """按能力匹配选模型，返回 model 名。"""
        cost_order = {"low": 0, "medium": 1, "high": 2}
        max_cost_val = cost_order[max_cost]

        candidates = [
            m for m in self._models
            if cost_order[m.cost_tier] <= max_cost_val
            and (not require_json or m.supports_json_schema)
        ]

        preferred = [m for m in candidates if task_type in m.preferred_tasks]
        if preferred:
            preferred.sort(key=lambda m: cost_order[m.cost_tier])
            return preferred[0].model

        if candidates:
            candidates.sort(key=lambda m: cost_order[m.cost_tier])
            return candidates[0].model

        return _DEFAULT_REGISTRY[0].model

    def get_profile_for(
        self,
        task_type: TaskType,
        *,
        tier: ProfileTier | str | None = None,
        require_json: bool = False,
        max_cost: ModelCostTier = "high",
    ) -> dict[str, Any]:
        """Return a concrete model profile for a graph node call."""
        _ = task_type, require_json, max_cost
        selected = str(tier or "low").strip().lower()
        if selected not in self._profiles:
            selected = "low"
        profile = self._profiles[selected]
        return {
            "tier": profile.tier,
            "model": profile.model,
            "reasoning_effort": profile.reasoning_effort,
            "verbosity": profile.verbosity,
            "max_tokens": profile.max_tokens,
        }

    def list_profiles(self) -> dict[str, dict[str, Any]]:
        return {
            tier: {
                "tier": profile.tier,
                "model": profile.model,
                "reasoning_effort": profile.reasoning_effort,
                "verbosity": profile.verbosity,
                "max_tokens": profile.max_tokens,
            }
            for tier, profile in self._profiles.items()
        }

    def register(self, cap: ModelCapability) -> None:
        self._models.append(cap)

    def list_models(self) -> list[ModelCapability]:
        return list(self._models)
