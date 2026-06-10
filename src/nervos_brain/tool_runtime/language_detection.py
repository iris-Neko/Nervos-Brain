"""Small deterministic language detector for inbound chat messages."""

from __future__ import annotations

import re


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+|<@!?\d+>")
_COMMAND_RE = re.compile(r"/[A-Za-z0-9_]+(?:@\w+)?")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_ENGLISH_INSTRUCTION_RE = re.compile(
    r"\b("
    r"answer|reply|respond|explain|say|use|speak|write"
    r")\b[^.?!\n\r]{0,80}\b("
    r"english|in english"
    r")\b",
    re.IGNORECASE,
)
_ENGLISH_PREFIX_RE = re.compile(r"\bplease\s+(answer|reply|respond)\s+in\s+english\b", re.IGNORECASE)
_CHINESE_INSTRUCTION_RE = re.compile(
    r"(用|使用|请用|请使用|请以|以|说|讲|解释|请解释|回答|回复)[^。！？\n\r]{0,30}"
    r"(中文|汉语|简体中文|繁体中文)",
)
_CHINESE_REQUEST_PREFIX_RE = re.compile(
    r"^\s*(请|帮我|麻烦|能不能|可以|解释|说明|讲讲|说说|回答|回复|介绍)"
)


def detect_message_locale(text: str, fallback_locale: str = "zh-CN") -> str:
    """Return ``zh-CN`` or ``en`` when message language is clear.

    Ambiguous short messages, URL-only messages, emoji-only messages, and
    balanced mixed-language text intentionally keep the platform fallback.
    """
    fallback = str(fallback_locale or "zh-CN").strip() or "zh-CN"
    normalized = _normalize_text(text)
    if not normalized:
        return fallback

    if _CHINESE_INSTRUCTION_RE.search(normalized):
        return "zh-CN"
    if _ENGLISH_PREFIX_RE.search(normalized) or _ENGLISH_INSTRUCTION_RE.search(normalized):
        return "en"
    if _CHINESE_REQUEST_PREFIX_RE.search(normalized):
        return "zh-CN"

    cjk_count = len(_CJK_RE.findall(normalized))
    latin_count = len(_LATIN_RE.findall(normalized))
    signal_count = cjk_count + latin_count
    if signal_count < 4:
        return fallback

    if cjk_count >= 2 and cjk_count >= latin_count:
        return "zh-CN"
    if latin_count >= 8 and cjk_count == 0:
        return "en"
    if latin_count >= 12 and latin_count >= cjk_count * 4:
        return "en"

    return fallback


def _normalize_text(text: str) -> str:
    value = str(text or "")
    value = _URL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _COMMAND_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
