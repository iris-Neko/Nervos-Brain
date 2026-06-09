from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from nervos_brain.graph_engine import llm


class _CompletionResponse:
    choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]
    usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


class _ResponsesResponse:
    output_text = "ok"
    usage = {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}


def test_call_llm_passes_service_tier_to_completion(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _CompletionResponse:
        captured.update(kwargs)
        return _CompletionResponse()

    fake_litellm = SimpleNamespace(
        completion=fake_completion,
        responses=lambda **_kwargs: _ResponsesResponse(),
        suppress_debug_info=False,
        set_verbose=True,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setattr(llm, "_config_cache", {"api_key": "", "api_base": "", "max_retries": 1})

    text = llm.call_llm("system", "user", model="openai/gpt-4.1-mini", service_tier="priority")

    assert text == "ok"
    assert captured["service_tier"] == "priority"
    assert llm.get_last_call_meta()["service_tier"] == "priority"


def test_call_llm_passes_service_tier_to_gpt5_responses(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_responses(**kwargs: Any) -> _ResponsesResponse:
        captured.update(kwargs)
        return _ResponsesResponse()

    fake_litellm = SimpleNamespace(
        completion=lambda **_kwargs: _CompletionResponse(),
        responses=fake_responses,
        suppress_debug_info=False,
        set_verbose=True,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setattr(llm, "_config_cache", {"api_key": "", "api_base": "", "max_retries": 1})

    text = llm.call_llm("system", "user", model="openai/gpt-5.4", service_tier="priority")

    assert text == "ok"
    assert captured["service_tier"] == "priority"
    assert llm.get_last_call_meta()["service_tier"] == "priority"


def test_call_llm_omits_empty_service_tier(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _CompletionResponse:
        captured.update(kwargs)
        return _CompletionResponse()

    fake_litellm = SimpleNamespace(
        completion=fake_completion,
        responses=lambda **_kwargs: _ResponsesResponse(),
        suppress_debug_info=False,
        set_verbose=True,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setattr(llm, "_config_cache", {"api_key": "", "api_base": "", "service_tier": "", "max_retries": 1})
    monkeypatch.delenv("LLM_SERVICE_TIER", raising=False)

    llm.call_llm("system", "user", model="openai/gpt-4.1-mini")

    assert "service_tier" not in captured
    assert llm.get_last_call_meta()["service_tier"] == ""
