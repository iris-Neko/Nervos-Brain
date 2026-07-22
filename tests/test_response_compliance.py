from unittest.mock import patch

from nervos_brain.graph_engine.full_nodes import response_compliance


def _state(**overrides):
    state = {
        "request_id": "compliance-test",
        "locale": "en",
        "response_locale": "en",
        "turn_contract": {
            "route": "direct",
            "policy": {"action": "allow"},
            "core_deliverable": "Answer the current request.",
        },
        "user_message": {"content": "Answer the current request."},
        "_final_response": {
            "request_id": "compliance-test",
            "text": "Draft answer.",
            "citations": [],
        },
        "evidence": [],
    }
    state.update(overrides)
    return state


def test_typed_policy_refusal_bypasses_response_rewriter():
    state = _state(
        turn_contract={"route": "policy_response", "policy": {"action": "refuse"}},
        _final_response={"request_id": "compliance-test", "text": "unsafe draft", "citations": []},
    )

    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json") as mock_call:
        result = response_compliance(state)

    mock_call.assert_not_called()
    assert result["_compliance_decision"] == "accept"
    assert result["_final_response"]["answer_mode"] == "policy_refusal"


def test_response_compliance_compiles_citations_from_a_bounded_replacement():
    evidence = [
        {
            "id": "ev-1",
            "source": "docs",
            "title": "Current guide",
            "url": "https://example.com/guide",
            "anchor": "section-1",
            "snippet": "The requested operation is supported.",
            "score": 0.9,
        }
    ]
    state = _state(evidence=evidence)

    def mock_json(system_prompt, user_prompt, **kwargs):
        _ = user_prompt, kwargs
        if "Prompt ID: model_router" in system_prompt:
            return {"tier": "low"}
        return {
            "decision": "replace",
            "issue_codes": ["task_incomplete"],
            "reasoning": "The result must be stated first.",
            "replacement_answer": "The operation is supported. {{cite:E1}}",
        }

    with patch("nervos_brain.graph_engine.full_nodes.call_llm_json", side_effect=mock_json):
        result = response_compliance(state)

    assert result["_compliance_decision"] == "replace"
    assert result["_final_response"]["text"] == "The operation is supported. [1]"
    assert result["_final_response"]["citations"][0]["label"] == "[1]"
