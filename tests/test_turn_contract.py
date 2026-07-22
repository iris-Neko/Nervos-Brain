from nervos_brain.graph_engine.product_policy import ProductPolicy
from nervos_brain.graph_engine.turn_contract import (
    derive_legacy_state,
    merge_contextual_contract,
    normalize_turn_contract,
)


def test_explicit_output_locale_wins_over_current_communication_locale():
    contract = normalize_turn_contract(
        {
            "core_deliverable": "Explain the API",
            "resolved_request": "Explain the API",
            "language": {
                "communication_locale": "en",
                "clarity": "clear",
                "requested_output_locale": "zh-CN",
            },
            "route": "direct",
        },
        current_message="Explain the API in Chinese.",
    )

    derived = derive_legacy_state(contract)

    assert derived["response_locale"] == "zh-CN"
    assert contract["language"]["requested_output_locale"] == "zh-CN"


def test_clear_current_locale_beats_confirmed_preference():
    contract = normalize_turn_contract(
        {
            "language": {"communication_locale": "en", "clarity": "clear"},
            "route": "direct",
        },
        confirmed_preference="zh-CN",
    )

    assert derive_legacy_state(contract, confirmed_preference="zh-CN")["response_locale"] == "en"


def test_ambiguous_language_uses_policy_default_without_text_heuristics():
    policy = ProductPolicy()
    contract = normalize_turn_contract(
        {"language": {"clarity": "ambiguous"}, "route": "direct"},
        policy,
        current_message="ok",
        confirmed_preference="zh-CN",
    )

    assert derive_legacy_state(contract, policy, confirmed_preference="zh-CN")["response_locale"] == "zh-CN"


def test_public_clarification_is_demoted_to_retrieval_or_direct_answer():
    contract = normalize_turn_contract(
        {
            "route": "clarify",
            "info_needs": [
                {
                    "kind": "public_facts",
                    "question": "Find the public operating conditions.",
                    "required": True,
                    "availability": "public",
                }
            ],
        }
    )

    assert contract["route"] == "retrieve"
    assert contract["retrieval_policy"] == "single"
    assert contract["info_needs"][0]["purpose"]


def test_user_owned_required_need_remains_a_clarification_route():
    contract = normalize_turn_contract(
        {
            "route": "clarify",
            "info_needs": [
                {
                    "kind": "missing_parameter",
                    "question": "Which private environment are you using?",
                    "required": True,
                    "availability": "user_owned",
                }
            ],
        }
    )

    assert contract["route"] == "clarify"
    assert contract["retrieval_policy"] == "none"


def test_context_selection_is_limited_to_available_message_ids():
    contract = normalize_turn_contract(
        {
            "context": {
                "requirement": "recent_history",
                "selected_message_ids": ["allowed", "not-available"],
            }
        },
        available_context_ids={"allowed"},
    )

    assert contract["context"]["selected_message_ids"] == ["allowed"]


def test_second_context_pass_can_refine_route_without_replacing_explicit_language():
    policy = ProductPolicy()
    first = normalize_turn_contract(
        {
            "resolved_request": "Compare the two interfaces.",
            "route": "retrieve",
            "retrieval_policy": "deep",
            "language": {
                "requested_output_locale": "en",
                "communication_locale": "en",
                "clarity": "clear",
            },
            "context": {
                "requirement": "recent_history",
                "selected_message_ids": [],
            },
        },
        policy,
    )
    merged = merge_contextual_contract(
        first,
        {
            "resolved_request": "Compare the two interfaces using the selected history.",
            "route": "direct",
            "retrieval_policy": "none",
            "language": {"communication_locale": "zh-CN", "clarity": "clear"},
            "context": {"requirement": "recent_history", "selected_message_ids": ["m1"]},
        },
        policy,
        available_context_ids={"m1"},
    )

    assert merged["resolved_request"] == "Compare the two interfaces using the selected history."
    assert merged["route"] == "direct"
    assert merged["retrieval_policy"] == "none"
    assert merged["language"]["requested_output_locale"] == "en"
    assert merged["context"]["selected_message_ids"] == ["m1"]


def test_legacy_state_is_derived_from_contract_without_reclassifying_the_request():
    contract = normalize_turn_contract(
        {
            "route": "policy_response",
            "retrieval_policy": "deep",
            "policy": {"action": "refuse", "categories": ["configured"], "reason": "policy"},
        }
    )
    derived = derive_legacy_state(contract)

    assert derived["_route_decision"] == "answer_direct"
    assert derived["retrieval_policy"] == "none"
    assert derived["_financial_guidance_refusal"] is True
