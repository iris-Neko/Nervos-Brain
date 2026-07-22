from nervos_brain.graph_engine.product_policy import load_product_policy


def test_product_policy_centralizes_identity_and_default_language():
    policy = load_product_policy(
        {
            "agent_policy": {
                "identity": {
                    "name": "Example Brain",
                    "aliases": ["EB"],
                    "description": "A configurable assistant.",
                },
                "language": {
                    "supported_locales": ["en", "zh-CN"],
                    "default_locale": "en",
                    "ambiguous_locale": "en",
                },
            }
        }
    )

    block = policy.render_prompt_block()
    assert policy.identity.name == "Example Brain"
    assert policy.identity.aliases == ("EB",)
    assert policy.language.default_locale == "en"
    assert "Example Brain" in block
    assert "EB" in block


def test_product_policy_message_catalog_falls_back_to_english():
    policy = load_product_policy(
        {
            "message_catalog": {
                "en": {"generation_failed": "English fallback"},
            }
        }
    )

    assert policy.message("generation_failed", "en") == "English fallback"
    assert policy.message("generation_failed", "unsupported") == "English fallback"
    assert policy.message("unknown_key", "en", default="default") == "default"


def test_product_policy_normalizes_unknown_profile_references_against_configured_profiles():
    policy = load_product_policy(
        {
            "llm_profiles": {"low": {}, "medium": {}},
            "agent_policy": {
                "model_profiles": {
                    "turn_interpreter": "missing",
                    "turn_interpreter_fallback": "medium",
                    "response_compliance": "missing",
                }
            },
        }
    )

    assert policy.model_profiles.turn_interpreter == "low"
    assert policy.model_profiles.turn_interpreter_fallback == "medium"
    assert policy.model_profiles.response_compliance == "medium"
