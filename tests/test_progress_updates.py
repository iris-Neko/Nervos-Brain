from nervos_brain.tool_runtime.progress import ProgressUpdateConfig


def test_progress_message_for_english_uses_builtin_english_defaults():
    cfg = ProgressUpdateConfig(messages=("还在处理 1", "还在处理 2"))

    assert cfg.message_for(0, locale="en") == (
        "I am still checking sources and organizing the evidence. Please wait a moment."
    )


def test_progress_message_for_chinese_uses_configured_messages():
    cfg = ProgressUpdateConfig(messages=("还在处理 1", "还在处理 2"))

    assert cfg.message_for(1, locale="zh-CN") == "还在处理 2"


def test_progress_message_for_english_can_be_overridden_by_locale_config():
    cfg = ProgressUpdateConfig.from_mapping(
        {
            "messages": ["还在处理"],
            "messages_by_locale": {
                "en": ["Still working", "Still checking sources"],
            },
        }
    )

    assert cfg.message_for(0, locale="en-US") == "Still working"
    assert cfg.message_for(2, locale="en-US") == "Still checking sources"


def test_progress_message_for_chinese_falls_back_when_messages_empty():
    cfg = ProgressUpdateConfig(messages=())

    assert cfg.message_for(0, locale="zh-CN") == "我还在处理这个问题，请稍等一下。"
