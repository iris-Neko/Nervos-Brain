from nervos_brain.tool_runtime.language_detection import detect_message_locale


def test_detects_english_question_over_chinese_fallback():
    assert detect_message_locale("What is CKB, I am a rookie", "zh-CN") == "en"


def test_detects_chinese_question_over_english_fallback():
    assert detect_message_locale("CKB 是什么？我是新手", "en") == "zh-CN"


def test_mixed_language_uses_clear_chinese_signal():
    assert detect_message_locale("请解释 CKB cell model", "en") == "zh-CN"


def test_explicit_english_instruction_wins():
    assert detect_message_locale("Please answer in English: CKB 是什么？", "zh-CN") == "en"


def test_explicit_chinese_instruction_wins():
    assert detect_message_locale("请用中文回答: What is CKB?", "en") == "zh-CN"


def test_ambiguous_messages_use_fallback():
    assert detect_message_locale("", "en") == "en"
    assert detect_message_locale("https://nervos.org", "en") == "en"
    assert detect_message_locale("😀😀", "en") == "en"
    assert detect_message_locale("OK", "zh-CN") == "zh-CN"
