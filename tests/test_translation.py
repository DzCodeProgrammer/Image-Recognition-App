from src.translation import translate_label


def test_translate_label_id_known():
    assert translate_label("cat", "id") == "kucing"


def test_translate_label_id_unknown_returns_raw():
    assert translate_label("unknown-label", "id") == "unknown-label"


def test_translate_label_en_returns_raw():
    assert translate_label("cat", "en") == "cat"
