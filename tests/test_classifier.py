from src.classifier import Prediction, generate_insight


def test_generate_insight_high_confidence():
    predictions = [
        Prediction(label="cat", confidence=91.0),
        Prediction(label="dog", confidence=20.0),
    ]
    text = generate_insight(predictions)
    assert "sangat yakin" in text
    assert "cat" in text


def test_generate_insight_low_confidence():
    predictions = [
        Prediction(label="object", confidence=51.0),
        Prediction(label="other", confidence=45.0),
    ]
    text = generate_insight(predictions)
    assert "masih ragu" in text
    assert "ambigu" in text
