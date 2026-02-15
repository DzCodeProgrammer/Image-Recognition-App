from src.classifier import Prediction
from src.url_analyzer import analyze_url


def test_analyze_url_webpage(monkeypatch):
    import src.url_analyzer as analyzer

    monkeypatch.setattr(
        analyzer,
        "_download_url_resource",
        lambda url, timeout: (
            b"<html><head><title>Hello</title></head><body>Ini halaman web untuk testing konten.</body></html>",
            "text/html",
            url,
        ),
    )

    result = analyze_url(url="https://example.com/page", mode="auto")
    assert result.content_kind == "webpage"
    assert result.document is not None
    assert result.document["title"] == "Hello"


def test_analyze_url_image(monkeypatch):
    import src.url_analyzer as analyzer

    monkeypatch.setattr(
        analyzer,
        "_download_url_resource",
        lambda url, timeout: (b"img", "image/png", url),
    )
    monkeypatch.setattr(
        analyzer,
        "analyze_image_bytes",
        lambda raw, top_k, min_conf: (
            [Prediction(label="cat", confidence=91.2)],
            [Prediction(label="cat", confidence=91.2)],
            "ok",
        ),
    )

    result = analyze_url(url="https://example.com/a.png", mode="auto")
    assert result.content_kind == "image"
    assert result.predictions[0].label == "cat"
