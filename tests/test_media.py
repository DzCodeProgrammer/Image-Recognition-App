import pytest
from pathlib import Path

from src import media
from src.media import (
    analyze_image_bytes,
    detect_media_kind,
    extract_media_urls_from_html,
    fetch_url_bytes,
    is_youtube_url,
)


def test_detect_media_kind_from_content_type():
    kind = detect_media_kind(
        url="https://example.com/file.bin",
        content_type="image/png",
        mode="auto",
    )
    assert kind == "image"


def test_detect_media_kind_from_extension():
    kind = detect_media_kind(
        url="https://example.com/video.mp4",
        content_type="application/octet-stream",
        mode="auto",
    )
    assert kind == "video"


def test_fetch_url_bytes_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        fetch_url_bytes("file:///etc/passwd")


def test_analyze_image_bytes_invalid_content():
    with pytest.raises(ValueError):
        analyze_image_bytes(b"not-an-image", top_k=5, min_conf=0.0)


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("https://youtu.be/abc123")
    assert not is_youtube_url("https://example.com/video.mp4")


def test_fetch_url_bytes_uses_youtube_downloader(monkeypatch):
    monkeypatch.setattr(
        media,
        "download_youtube_video_bytes",
        lambda url: (b"video-bytes", "video/mp4"),
    )
    raw, content_type = fetch_url_bytes("https://www.youtube.com/watch?v=abc123")
    assert raw == b"video-bytes"
    assert content_type == "video/mp4"


def test_extract_media_urls_from_html():
    html = """
    <html>
      <head>
        <meta property="og:image" content="/assets/cover.jpg" />
      </head>
      <body>
        <video src="https://cdn.example.com/a.mp4"></video>
      </body>
    </html>
    """
    urls = extract_media_urls_from_html(html, base_url="https://example.com/post/123")
    assert "https://cdn.example.com/a.mp4" in urls
    assert "https://example.com/assets/cover.jpg" in urls


def test_fetch_url_bytes_scrapes_html_and_resolves_media(monkeypatch):
    calls = []

    def fake_download(url: str, timeout: int):
        calls.append(url)
        if url == "https://example.com/post":
            return (
                b'<html><head><meta property="og:image" content="https://cdn.example.com/pic.jpg"></head></html>',
                "text/html",
                "https://example.com/post",
            )
        if url == "https://cdn.example.com/pic.jpg":
            return (b"jpeg-bytes", "image/jpeg", url)
        raise ValueError("unexpected url")

    monkeypatch.setattr(media, "_download_url_bytes", fake_download)
    raw, content_type = fetch_url_bytes("https://example.com/post")
    assert raw == b"jpeg-bytes"
    assert content_type == "image/jpeg"
    assert calls == ["https://example.com/post", "https://cdn.example.com/pic.jpg"]


def test_resolve_downloaded_video_path_prefers_requested_downloads(tmp_path: Path):
    preferred = tmp_path / "video.webm"
    preferred.write_bytes(b"x")
    out = media._resolve_downloaded_video_path(
        info={"requested_downloads": [{"filepath": str(preferred)}]},
        prepared_filename=str(tmp_path / "unused.mp4"),
        tmpdir=str(tmp_path),
    )
    assert out == str(preferred)


def test_resolve_downloaded_video_path_fallback_scans_tmpdir(tmp_path: Path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.webm"
    a.write_bytes(b"123")
    b.write_bytes(b"12345")
    out = media._resolve_downloaded_video_path(
        info={},
        prepared_filename=str(tmp_path / "missing.mp4"),
        tmpdir=str(tmp_path),
    )
    assert out == str(b)
