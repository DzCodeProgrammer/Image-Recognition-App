from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

from src.classifier import Prediction
from src.media import (
    analyze_image_bytes,
    analyze_video_bytes,
    download_youtube_video_bytes,
    is_youtube_url,
)

ContentKind = Literal["image", "video", "webpage", "pdf", "text", "unknown"]

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "will",
    "you",
    "your",
    "have",
    "has",
    "not",
    "but",
    "dan",
    "yang",
    "untuk",
    "dengan",
    "dari",
    "atau",
    "akan",
    "pada",
    "dalam",
    "tidak",
    "ini",
    "itu",
    "ada",
    "juga",
    "saat",
}


@dataclass(frozen=True)
class URLAnalysisResult:
    url: str
    final_url: str
    content_type: str
    content_kind: ContentKind
    insight: str
    predictions: List[Prediction]
    filtered_predictions: List[Prediction]
    sampled_frames: Optional[int]
    total_frames: Optional[int]
    document: Optional[Dict[str, Any]]


def analyze_url(
    *,
    url: str,
    mode: Literal["auto", "image", "video"] = "auto",
    top_k: int = 5,
    min_conf: float = 0.0,
    sample_every_n_frames: int = 15,
    max_sampled_frames: int = 12,
    timeout: int = 20,
) -> URLAnalysisResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL harus menggunakan http/https")

    if is_youtube_url(url):
        raw, content_type = download_youtube_video_bytes(url)
        analyzed = analyze_video_bytes(
            raw,
            top_k=top_k,
            min_conf=min_conf,
            sample_every_n_frames=sample_every_n_frames,
            max_sampled_frames=max_sampled_frames,
        )
        return URLAnalysisResult(
            url=url,
            final_url=url,
            content_type=content_type,
            content_kind="video",
            insight=analyzed.insight,
            predictions=analyzed.predictions,
            filtered_predictions=analyzed.filtered,
            sampled_frames=analyzed.sampled_frames,
            total_frames=analyzed.total_frames,
            document=None,
        )

    raw, content_type, final_url = _download_url_resource(url, timeout=timeout)
    kind = _classify_content(
        mode=mode,
        url=final_url,
        content_type=content_type,
        raw=raw,
    )

    if kind == "image":
        predictions, filtered, insight = analyze_image_bytes(
            raw,
            top_k=top_k,
            min_conf=min_conf,
        )
        return URLAnalysisResult(
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_kind="image",
            insight=insight,
            predictions=predictions,
            filtered_predictions=filtered,
            sampled_frames=None,
            total_frames=None,
            document=None,
        )

    if kind == "video":
        analyzed = analyze_video_bytes(
            raw,
            top_k=top_k,
            min_conf=min_conf,
            sample_every_n_frames=sample_every_n_frames,
            max_sampled_frames=max_sampled_frames,
        )
        return URLAnalysisResult(
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_kind="video",
            insight=analyzed.insight,
            predictions=analyzed.predictions,
            filtered_predictions=analyzed.filtered,
            sampled_frames=analyzed.sampled_frames,
            total_frames=analyzed.total_frames,
            document=None,
        )

    if kind == "pdf":
        text = _extract_pdf_text(raw)
        document = _build_document_analysis(text=text, title="PDF Document")
        return URLAnalysisResult(
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_kind="pdf",
            insight=f"Dokumen PDF terdeteksi. {document['summary']}",
            predictions=[],
            filtered_predictions=[],
            sampled_frames=None,
            total_frames=None,
            document=document,
        )

    if kind == "webpage":
        title, text = _extract_webpage_text(raw)
        document = _build_document_analysis(text=text, title=title or "Web Page")
        return URLAnalysisResult(
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_kind="webpage",
            insight=f"Halaman web terdeteksi. {document['summary']}",
            predictions=[],
            filtered_predictions=[],
            sampled_frames=None,
            total_frames=None,
            document=document,
        )

    if kind == "text":
        text = raw.decode("utf-8", errors="ignore")
        document = _build_document_analysis(text=text, title="Text Document")
        return URLAnalysisResult(
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_kind="text",
            insight=f"Dokumen teks terdeteksi. {document['summary']}",
            predictions=[],
            filtered_predictions=[],
            sampled_frames=None,
            total_frames=None,
            document=document,
        )

    raise ValueError("URL tidak dikenali sebagai media atau dokumen yang didukung")


def _download_url_resource(url: str, timeout: int) -> tuple[bytes, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "MultimediaRecognitionApp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read(MAX_DOWNLOAD_BYTES + 1)
        final_url = resp.geturl() or url

    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Ukuran file terlalu besar (maks 50MB)")
    return raw, content_type, final_url


def _classify_content(
    *,
    mode: Literal["auto", "image", "video"],
    url: str,
    content_type: str,
    raw: bytes,
) -> ContentKind:
    if mode in {"image", "video"}:
        return mode

    ct = content_type.lower()
    path = urllib.parse.urlparse(url).path.lower()

    if ct.startswith("image/"):
        return "image"
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("application/pdf") or path.endswith(".pdf") or raw.startswith(b"%PDF"):
        return "pdf"
    if ct.startswith("text/html") or _looks_like_html(raw):
        return "webpage"
    if ct.startswith("text/plain") or path.endswith(".txt"):
        return "text"

    if path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return "image"
    if path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return "video"
    return "unknown"


def _looks_like_html(raw: bytes) -> bool:
    head = raw[:1024].decode("utf-8", errors="ignore").lower()
    return "<html" in head or "<!doctype html" in head


def _extract_webpage_text(raw: bytes) -> tuple[str, str]:
    html = raw.decode("utf-8", errors="ignore")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group(1).strip()) if title_match else ""

    cleaned = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, cleaned


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf belum terpasang untuk analisis PDF") from exc

    reader = PdfReader(BytesIO(raw))
    parts = []
    for page in reader.pages[:10]:
        parts.append(page.extract_text() or "")
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_document_analysis(text: str, title: str) -> Dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    summary = " ".join(sentences[:3]).strip() if sentences else ""
    if not summary:
        summary = "Konten teks sangat pendek atau tidak dapat diekstrak."

    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{2,}", normalized.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    keywords = [w for w, _ in Counter(filtered).most_common(10)]

    return {
        "title": title,
        "summary": summary[:800],
        "keywords": keywords,
        "text_preview": normalized[:1200],
        "char_count": len(normalized),
    }
