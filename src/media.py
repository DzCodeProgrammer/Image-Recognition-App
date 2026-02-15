from __future__ import annotations

import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal

from PIL import Image, UnidentifiedImageError

from src.classifier import Prediction, analyze_image, generate_insight

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_VIDEO_SAMPLE_EVERY = 15
DEFAULT_VIDEO_MAX_FRAMES = 12

MediaKind = Literal["image", "video"]


@dataclass(frozen=True)
class VideoAnalysis:
    predictions: List[Prediction]
    filtered: List[Prediction]
    insight: str
    sampled_frames: int
    total_frames: int


def fetch_url_bytes(url: str, timeout: int = 20) -> tuple[bytes, str]:
    return _fetch_url_bytes(url=url, timeout=timeout, depth=0, visited=set())


def _fetch_url_bytes(
    *,
    url: str,
    timeout: int,
    depth: int,
    visited: set[str],
) -> tuple[bytes, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL harus menggunakan http/https")
    normalized_url = urllib.parse.urldefrag(url).url
    if normalized_url in visited:
        raise ValueError("Terjadi loop URL saat mengambil media")
    visited.add(normalized_url)

    if is_youtube_url(url):
        return download_youtube_video_bytes(url)

    content, content_type, final_url = _download_url_bytes(url=url, timeout=timeout)
    lowered_ct = content_type.lower()

    is_html = lowered_ct.startswith("text/html") or _looks_like_html(content)
    if is_html and depth < 1:
        html_text = content.decode("utf-8", errors="ignore")
        candidates = extract_media_urls_from_html(html_text, base_url=final_url)
        for candidate in candidates:
            try:
                return _fetch_url_bytes(
                    url=candidate,
                    timeout=timeout,
                    depth=depth + 1,
                    visited=visited,
                )
            except ValueError:
                continue
        raise ValueError("Tidak bisa menentukan jenis media dari URL")

    return content, content_type


def _download_url_bytes(url: str, timeout: int) -> tuple[bytes, str, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ImageRecognitionApp/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        content = resp.read(MAX_DOWNLOAD_BYTES + 1)
        final_url = resp.geturl() or url

    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("File dari URL terlalu besar (maks 50MB)")
    return content, content_type, final_url


def _looks_like_html(raw: bytes) -> bool:
    snippet = raw[:1024].decode("utf-8", errors="ignore").lower()
    return "<html" in snippet or "<!doctype html" in snippet


def is_youtube_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    return (
        "youtube.com" in host
        or "youtu.be" in host
        or "m.youtube.com" in host
    )


def download_youtube_video_bytes(url: str) -> tuple[bytes, str]:
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise RuntimeError("yt-dlp belum terpasang untuk URL YouTube") from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "yt_video.%(ext)s")
        opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": output_template,
            # Hindari merge audio+video agar tetap jalan tanpa ffmpeg.
            "format": "bv*[height<=720]/bestvideo[height<=720]/best[height<=720]/best",
            "max_filesize": MAX_DOWNLOAD_BYTES,
            "noplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = ydl.prepare_filename(info)
        except Exception as exc:
            raise ValueError(f"Gagal mengambil video YouTube: {exc}") from exc

        final_path = _resolve_downloaded_video_path(
            info=info,
            prepared_filename=downloaded,
            tmpdir=tmpdir,
        )

        if not os.path.exists(final_path):
            raise ValueError("File video YouTube tidak ditemukan setelah download")

        with open(final_path, "rb") as f:
            raw = f.read(MAX_DOWNLOAD_BYTES + 1)
        if len(raw) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Video YouTube terlalu besar (maks 50MB)")
        ext = Path(final_path).suffix.lower()
        content_type = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".mov": "video/quicktime",
        }.get(ext, "video/mp4")
        return raw, content_type


def _resolve_downloaded_video_path(
    *,
    info: dict,
    prepared_filename: str,
    tmpdir: str,
) -> str:
    requested = info.get("requested_downloads") or []
    for item in requested:
        filepath = item.get("filepath")
        if filepath and os.path.exists(filepath):
            return filepath

    if os.path.exists(prepared_filename):
        return prepared_filename

    base, _ = os.path.splitext(prepared_filename)
    candidates = [f"{base}.mp4", f"{base}.mkv", f"{base}.webm", f"{base}.mov"]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    video_exts = {".mp4", ".webm", ".mkv", ".mov"}
    tmp_candidates = sorted(
        [
            str(p)
            for p in Path(tmpdir).glob("*")
            if p.is_file() and p.suffix.lower() in video_exts
        ],
        key=lambda p: os.path.getsize(p),
        reverse=True,
    )
    if tmp_candidates:
        return tmp_candidates[0]

    return prepared_filename


def detect_media_kind(
    *,
    url: str,
    content_type: str,
    mode: Literal["auto", "image", "video"] = "auto",
) -> MediaKind:
    if mode in {"image", "video"}:
        return mode

    lowered_ct = content_type.lower()
    if lowered_ct.startswith("image/"):
        return "image"
    if lowered_ct.startswith("video/"):
        return "video"

    path = urllib.parse.urlparse(url).path.lower()
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if path.endswith(image_exts):
        return "image"
    if path.endswith(video_exts):
        return "video"
    raise ValueError("Tidak bisa menentukan jenis media dari URL")


def extract_media_urls_from_html(html: str, base_url: str) -> List[str]:
    # Prioritaskan metadata OpenGraph/Twitter lalu fallback ke tag media.
    patterns = [
        r'<meta[^>]+property=["\']og:video(?::url|:secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:player:stream["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:image(?::url|:secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<video[^>]+src=["\']([^"\']+)["\']',
        r'<source[^>]+src=["\']([^"\']+)["\']',
        r'<img[^>]+src=["\']([^"\']+)["\']',
    ]
    candidates: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            raw_url = unescape(match.strip())
            if not raw_url:
                continue
            absolute = urllib.parse.urljoin(base_url, raw_url)
            absolute = urllib.parse.urldefrag(absolute).url
            parsed = urllib.parse.urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            candidates.append(absolute)
    return candidates


def analyze_image_bytes(
    raw: bytes,
    *,
    top_k: int,
    min_conf: float,
) -> tuple[List[Prediction], List[Prediction], str]:
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Konten bukan gambar valid") from exc
    return analyze_image(image=image, top_k=top_k, min_conf=min_conf)


def _aggregate_predictions(
    all_predictions: List[List[Prediction]],
    *,
    top_k: int,
) -> List[Prediction]:
    if not all_predictions:
        return []

    score_sum: Dict[str, float] = {}
    count: Dict[str, int] = {}

    for frame_predictions in all_predictions:
        for item in frame_predictions:
            score_sum[item.label] = score_sum.get(item.label, 0.0) + item.confidence
            count[item.label] = count.get(item.label, 0) + 1

    averaged = [
        Prediction(label=label, confidence=(score_sum[label] / count[label]))
        for label in score_sum
    ]
    averaged.sort(key=lambda x: x.confidence, reverse=True)
    return averaged[:top_k]


def analyze_video_bytes(
    raw: bytes,
    *,
    top_k: int,
    min_conf: float,
    sample_every_n_frames: int = DEFAULT_VIDEO_SAMPLE_EVERY,
    max_sampled_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
) -> VideoAnalysis:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless belum terpasang") from exc

    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames harus >= 1")
    if max_sampled_frames < 1:
        raise ValueError("max_sampled_frames harus >= 1")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw)
        temp_path = tmp.name

    sampled_frames = 0
    total_frames = 0
    collected: List[List[Prediction]] = []
    capture = cv2.VideoCapture(temp_path)
    try:
        if not capture.isOpened():
            raise ValueError("Video tidak dapat dibuka")

        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            total_frames += 1
            if frame_idx % sample_every_n_frames != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            predictions, _, _ = analyze_image(image=image, top_k=top_k, min_conf=0.0)
            collected.append(predictions)
            sampled_frames += 1
            frame_idx += 1

            if sampled_frames >= max_sampled_frames:
                break
    finally:
        capture.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if sampled_frames == 0:
        raise ValueError("Tidak ada frame yang bisa dianalisis dari video")

    predictions = _aggregate_predictions(collected, top_k=top_k)
    filtered = [item for item in predictions if item.confidence >= min_conf]
    insight = (
        f"Analisis video dari {sampled_frames} frame sampel. "
        + generate_insight(filtered or predictions)
    )
    return VideoAnalysis(
        predictions=predictions,
        filtered=filtered,
        insight=insight,
        sampled_frames=sampled_frames,
        total_frames=total_frames,
    )
