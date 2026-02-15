import csv
from datetime import datetime
from io import StringIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.classifier import analyze_image
from src.history import HistoryRepository
from src.media import analyze_video_bytes
from src.translation import translate_label
from src.url_analyzer import analyze_url

st.set_page_config(
    page_title="Multimedia Recognition App",
    page_icon=":frame_with_picture:",
    layout="centered",
)

repo = HistoryRepository("history.db")

st.title("Multimedia Recognition App")
st.caption("Analisis objek dari gambar, video, atau URL media.")

with st.sidebar:
    st.header("Pengaturan Analisis")
    top_k = st.slider("Jumlah Top Prediksi", min_value=3, max_value=10, value=5)
    min_conf = st.slider(
        "Filter Confidence Minimum (%)", min_value=0, max_value=100, value=0
    )
    language = st.selectbox("Bahasa Label", options=["id", "en"], index=0)
    save_history = st.toggle("Simpan ke Database", value=True)
    sample_every_n_frames = st.slider("Sampling Frame Video", 1, 60, 15)
    max_sampled_frames = st.slider("Maksimum Frame Sampel", 1, 30, 12)
    history_limit = st.slider("Jumlah Riwayat Ditampilkan", 10, 500, 100)

    if st.button("Hapus Riwayat Database"):
        repo.clear()
        st.success("Riwayat database dibersihkan.")

def render_predictions(predictions):
    for item in predictions:
        label = translate_label(item.label, language=language)
        st.write(f"**{label}**")
        st.progress(int(item.confidence), text=f"{item.confidence:.2f}%")


def save_history_row(name: str, source: str, predictions):
    if not (save_history and predictions):
        return
    top = predictions[0]
    repo.add(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        filename=name,
        top_label=top.label,
        top_confidence=top.confidence,
        source=source,
        top_predictions=[
            {"label": item.label, "confidence": round(item.confidence, 2)}
            for item in predictions
        ],
    )


image_tab, video_tab, url_tab = st.tabs(["Image", "Video", "URL"])

with image_tab:
    uploaded_images = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="image_uploader",
    )
    if uploaded_images:
        st.subheader("Hasil Analisis Gambar")
        for uploaded_file in uploaded_images:
            st.markdown(f"### {uploaded_file.name}")
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption=f"Input: {uploaded_file.name}",
                    use_container_width=True,
                )
                with st.spinner(f"Menganalisis {uploaded_file.name}..."):
                    predictions, filtered, insight = analyze_image(
                        image=image,
                        top_k=top_k,
                        min_conf=float(min_conf),
                    )
                st.info(insight)
                render_predictions(filtered if filtered else predictions)
                save_history_row(uploaded_file.name, "streamlit_image", predictions)
            except (UnidentifiedImageError, OSError):
                st.error(f"File {uploaded_file.name} bukan gambar valid.")
            except ValueError as exc:
                st.error(f"Input tidak valid untuk {uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"Terjadi error saat analisis {uploaded_file.name}: {exc}")
            st.divider()
    else:
        st.info("Upload minimal satu gambar untuk mulai klasifikasi.")

with video_tab:
    uploaded_videos = st.file_uploader(
        "Pilih video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
        key="video_uploader",
    )
    if uploaded_videos:
        st.subheader("Hasil Analisis Video")
        for uploaded_file in uploaded_videos:
            st.markdown(f"### {uploaded_file.name}")
            try:
                raw = uploaded_file.getvalue()
                st.video(raw)
                with st.spinner(f"Menganalisis video {uploaded_file.name}..."):
                    analyzed = analyze_video_bytes(
                        raw,
                        top_k=top_k,
                        min_conf=float(min_conf),
                        sample_every_n_frames=sample_every_n_frames,
                        max_sampled_frames=max_sampled_frames,
                    )
                st.info(analyzed.insight)
                st.caption(
                    f"Frame dianalisis: {analyzed.sampled_frames} dari total {analyzed.total_frames}"
                )
                visible = analyzed.filtered if analyzed.filtered else analyzed.predictions
                render_predictions(visible)
                save_history_row(uploaded_file.name, "streamlit_video", analyzed.predictions)
            except ValueError as exc:
                st.error(f"Input video tidak valid untuk {uploaded_file.name}: {exc}")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Terjadi error saat analisis {uploaded_file.name}: {exc}")
            st.divider()
    else:
        st.info("Upload minimal satu video untuk mulai analisis.")

with url_tab:
    st.subheader("Analisis dari URL")
    media_mode = st.selectbox("Jenis Media URL", ["auto", "image", "video"], index=0)
    media_url = st.text_input("Masukkan URL gambar/video", value="")
    if st.button("Analisis URL"):
        if not media_url.strip():
            st.warning("URL tidak boleh kosong.")
        else:
            try:
                with st.spinner("Mendownload dan menganalisis URL..."):
                    analyzed = analyze_url(
                        url=media_url.strip(),
                        mode=media_mode,
                        top_k=top_k,
                        min_conf=float(min_conf),
                        sample_every_n_frames=sample_every_n_frames,
                        max_sampled_frames=max_sampled_frames,
                    )

                    st.caption(
                        f"Tipe konten: {analyzed.content_kind} | Content-Type: {analyzed.content_type}"
                    )
                    st.info(analyzed.insight)

                    if analyzed.content_kind in {"image", "video"}:
                        visible = (
                            analyzed.filtered_predictions
                            if analyzed.filtered_predictions
                            else analyzed.predictions
                        )
                        render_predictions(visible)
                        if analyzed.content_kind == "video":
                            st.caption(
                                f"Frame dianalisis: {analyzed.sampled_frames} dari total {analyzed.total_frames}"
                            )
                        save_history_row(
                            analyzed.final_url,
                            f"streamlit_url_{analyzed.content_kind}",
                            analyzed.predictions,
                        )
                    else:
                        doc = analyzed.document or {}
                        if doc.get("title"):
                            st.markdown(f"**Judul:** {doc['title']}")
                        if doc.get("summary"):
                            st.write(doc["summary"])
                        keywords = doc.get("keywords") or []
                        if keywords:
                            st.markdown("**Keyword Utama:** " + ", ".join(keywords))
                        if doc.get("text_preview"):
                            with st.expander("Preview Konten"):
                                st.write(doc["text_preview"])
            except ValueError as exc:
                st.error(f"URL/Input tidak valid: {exc}")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Terjadi error saat analisis URL: {exc}")

history_rows = repo.list_recent(limit=history_limit)
if history_rows:
    st.subheader("Riwayat Prediksi (Database)")
    st.dataframe(history_rows, use_container_width=True)

    csv_buffer = StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["timestamp", "filename", "top_label", "top_confidence", "source"],
    )
    writer.writeheader()
    writer.writerows(history_rows)

    st.download_button(
        label="Download Riwayat (CSV)",
        data=csv_buffer.getvalue(),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
