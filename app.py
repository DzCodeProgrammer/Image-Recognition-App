import csv
from datetime import datetime
from io import StringIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.classifier import analyze_image
from src.history import HistoryRepository
from src.translation import translate_label

st.set_page_config(
    page_title="Image Recognition App",
    page_icon=":frame_with_picture:",
    layout="centered",
)

repo = HistoryRepository("history.db")

st.title("Image Recognition App")
st.caption("Upload satu/lebih gambar, lalu sistem menganalisis objek utama.")

with st.sidebar:
    st.header("Pengaturan Analisis")
    top_k = st.slider("Jumlah Top Prediksi", min_value=3, max_value=10, value=5)
    min_conf = st.slider(
        "Filter Confidence Minimum (%)", min_value=0, max_value=100, value=0
    )
    language = st.selectbox("Bahasa Label", options=["id", "en"], index=0)
    save_history = st.toggle("Simpan ke Database", value=True)
    history_limit = st.slider("Jumlah Riwayat Ditampilkan", 10, 500, 100)

    if st.button("Hapus Riwayat Database"):
        repo.clear()
        st.success("Riwayat database dibersihkan.")

uploaded_files = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Hasil Analisis")

    for uploaded_file in uploaded_files:
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

            visible = filtered if filtered else predictions
            for item in visible:
                label = translate_label(item.label, language=language)
                st.write(f"**{label}**")
                st.progress(int(item.confidence), text=f"{item.confidence:.2f}%")

            if save_history and predictions:
                top = predictions[0]
                repo.add(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    filename=uploaded_file.name,
                    top_label=top.label,
                    top_confidence=top.confidence,
                    source="streamlit",
                )
        except (UnidentifiedImageError, OSError):
            st.error(f"File {uploaded_file.name} bukan gambar valid.")
        except ValueError as exc:
            st.error(f"Input tidak valid untuk {uploaded_file.name}: {exc}")
        except Exception as exc:
            st.error(f"Terjadi error saat analisis {uploaded_file.name}: {exc}")

        st.divider()
else:
    st.info("Upload minimal satu gambar untuk mulai klasifikasi.")

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
