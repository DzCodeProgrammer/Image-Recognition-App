# Image Recognition App

Aplikasi image recognition berbasis **ResNet50** dengan:
- UI interaktif (Streamlit)
- API backend (FastAPI)
- Penyimpanan riwayat prediksi ke SQLite

## Fitur
- Upload satu atau banyak gambar (`jpg`, `jpeg`, `png`, `webp`)
- Pengaturan jumlah top prediksi (3-10)
- Filter confidence minimum
- Terjemahan label (Indonesia/English)
- Ringkasan insight otomatis
- Riwayat prediksi persisten di `history.db`
- Download riwayat ke CSV
- REST API untuk integrasi aplikasi lain

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menjalankan Streamlit
```bash
python -m streamlit run app.py
```

## Menjalankan FastAPI
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint API
- `GET /health`
- `POST /predict` (single image)
- `POST /predict/batch` (multi image)
- `GET /history?limit=100`

Docs otomatis tersedia di:
- `http://localhost:8000/docs`

## Menjalankan Test
```bash
python -m pytest -q
```

## Catatan
- Saat run pertama, bobot model pretrained akan diunduh otomatis oleh `torchvision`.
- `history.db` dibuat otomatis di root project.
