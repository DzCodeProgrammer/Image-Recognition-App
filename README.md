# Multimedia Recognition App

![Multimedia Recognition App Preview](docs/images/preview.png)

Aplikasi recognition multimedia berbasis **ResNet50** dengan:
- UI interaktif (Streamlit)
- API backend (FastAPI)
- Penyimpanan riwayat prediksi ke SQLite

## Fitur
- Upload satu atau banyak gambar (`jpg`, `jpeg`, `png`, `webp`)
- Upload video (`mp4`, `mov`, `avi`, `mkv`, `webm`)
- Analisis URL universal: gambar, video, halaman web, PDF, teks
- Dukungan URL YouTube (otomatis diunduh lalu dianalisis sebagai video)
- Pengaturan jumlah top prediksi (3-10)
- Filter confidence minimum
- Terjemahan label (Indonesia/English)
- Ringkasan insight otomatis
- Riwayat prediksi persisten di `history.db`
- Simpan top-k prediksi penuh ke database (format JSON)
- Index database otomatis untuk query history yang lebih cepat
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
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8011
```

### Endpoint API
- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/refresh`
- `POST /auth/logout`
- `GET /admin/users?limit=50&offset=0` (admin only)
- `PATCH /admin/users/{user_id}/role` (admin only)
- `POST /predict` (single image)
- `POST /predict/video` (single video)
- `POST /predict/url` (image/video URL)
- `POST /predict/url` (analisis konten URL universal)
- `POST /predict/batch` (multi image)
- `GET /history?limit=100&offset=0&source=api&label=cat&date_from=2026-02-14T00:00:00&date_to=2026-02-14T23:59:59&include_predictions=true&user_id=1`

## Auth Flow API
1. Register user:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"demo\",\"password\":\"demo12345\",\"role\":\"user\"}"
```
2. Login untuk ambil token:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"demo\",\"password\":\"demo12345\"}"
```
3. Refresh token saat access token expired:
```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"<REFRESH_TOKEN>\"}"
```
4. Pakai access token di endpoint protected:
```bash
curl -X GET "http://localhost:8000/history?limit=20" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```
5. Logout dan revoke token:
```bash
curl -X POST http://localhost:8000/auth/logout \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"<REFRESH_TOKEN>\"}"
```

Catatan:
- Endpoint `predict`, `predict/batch`, dan `history` sekarang membutuhkan bearer token.
- Endpoint `predict/video` dan `predict/url` juga membutuhkan bearer token.
- User biasa hanya bisa melihat history miliknya sendiri.
- Role `admin` bisa filter semua user dengan query `user_id`.
- Set environment variable `APP_SECRET_KEY` di server untuk secret JWT production.
- Opsional: set `APP_DB_PATH` untuk lokasi database custom (default `history.db`).
- Parameter `date_from` dan `date_to` di `/history` harus format ISO 8601 valid.
- Analisis video membutuhkan paket `opencv-python-headless`.
- Analisis URL YouTube membutuhkan paket `yt-dlp`.
- Analisis PDF dari URL membutuhkan paket `pypdf`.
- Untuk URL YouTube, gunakan link video tunggal (bukan playlist).
- Untuk URL halaman web, sistem mengekstrak teks halaman lalu memberi ringkasan dan keyword utama.

Docs otomatis tersedia di:
- `http://localhost:8011/docs`

## Menjalankan Test
```bash
python -m pytest -q
```

## Catatan
- Saat run pertama, bobot model pretrained akan diunduh otomatis oleh `torchvision`.
- `history.db` dibuat otomatis di root project.

## Roadmap
- Lihat rencana pengembangan modern dan kompleks di `ROADMAP.md`.
