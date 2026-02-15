# Roadmap 30/60/90 Hari

Dokumen ini memecah pengembangan ke arah aplikasi image recognition yang modern, scalable, dan siap production.

## Tujuan Utama
- Multi-user + secure API
- Arsitektur yang siap scale (async processing)
- Fondasi MLOps (model versioning, monitoring, feedback loop)

## Hari 0-30 (Fondasi Production)
### Fokus
- Security, data model yang lebih rapi, dan operasional dasar.

### Deliverables
1. Authentication & authorization
- JWT login/register sederhana.
- Role `admin` dan `user`.
- Endpoint history dibatasi per user (kecuali admin).

2. Migrasi database
- Pindah dari SQLite ke PostgreSQL untuk environment production.
- Tambah migration tool (`alembic`).
- Skema awal:
  - `users`
  - `prediction_history` (tambahkan `user_id`, `model_version`, `request_id`)

3. API hardening
- Rate limiting per API key / user.
- Validasi input lebih ketat (ukuran file, mime type, parameter).
- Error response konsisten dengan error code internal.

4. Observability minimum
- Structured logging JSON.
- Metrics dasar: latency, throughput, error rate.
- Health check lebih lengkap (`/health/live`, `/health/ready`).

### KPI
- p95 latency single image < 1.5s (tanpa antrean).
- 0 endpoint tanpa auth (kecuali health/public docs jika diizinkan).
- Semua schema change lewat migration.

## Hari 31-60 (Scalability + UX API)
### Fokus
- Decouple inference dari request-response sinkron.

### Deliverables
1. Async job processing
- Tambah Redis + Celery/RQ.
- Endpoint baru:
  - `POST /jobs/predict`
  - `GET /jobs/{job_id}`
  - `GET /jobs/{job_id}/result`
- Batch prediction dipindah ke background worker.

2. Object storage
- Simpan file upload ke S3/MinIO.
- DB hanya menyimpan metadata + URL/path object.

3. Versioning model
- Set `model_version` di setiap hasil prediksi.
- Endpoint info model:
  - `GET /models/current`
  - `GET /models/versions`

4. Dashboard analytics dasar
- Statistik usage (prediksi per hari, top label, average confidence).
- Filter per user/source/date.

### KPI
- API tetap responsif untuk batch besar (request submit < 500ms).
- Worker dapat memproses > 100 image/batch tanpa timeout API.
- Semua hasil prediksi memiliki `model_version` terisi.

## Hari 61-90 (MLOps + Quality Loop)
### Fokus
- Kualitas model berkelanjutan dan visibilitas performa end-to-end.

### Deliverables
1. Feedback loop
- Endpoint feedback:
  - `POST /feedback`
- User bisa koreksi label.
- Data feedback tersimpan untuk dataset retraining.

2. Training & evaluation pipeline
- Pipeline training terjadwal.
- Tracking eksperimen (MLflow/W&B).
- Evaluasi otomatis (accuracy, precision/recall per class).

3. Monitoring drift & alerting
- Pantau distribusi confidence dan top-label drift.
- Alert saat anomali (misal confidence median drop signifikan).

4. Ekstensi model task
- Mulai dukungan object detection (mis. YOLO) untuk use case lanjutan.
- Endpoint terpisah untuk detection.

### KPI
- Ada siklus retraining terjadwal minimal mingguan/bulanan.
- Feedback-to-dataset pipeline berjalan otomatis.
- Alerting aktif untuk error rate dan drift threshold.

## Backlog Prioritas Teknis (Implementasi Bertahap)
1. Tambah modul `src/auth.py` + middleware JWT.
2. Refactor repository pattern agar siap Postgres.
3. Integrasi Alembic migration.
4. Tambah `request_id` dan tracing di semua endpoint.
5. Integrasi Redis queue untuk batch.
6. Tambah endpoint feedback + table `prediction_feedback`.
7. Tambah metrik Prometheus endpoint `/metrics`.

## Rencana Eksekusi Minggu Ini (Quick Start)
1. Implement JWT auth + user table.
2. Batasi `/history` agar hanya menampilkan data milik user login.
3. Tambah `model_version` kolom dan isi default `"resnet50-imagenet-v1"`.
4. Dokumentasi API auth flow di README.
