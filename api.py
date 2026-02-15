from datetime import datetime
from io import BytesIO
import os
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.auth import (
    AuthRepository,
    User,
    create_access_token,
    create_refresh_token,
    decode_access_token,
)
from src.classifier import analyze_image
from src.history import HistoryRepository
from src.media import analyze_video_bytes
from src.translation import translate_label
from src.url_analyzer import analyze_url

app = FastAPI(title="Multimedia Recognition API", version="1.0.0")
DB_PATH = os.getenv("APP_DB_PATH", "history.db")
repo = HistoryRepository(DB_PATH)
auth_repo = AuthRepository(DB_PATH)
security = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)
    role: str = Field(default="user")


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)


class UpdateRoleRequest(BaseModel):
    role: str = Field(pattern="^(user|admin)$")


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20)


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class URLPredictRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    media_type: str = Field(default="auto", pattern="^(auto|image|video)$")


def _decode_and_validate_token(token: str, expected_type: str) -> dict:
    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token tidak valid atau kedaluwarsa",
        )

    token_type = payload.get("typ")
    if token_type is not None and token_type != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token harus bertipe {expected_type}",
        )

    jti = payload.get("jti")
    if jti and auth_repo.is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token sudah tidak aktif",
        )
    return payload


def get_access_payload(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token dibutuhkan",
        )
    return _decode_and_validate_token(credentials.credentials, "access")


def get_current_user(payload: dict = Depends(get_access_payload)) -> User:
    user = auth_repo.get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User tidak ditemukan",
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Akses admin dibutuhkan")
    return current_user


def parse_iso8601(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} harus format ISO 8601 yang valid",
        )


def _to_response_rows(predictions: list, language: str) -> list[dict]:
    return [
        {
            "label": translate_label(item.label, language=language),
            "label_raw": item.label,
            "confidence": round(item.confidence, 2),
        }
        for item in predictions
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/register")
def register(payload: RegisterRequest) -> dict:
    try:
        user = auth_repo.create_user(
            username=payload.username,
            password=payload.password,
            role=payload.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    token = create_access_token(user_id=user["id"], role=user["role"])
    refresh_token = create_refresh_token(user_id=user["id"], role=user["role"])
    return {
        "user": user,
        "access_token": token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@app.post("/auth/login")
def login(payload: LoginRequest) -> dict:
    user = auth_repo.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Username/password salah")

    token = create_access_token(user_id=user["id"], role=user["role"])
    refresh_token = create_refresh_token(user_id=user["id"], role=user["role"])
    return {
        "user": user,
        "access_token": token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@app.post("/auth/refresh")
def refresh(payload: RefreshRequest) -> dict:
    decoded = _decode_and_validate_token(payload.refresh_token, "refresh")
    user = auth_repo.get_user_by_id(int(decoded["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="User tidak ditemukan")

    old_jti = decoded.get("jti")
    if old_jti:
        auth_repo.revoke_token(old_jti)

    new_access_token = create_access_token(user_id=user["id"], role=user["role"])
    new_refresh_token = create_refresh_token(user_id=user["id"], role=user["role"])
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


@app.post("/auth/logout")
def logout(
    payload: LogoutRequest,
    access_payload: dict = Depends(get_access_payload),
) -> dict:
    access_jti = access_payload.get("jti")
    if access_jti:
        auth_repo.revoke_token(access_jti)

    if payload.refresh_token:
        decoded_refresh = _decode_and_validate_token(payload.refresh_token, "refresh")
        refresh_jti = decoded_refresh.get("jti")
        if refresh_jti:
            auth_repo.revoke_token(refresh_jti)

    return {"message": "logout berhasil"}


@app.patch("/admin/users/{user_id}/role")
def update_user_role(
    user_id: int,
    payload: UpdateRoleRequest,
    _: User = Depends(require_admin),
) -> dict:
    try:
        updated = auth_repo.update_role(user_id=user_id, role=payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if updated is None:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    return {"user": updated}


@app.get("/admin/users")
def list_users(
    limit: int = 50,
    offset: int = 0,
    _: User = Depends(require_admin),
) -> dict:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    users = auth_repo.list_users(limit=limit, offset=offset)
    total = auth_repo.count_users()
    return {
        "rows": users,
        "count": len(users),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        predictions, filtered, insight = analyze_image(image, top_k=top_k, min_conf=min_conf)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File bukan gambar valid.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    visible = filtered if filtered else predictions
    result = _to_response_rows(visible, language)

    if predictions:
        top = predictions[0]
        repo.add(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            filename=file.filename or "unknown",
            top_label=top.label,
            top_confidence=top.confidence,
            source="api",
            top_predictions=[
                {
                    "label": item.label,
                    "confidence": round(item.confidence, 2),
                }
                for item in predictions
            ],
            user_id=current_user["id"],
        )

    return {
        "filename": file.filename,
        "insight": insight,
        "predictions": result,
        "count": len(result),
    }


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
    sample_every_n_frames: int = 15,
    max_sampled_frames: int = 12,
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        content = await file.read()
        analyzed = analyze_video_bytes(
            content,
            top_k=top_k,
            min_conf=min_conf,
            sample_every_n_frames=sample_every_n_frames,
            max_sampled_frames=max_sampled_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    visible = analyzed.filtered if analyzed.filtered else analyzed.predictions
    result = _to_response_rows(visible, language)

    if analyzed.predictions:
        top = analyzed.predictions[0]
        repo.add(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            filename=file.filename or "unknown-video",
            top_label=top.label,
            top_confidence=top.confidence,
            source="api_video",
            top_predictions=[
                {"label": item.label, "confidence": round(item.confidence, 2)}
                for item in analyzed.predictions
            ],
            user_id=current_user["id"],
        )

    return {
        "filename": file.filename,
        "insight": analyzed.insight,
        "predictions": result,
        "count": len(result),
        "sampled_frames": analyzed.sampled_frames,
        "total_frames": analyzed.total_frames,
    }


@app.post("/predict/url")
def predict_from_url(
    payload: URLPredictRequest,
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
    sample_every_n_frames: int = 15,
    max_sampled_frames: int = 12,
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        analyzed = analyze_url(
            url=payload.url,
            mode=payload.media_type,
            top_k=top_k,
            min_conf=min_conf,
            sample_every_n_frames=sample_every_n_frames,
            max_sampled_frames=max_sampled_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    predictions = analyzed.predictions
    visible = analyzed.filtered_predictions if analyzed.filtered_predictions else analyzed.predictions
    result = _to_response_rows(visible, language)

    if predictions:
        top = predictions[0]
        save_source = f"api_url_{analyzed.content_kind}"
        repo.add(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            filename=analyzed.final_url,
            top_label=top.label,
            top_confidence=top.confidence,
            source=save_source,
            top_predictions=[
                {"label": item.label, "confidence": round(item.confidence, 2)}
                for item in predictions
            ],
            user_id=current_user["id"],
        )

    return {
        "url": payload.url,
        "final_url": analyzed.final_url,
        "content_type": analyzed.content_type,
        "content_kind": analyzed.content_kind,
        "insight": analyzed.insight,
        "predictions": result,
        "count": len(result),
        "sampled_frames": analyzed.sampled_frames,
        "total_frames": analyzed.total_frames,
        "document": analyzed.document,
    }


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
    current_user: User = Depends(get_current_user),
) -> dict:
    outputs = []
    for file in files:
        try:
            content = await file.read()
            image = Image.open(BytesIO(content)).convert("RGB")
            predictions, filtered, insight = analyze_image(
                image, top_k=top_k, min_conf=min_conf
            )
            visible = filtered if filtered else predictions

            rows = _to_response_rows(visible, language)

            if predictions:
                top = predictions[0]
                repo.add(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    filename=file.filename or "unknown",
                    top_label=top.label,
                    top_confidence=top.confidence,
                    source="api",
                    top_predictions=[
                        {
                            "label": item.label,
                            "confidence": round(item.confidence, 2),
                        }
                        for item in predictions
                    ],
                    user_id=current_user["id"],
                )

            outputs.append(
                {
                    "filename": file.filename,
                    "insight": insight,
                    "predictions": rows,
                    "error": None,
                }
            )
        except (UnidentifiedImageError, OSError):
            outputs.append(
                {
                    "filename": file.filename,
                    "insight": None,
                    "predictions": [],
                    "error": "File bukan gambar valid.",
                }
            )
        except ValueError as exc:
            outputs.append(
                {
                    "filename": file.filename,
                    "insight": None,
                    "predictions": [],
                    "error": str(exc),
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "filename": file.filename,
                    "insight": None,
                    "predictions": [],
                    "error": str(exc),
                }
            )

    return {"results": outputs, "count": len(outputs)}


@app.get("/history")
def history(
    limit: int = 100,
    offset: int = 0,
    source: str | None = None,
    label: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_predictions: bool = False,
    user_id: int | None = None,
    current_user: User = Depends(get_current_user),
) -> dict:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if date_from:
        parse_iso8601(date_from, "date_from")
    if date_to:
        parse_iso8601(date_to, "date_to")
    if date_from and date_to:
        parsed_from = parse_iso8601(date_from, "date_from")
        parsed_to = parse_iso8601(date_to, "date_to")
        if parsed_from > parsed_to:
            raise HTTPException(
                status_code=400,
                detail="date_from tidak boleh lebih besar dari date_to",
            )

    scoped_user_id = user_id
    if current_user["role"] != "admin":
        scoped_user_id = current_user["id"]

    rows = repo.list_recent(
        limit=limit,
        offset=offset,
        source=source,
        label=label,
        date_from=date_from,
        date_to=date_to,
        include_predictions=include_predictions,
        user_id=scoped_user_id,
    )
    total = repo.count(
        source=source,
        label=label,
        date_from=date_from,
        date_to=date_to,
        user_id=scoped_user_id,
    )
    return {
        "rows": rows,
        "count": len(rows),
        "total": total,
        "limit": limit,
        "offset": offset,
    }
