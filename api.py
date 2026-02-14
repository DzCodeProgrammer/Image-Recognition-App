from datetime import datetime
from io import BytesIO
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.classifier import analyze_image
from src.history import HistoryRepository
from src.translation import translate_label

app = FastAPI(title="Image Recognition API", version="1.0.0")
repo = HistoryRepository("history.db")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
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
    result = [
        {
            "label": translate_label(item.label, language=language),
            "label_raw": item.label,
            "confidence": round(item.confidence, 2),
        }
        for item in visible
    ]

    if predictions:
        top = predictions[0]
        repo.add(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            filename=file.filename or "unknown",
            top_label=top.label,
            top_confidence=top.confidence,
            source="api",
        )

    return {
        "filename": file.filename,
        "insight": insight,
        "predictions": result,
        "count": len(result),
    }


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = 5,
    min_conf: float = 0.0,
    language: str = "id",
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

            rows = [
                {
                    "label": translate_label(item.label, language=language),
                    "label_raw": item.label,
                    "confidence": round(item.confidence, 2),
                }
                for item in visible
            ]

            if predictions:
                top = predictions[0]
                repo.add(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    filename=file.filename or "unknown",
                    top_label=top.label,
                    top_confidence=top.confidence,
                    source="api",
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
def history(limit: int = 100) -> dict:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    rows = repo.list_recent(limit=limit)
    return {"rows": rows, "count": len(rows)}
