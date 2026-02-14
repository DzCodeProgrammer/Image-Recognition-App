from dataclasses import dataclass
from functools import lru_cache
from typing import List

import torch
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Load and cache the pretrained ResNet50 model."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def predict(image: Image.Image, top_k: int = 5) -> List[Prediction]:
    """Return top-k predictions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = load_model()(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top_probs, top_indices = torch.topk(probs, top_k)
    labels = weights.meta["categories"]

    return [
        Prediction(label=labels[int(idx)], confidence=float(prob.item() * 100.0))
        for prob, idx in zip(top_probs, top_indices)
    ]


def generate_insight(predictions: List[Prediction]) -> str:
    """Generate a simple explanation from prediction distribution."""
    if not predictions:
        return "Model belum menemukan prediksi."

    top = predictions[0]
    if top.confidence >= 80:
        level = "sangat yakin"
    elif top.confidence >= 60:
        level = "cukup yakin"
    else:
        level = "masih ragu"

    runner_up_text = ""
    if len(predictions) > 1:
        runner_up = predictions[1]
        gap = top.confidence - runner_up.confidence
        if gap >= 20:
            runner_up_text = (
                f" Prediksi utama unggul jauh dari kandidat kedua ({gap:.2f} poin)."
            )
        else:
            runner_up_text = (
                " Selisih prediksi tipis, jadi ada kemungkinan objek terlihat ambigu."
            )

    return (
        f"Model {level} bahwa objek utama adalah '{top.label}' "
        f"dengan confidence {top.confidence:.2f}%."
        f"{runner_up_text}"
    )


def analyze_image(
    image: Image.Image, top_k: int = 5, min_conf: float = 0.0
) -> tuple[List[Prediction], List[Prediction], str]:
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image.")
    if not 0 <= min_conf <= 100:
        raise ValueError("min_conf must be between 0 and 100.")

    predictions = predict(image, top_k=top_k)
    filtered = [item for item in predictions if item.confidence >= min_conf]
    insight = generate_insight(filtered or predictions)
    return predictions, filtered, insight
