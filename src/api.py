from pathlib import Path
import tempfile
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

from .model import EmotionCNN
from .features import extract_feature

app = FastAPI(title="RAVDESS Speech Emotion API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("artifacts/best_model.pt")

if not CHECKPOINT_PATH.exists():
    raise RuntimeError(
        "Model checkpoint not found. Train the model first with src/train.py."
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
LABELS = checkpoint["labels"]

MODEL = EmotionCNN(num_classes=len(LABELS)).to(DEVICE)
MODEL.load_state_dict(checkpoint["model_state"])
MODEL.eval()

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix.lower()

    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(400, "Unsupported audio format.")

    data = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(data)
        temp_path = temp.name

    try:
        spec = extract_feature(temp_path)
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probabilities = torch.softmax(MODEL(x), dim=1)[0].cpu().numpy()

        order = np.argsort(probabilities)[::-1]

        return {
            "emotion": LABELS[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "probabilities": {
                LABELS[int(i)]: float(probabilities[i])
                for i in order
            }
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)
