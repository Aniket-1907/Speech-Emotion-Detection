"""Minimal FastAPI endpoint for audio emotion prediction."""
from pathlib import Path
import tempfile

import numpy as np
import os
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from .data import PROJECT_ROOT
from .features import extract_feature
from .model import EmotionCNN
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import imageio_ffmpeg



origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,https://aniket-1907.github.io/Speech-Emotion-Detection/"
    ).split(",")
    if origin.strip()
]


app = FastAPI(title="Speech Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "best_model.pt"
model = None
labels = []

@app.on_event("startup")
def load_model():
    global model, labels
    if not CHECKPOINT_PATH.is_file():
        # Let the API start so its health endpoint can explain missing training.
        return
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    labels = checkpoint["labels"]
    model = EmotionCNN(num_classes=len(labels)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found; train first."
        )

    supported_formats = {
        ".wav", ".flac", ".mp3", ".ogg", ".m4a",
        ".webm", ".mp4", ".mpeg", ".mpga"
    }

    suffix = Path(file.filename or "audio.wav").suffix.lower()

    if suffix not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}"
        )

    payload = await file.read()

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    source_path = None
    wav_path = None

    try:
        # Save the uploaded audio temporarily.
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp.write(payload)
            source_path = Path(temp.name)

        # Create a temporary WAV file for feature extraction.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp:
            wav_path = Path(temp.name)

        # Convert browser audio (WebM/Opus, etc.) into WAV.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", str(source_path),
                "-vn",
                "-ac", "1",
                "-ar", "22050",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail="Could not decode audio. Try uploading a valid WAV file."
            )

        # Use your existing feature extraction pipeline.
        feature = extract_feature(str(wav_path))

        x = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

        order = np.argsort(probs)[::-1]

        return {
            "predicted_emotion": labels[int(order[0])],
            "softmax_score_not_calibrated_confidence": float(
                probs[int(order[0])]
            ),
            "probabilities": {
                labels[int(i)]: float(probs[int(i)])
                for i in order
            },
        }

    except HTTPException:
        raise

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=400,
            detail="Audio conversion timed out. Try a shorter recording."
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process audio: {exc}"
        )

    finally:
        for path in (source_path, wav_path):
            if path and path.exists():
                path.unlink()