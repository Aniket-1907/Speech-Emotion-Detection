# Speech Emotion Lab — Local UI

A lightweight, responsive browser UI for testing the existing speech-emotion FastAPI model. Includes audio file upload, microphone recording, prediction display, and score visualization. It does not train the model or upload data to a third-party service.

## 1. Start the model API

From the root of your `Speech-Emotion-Detection` project, activate the same Python environment used for training, then run:

```powershell
python -m src.api
```

If your `src/api.py` does not start Uvicorn when run as a module, start it with:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and verify that the API exposes `POST /predict`. The UI sends audio as multipart form data under the field name `file`, and expects a JSON response containing an emotion field (`emotion`, `predicted_emotion`, `prediction`, or `label`) and optionally a scores/probabilities object.

## 2. Serve the UI

Unzip this folder somewhere convenient. Open a second PowerShell terminal in the folder containing `index.html` and run:

```powershell
python -m http.server 5500
```

Open `http://127.0.0.1:5500`.

Use the API base URL field in the UI; default is `http://127.0.0.1:8000`.

## 3. Enable CORS in FastAPI if needed

Because the UI and API use different ports, the browser may block requests unless CORS is enabled. In `src/api.py`, add this after creating your FastAPI `app`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Do not create a second `app = FastAPI()`; add the middleware to the app that already exists.

## 4. Match the API contract

The UI posts to `POST /predict` and sends the uploaded audio under the multipart field `file`. If your endpoint uses another route or field name, change the fetch URL or `form.append("file", ...)` in `app.js`.

The UI supports JSON responses such as:

```json
{
  "emotion": "sad",
  "scores": {
    "sad": 0.46,
    "fearful": 0.23,
    "happy": 0.08
  },
  "model_version": "v1"
}
```

This is an example response shape only. The frontend also accepts several common alternate field names. If your actual API response differs, update `findEmotion()` and `findScores()` in `app.js`.

## Notes

- WAV is recommended for predictable compatibility with audio-processing libraries.
- Microphone recording commonly produces WebM/Opus in Chrome/Edge. Your backend's audio loader must support that format, or convert it to WAV before feature extraction.
- The displayed scores are whatever the model API returns. They are not guaranteed to be calibrated probabilities.
- This UI is for local testing. Add authentication, request-size limits, rate limiting, consent, and persistent storage before exposing it publicly.
