# RAVDESS Speech Emotion Recognition

A complete speech-emotion-recognition project trained on the RAVDESS dataset.

## What it does

- Reads RAVDESS filenames and extracts:
  - modality
  - vocal channel
  - emotion
  - intensity
  - statement
  - repetition
  - actor
- Uses **audio-only speech files** (`03-01-...wav`).
- Uses the actor ID for a **speaker-independent train/validation/test split**.
- Converts speech into fixed-size **log-Mel spectrograms**.
- Trains a CNN using PyTorch.
- Evaluates accuracy, macro-F1, confusion matrix and classification report.
- Predicts emotion from any compatible `.wav` speech recording.
- Includes an optional FastAPI endpoint.

## Important RAVDESS detail

For speech emotion recognition, this project intentionally filters:

- Modality = `03` → audio-only
- Vocal channel = `01` → speech

Do not train on `02-01-...wav` if you want the audio-only speech subset. In the official naming scheme, `02` means video-only, while `03` means audio-only.

## Dataset layout

Extract RAVDESS so that the actor folders are inside `data/`.

Example:

data/
  Actor_01/
    03-01-01-01-01-01-01.wav
    ...
  Actor_02/
    ...

You can also point the training script directly at another dataset directory.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data_dir data --epochs 40 --batch_size 32
```

The best model is saved under:

```text
artifacts/best_model.pt
```

## Evaluate

```bash
python src/evaluate.py --data_dir data --checkpoint artifacts/best_model.pt
```

## Predict a new speech file

```bash
python src/predict.py --audio my_speech.wav --checkpoint artifacts/best_model.pt
```

Example output:

```text
Predicted emotion: angry
Confidence: 0.87

Probabilities:
angry      0.87
fearful    0.05
sad        0.03
...
```

## API

Start:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then POST a WAV file to:

```text
POST /predict
```

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@my_speech.wav"
```

## Why speaker-independent splitting?

A random file-level split can leak the same actor into training and testing. That can make the accuracy look better than the model's real generalization.

This project splits by actor:

- Train actors: 01-16
- Validation actors: 17-20
- Test actors: 21-24

The exact split can be changed in `src/data.py`.

## Model

Audio -> resampling -> mono -> fixed 3-second crop/pad -> log-Mel spectrogram -> CNN -> global pooling -> dense layer -> 8 emotion classes.

The eight labels are:

neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Limitations

RAVDESS is acted emotional speech recorded in a controlled environment. A model trained only on RAVDESS will not reliably understand emotion in arbitrary real-world speech.

For a stronger production model, use a pretrained speech representation such as wav2vec 2.0 / HuBERT / WavLM and fine-tune it on RAVDESS plus real-world emotion datasets.
