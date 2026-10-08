import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

def load_audio(path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(path, sr=sr, mono=True)

    # Normalize amplitude.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        # Center crop for deterministic inference.
        start = (len(audio) - NUM_SAMPLES) // 2
        audio = audio[start:start + NUM_SAMPLES]

    return audio.astype(np.float32)

def audio_to_logmel(audio, sr=SAMPLE_RATE, n_mels=64):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # Per-sample standardization.
    mean = logmel.mean()
    std = logmel.std() + 1e-6
    logmel = (logmel - mean) / std

    return logmel.astype(np.float32)

def extract_feature(path):
    audio = load_audio(path)
    spec = audio_to_logmel(audio)
    return spec
