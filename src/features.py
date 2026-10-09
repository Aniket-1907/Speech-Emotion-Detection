"""Audio loading and log-Mel feature extraction."""
import librosa
import numpy as np

SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_MELS = 64

def load_audio(path, sample_rate=SAMPLE_RATE, duration=DURATION_SECONDS):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    # Remove DC offset and normalize peak amplitude without changing labels.
    audio = audio.astype(np.float32)
    audio -= float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio /= peak

    target_len = int(sample_rate * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        # Fixed-length baseline: center crop. Whole-recording inference can be
        # added later for longer real-world recordings.
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    return audio

def audio_to_logmel(audio, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        fmin=20,
        fmax=sample_rate // 2,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    mean = float(logmel.mean())
    std = float(logmel.std())
    return ((logmel - mean) / (std + 1e-6)).astype(np.float32)

def extract_feature(path):
    return audio_to_logmel(load_audio(path))
