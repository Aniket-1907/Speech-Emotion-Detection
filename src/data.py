from pathlib import Path
from dataclasses import dataclass
import random

EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

@dataclass(frozen=True)
class RAVDESSRecord:
    path: str
    modality: int
    channel: int
    emotion_id: int
    emotion: str
    intensity: int
    statement: int
    repetition: int
    actor: int

def parse_filename(path: Path) -> RAVDESSRecord:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename: {path.name}")

    modality, channel, emotion, intensity, statement, repetition, actor = map(int, parts)

    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion {emotion}: {path.name}")

    return RAVDESSRecord(
        path=str(path),
        modality=modality,
        channel=channel,
        emotion_id=emotion,
        emotion=EMOTIONS[emotion],
        intensity=intensity,
        statement=statement,
        repetition=repetition,
        actor=actor,
    )

def load_records(data_dir: str):
    data_path = Path(data_dir)
    records = []

    for wav in sorted(data_path.rglob("*.wav")):
        try:
            record = parse_filename(wav)
        except ValueError:
            continue

        # Audio-only speech:
        # modality 03 = audio-only
        # channel 01 = speech
        if record.modality == 3 and record.channel == 1:
            records.append(record)

    if not records:
        raise RuntimeError(
            f"No RAVDESS audio-only speech files found under {data_dir}. "
            "Expected files beginning with 03-01-."
        )

    return records

def split_by_actor(records):
    # Speaker-independent split.
    train_actors = set(range(1, 17))
    val_actors = set(range(17, 21))
    test_actors = set(range(21, 25))

    train = [r for r in records if r.actor in train_actors]
    val = [r for r in records if r.actor in val_actors]
    test = [r for r in records if r.actor in test_actors]

    return train, val, test
