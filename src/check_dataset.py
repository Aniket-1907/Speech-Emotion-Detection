"""
Scan RAVDESS, TESS, and Indian English speech datasets.

Defaults are resolved relative to the project root (the parent of src/),
not relative to the terminal's current working directory.

Run from the project root:
    python -m src.check_dataset

Optional:
    python -m src.check_dataset --data_dir data
    python -m src.check_dataset --show 30
    python -m src.check_dataset --manifest data/manifest.csv

Creates a manifest CSV with:
    id, emotion, file, dataset, group

The manifest stores file paths relative to the project root so it remains
portable if the project folder is moved.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

LABEL_MAP = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgust",
    "surprise": "surprised",
    "surprised": "surprised",
    "ps": "surprised",  # TESS: pleasant surprise
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def to_project_relative(path: Path) -> str:
    """Return a portable path relative to project root when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Custom external data directories cannot be represented as a path
        # relative to the project root; keep the user-supplied location.
        return str(path)


def resolve_from_project(path_value: str | None) -> Path:
    """Resolve relative CLI paths from project root, not current directory."""
    if path_value is None:
        return DEFAULT_DATA_DIR

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def dataset_name(path: Path, root: Path) -> str:
    relative_parts = [part.lower() for part in path.relative_to(root).parts]

    if any("ravdess" in part for part in relative_parts):
        return "ravdess"

    if any("tess" in part for part in relative_parts):
        return "tess"

    if path.stem.lower().startswith(("oaf_", "yaf_")):
        return "tess"

    if any("indian" in part or "mendeley" in part for part in relative_parts):
        return "indian_english"

    return relative_parts[0] if len(relative_parts) > 1 else "unknown"


def infer_emotion_folder(path: Path, root: Path):
    """Find a supported emotion label in a parent folder below data root."""
    for parent in path.parents:
        if parent == root.parent:
            break

        label = LABEL_MAP.get(parent.name.strip().lower())
        if label:
            return label

        if parent == root:
            break

    return None


def parse_ravdess(path: Path):
    parts = path.stem.split("-")

    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None

    modality, channel, emotion_id, _intensity, _statement, _repetition, actor = (
        map(int, parts)
    )

    # Keep audio-only speech: modality 03, vocal channel 01.
    if modality != 3 or channel != 1:
        return None

    emotion = RAVDESS_EMOTIONS.get(emotion_id)
    if emotion is None:
        return None

    return emotion, f"ravdess_actor_{actor:02d}"


def parse_tess(path: Path):
    stem = path.stem.lower()

    if not stem.startswith(("oaf_", "yaf_")):
        return None

    label = LABEL_MAP.get(stem.rsplit("_", 1)[-1])
    if not label:
        return None

    speaker = stem.split("_", 1)[0].upper()
    return label, f"tess_{speaker}"


def infer_group(path: Path, root: Path, source: str):
    if source == "ravdess":
        parsed = parse_ravdess(path)
        if parsed:
            return parsed[1]

    if source == "tess":
        parsed = parse_tess(path)
        if parsed:
            return parsed[1]

    # In the shown Indian English structure, Original_dataset/Angry/001.wav
    # and Augmented/Angry/001.wav may be related versions. Group by emotion
    # and basename so matching numbered variants remain in the same split.
    if source == "indian_english":
        emotion = infer_emotion_folder(path, root) or "unknown"
        return f"indian_{emotion}_{path.stem.lower()}"

    # Conservative fallback; inspect these groups if filenames encode
    # augmentation relationships in a different way.
    return f"{source}_{path.stem.lower()}"


def scan_dataset(root: Path):
    records = []
    skipped = Counter()
    source_counts = Counter()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        source = dataset_name(path, root)
        emotion = None
        group = None

        if source == "ravdess":
            parsed = parse_ravdess(path)
            if parsed:
                emotion, group = parsed
            else:
                skipped["RAVDESS files not audio-only speech"] += 1
                continue

        elif source == "tess":
            parsed = parse_tess(path)
            if parsed:
                emotion, group = parsed
            else:
                skipped["TESS filename not recognized"] += 1
                continue

        else:
            emotion = infer_emotion_folder(path, root)
            if not emotion:
                skipped["No supported emotion folder found"] += 1
                continue
            group = infer_group(path, root, source)

        records.append({
            "id": 0,
            "emotion": emotion,
            "file": to_project_relative(path),
            "dataset": source,
            "group": group,
        })
        source_counts[source] += 1

    records.sort(key=lambda row: (
        row["dataset"], row["emotion"], row["file"].lower()
    ))

    for index, record in enumerate(records, start=1):
        record["id"] = index

    return records, source_counts, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Scan speech emotion datasets and create a portable manifest."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Dataset directory; relative paths resolve from project root (default: data/).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output CSV path; relative paths resolve from project root (default: data/manifest.csv).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of manifest rows to print (default: 20).",
    )
    args = parser.parse_args()

    root = resolve_from_project(args.data_dir)

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {to_project_relative(root)}")

    records, source_counts, skipped = scan_dataset(root)

    if not records:
        raise SystemExit(
            "No supported audio files found. Check the data folder structure "
            "and the dataset filename/folder conventions."
        )

    if args.manifest:
        manifest_path = resolve_from_project(args.manifest)
    else:
        manifest_path = root / "manifest.csv"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["id", "emotion", "file", "dataset", "group"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("\n=== Dataset loading summary ===")

    for source in ("ravdess", "tess", "indian_english"):
        count = source_counts.get(source, 0)
        if count:
            print(f"{source.upper()} dataset loaded: {count} audio files")
        else:
            print(f"{source.upper()} dataset: no files recognized")

    other_sources = sorted(set(source_counts) - {
        "ravdess", "tess", "indian_english"
    })
    for source in other_sources:
        print(f"{source.upper()} dataset loaded: {source_counts[source]} audio files")

    print(f"\nTOTAL audio files available for training: {len(records)}")
    print("\nEmotion counts:")

    for emotion, count in sorted(
        Counter(row["emotion"] for row in records).items()
    ):
        print(f"  {emotion:10s}: {count}")

    print(f"\nManifest created: {to_project_relative(manifest_path)}")
    print("Columns: id, emotion, file, dataset, group")
    print(f"\nFirst {min(max(args.show, 0), len(records))} manifest rows:")
    print(f"{'ID':>5}  {'EMOTION':<10} {'DATASET':<15} FILE")
    print("-" * 110)

    for row in records[:max(args.show, 0)]:
        print(
            f"{row['id']:>5}  {row['emotion']:<10} "
            f"{row['dataset']:<15} {row['file']}"
        )

    if skipped:
        print("\nSkipped audio files:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    print(
        "\nNext step: make src/data.py load data/manifest.csv so the training "
        "Dataset uses these validated file paths and emotion labels."
    )


if __name__ == "__main__":
    main()
