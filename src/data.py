"""Generic manifest loader; no dataset-specific filename parsing here."""
import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"

@dataclass(frozen=True)
class Record:
    path: str
    emotion: str
    source: str
    group: str
    record_id: str = ""

def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

def load_records(manifest_path=None):
    manifest = resolve_project_path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            "Run `python -m src.check_dataset` from the project root first."
        )

    required = {"id", "emotion", "file", "dataset", "group"}
    records, missing, invalid = [], [], []

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest}")
        absent = required - set(reader.fieldnames)
        if absent:
            raise ValueError(f"Manifest is missing columns: {sorted(absent)}")

        for line_no, row in enumerate(reader, start=2):
            try:
                record_id = str(row["id"]).strip()
                emotion = str(row["emotion"]).strip().lower()
                file_value = str(row["file"]).strip()
                source = str(row["dataset"]).strip() or "unknown"
                group = str(row["group"]).strip()
                if not all((record_id, emotion, file_value, group)):
                    raise ValueError("empty required field")

                audio_path = resolve_project_path(file_value)
                if not audio_path.is_file():
                    missing.append((line_no, file_value))
                    continue

                records.append(Record(str(audio_path), emotion, source, group, record_id))
            except Exception as exc:
                invalid.append((line_no, str(exc)))

    if missing:
        print(f"Warning: {len(missing)} audio path(s) from the manifest were not found.")
        for line, path in missing[:10]:
            print(f"  CSV line {line}: {path}")
        if len(missing) > 10:
            print("  ...")
    if invalid:
        print(f"Warning: {len(invalid)} invalid manifest row(s) skipped.")
        for line, message in invalid[:10]:
            print(f"  CSV line {line}: {message}")

    # Deduplicate repeated paths so the same recording is not counted twice.
    unique = {}
    for record in records:
        unique.setdefault(Path(record.path).resolve().as_posix().casefold(), record)
    records = list(unique.values())
    if not records:
        raise RuntimeError(f"No usable audio records found in {manifest}")

    print(f"Loaded {len(records)} audio records from {manifest}")
    return records
