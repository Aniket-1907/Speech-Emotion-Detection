import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_records, split_by_actor

records = load_records(sys.argv[1] if len(sys.argv) > 1 else "data")
train, val, test = split_by_actor(records)

print("Total:", len(records))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Emotion distribution:")
print(Counter(r.emotion for r in records))
print("Actors:", sorted(set(r.actor for r in records)))
