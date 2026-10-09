"""Train a dataset-agnostic emotion classifier using data/manifest.csv."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .data import PROJECT_ROOT, load_records
from .dataset import AudioEmotionDataset
from .model import EmotionCNN

def split_by_group(records, seed=42):
    groups = np.asarray([r.group for r in records])
    indices = np.arange(len(records))
    if len(set(groups.tolist())) < 3:
        raise ValueError(
            "Need at least 3 distinct groups for train/validation/test splitting. "
            "Check the manifest 'group' column."
        )

    split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, hold_idx = next(split1.split(indices, groups=groups))
    hold_groups = groups[hold_idx]
    hold_indices = np.arange(len(hold_idx))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_local, test_local = next(split2.split(hold_indices, groups=hold_groups))

    train = [records[i] for i in train_idx]
    hold = [records[i] for i in hold_idx]
    val = [hold[i] for i in val_local]
    test = [hold[i] for i in test_local]

    sets = [{r.group for r in part} for part in (train, val, test)]
    if not (sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2])
            and sets[1].isdisjoint(sets[2])):
        raise RuntimeError("Group leakage detected across data splits.")
    return train, val, test

def evaluate(model, loader, device, labels):
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    accuracy = accuracy_score(truth, predictions) if truth else 0.0
    macro_f1 = f1_score(
        truth, predictions, labels=list(range(len(labels))),
        average="macro", zero_division=0
    ) if truth else 0.0
    return accuracy, macro_f1, truth, predictions

def summarize(name, records, labels):
    print(f"\n{name}: {len(records)} files, {len({r.group for r in records})} groups")
    print("  datasets:", dict(Counter(r.source for r in records)))
    counts = Counter(r.emotion for r in records)
    print("  emotions:", {label: counts.get(label, 0) for label in labels})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None,
                        help="Manifest path; default is project_root/data/manifest.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    records = load_records(args.manifest)
    labels = sorted({r.emotion for r in records})
    if len(labels) < 2:
        raise ValueError(f"At least two emotion classes are needed; found {labels}")
    label_to_id = {label: i for i, label in enumerate(labels)}
    print("Classes from manifest:", labels)

    train_records, val_records, test_records = split_by_group(records)
    summarize("TRAIN", train_records, labels)
    summarize("VALIDATION", val_records, labels)
    summarize("TEST", test_records, labels)

    train_ds = AudioEmotionDataset(train_records, label_to_id, augment=True)
    val_ds = AudioEmotionDataset(val_records, label_to_id, augment=False)
    test_ds = AudioEmotionDataset(test_records, label_to_id, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    counts = Counter(r.emotion for r in train_records)
    weights = [
        len(train_records) / (len(labels) * counts[label]) if counts.get(label, 0) else 0.0
        for label in labels
    ]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    model = EmotionCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "best_model.pt"
    best_f1, stale_epochs = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device, labels)
        scheduler.step(val_f1)
        print(
            f"loss={total_loss / max(1, len(train_ds)):.4f} "
            f"val_accuracy={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, stale_epochs = val_f1, 0
            torch.save({
                "model_state": model.state_dict(),
                "labels": labels,
                "sample_rate": 16000,
                "duration": 4.0,
            }, checkpoint_path)
            print("Saved best checkpoint:", checkpoint_path.relative_to(PROJECT_ROOT))
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_acc, test_f1, truth, predictions = evaluate(model, test_loader, device, labels)
    print("\n=== Held-out test results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print(classification_report(
        truth, predictions, labels=list(range(len(labels))),
        target_names=labels, digits=4, zero_division=0
    ))

    metadata = {
        "labels": labels,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "record_count": len(records),
        "datasets": sorted({r.source for r in records}),
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved metadata:", metadata_path.relative_to(PROJECT_ROOT))

if __name__ == "__main__":
    main()
