import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .data import load_records, split_by_actor, EMOTIONS
from .dataset import RAVDESSDataset
from .model import EmotionCNN

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    return (
        accuracy_score(ys, preds),
        f1_score(ys, preds, average="macro"),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    records = load_records(args.data_dir)
    train_records, val_records, test_records = split_by_actor(records)

    labels = sorted(EMOTIONS.values())
    label_to_id = {label: i for i, label in enumerate(labels)}

    print(f"Total files: {len(records)}")
    print(f"Train: {len(train_records)}")
    print(f"Validation: {len(val_records)}")
    print(f"Test: {len(test_records)}")

    train_ds = RAVDESSDataset(train_records, label_to_id, augment=True)
    val_ds = RAVDESSDataset(val_records, label_to_id)
    test_ds = RAVDESSDataset(test_records, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    model = EmotionCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    best_f1 = -1.0
    patience = 10
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_f1)

        avg_loss = running_loss / len(train_ds)
        print(
            f"loss={avg_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improvement = 0

            torch.save({
                "model_state": model.state_dict(),
                "labels": labels,
                "sample_rate": 16000,
                "duration": 3.0,
            }, artifacts / "best_model.pt")
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping.")
            break

    # Final test evaluation uses the best validation checkpoint.
    checkpoint = torch.load(
        artifacts / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])

    test_acc, test_f1 = evaluate(model, test_loader, device)

    print(f"\nTEST accuracy: {test_acc:.4f}")
    print(f"TEST macro-F1: {test_f1:.4f}")

    with open(artifacts / "metadata.json", "w") as f:
        json.dump({
            "labels": labels,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "train_actors": list(range(1, 17)),
            "validation_actors": list(range(17, 21)),
            "test_actors": list(range(21, 25)),
        }, f, indent=2)

if __name__ == "__main__":
    main()
