"""Evaluate the current checkpoint using a fresh group split from the manifest."""
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

from .data import load_records, resolve_project_path
from .train import split_by_group
from .dataset import AudioEmotionDataset
from .model import EmotionCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    records = load_records(args.manifest)
    _, _, test_records = split_by_group(records)
    checkpoint_path = resolve_project_path(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    test_records = [r for r in test_records if r.emotion in label_to_id]
    loader = DataLoader(
        AudioEmotionDataset(test_records, label_to_id),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = EmotionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    if not truth:
        raise RuntimeError("Test split is empty.")
    print(f"Accuracy: {accuracy_score(truth, pred):.4f}")
    print(f"Macro-F1: {f1_score(truth, pred, average='macro', zero_division=0):.4f}")
    print(classification_report(
        truth, pred, labels=list(range(len(labels))),
        target_names=labels, digits=4, zero_division=0
    ))

if __name__ == "__main__":
    main()
