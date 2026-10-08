import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from .data import load_records, split_by_actor, EMOTIONS
from .dataset import RAVDESSDataset
from .model import EmotionCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_records(args.data_dir)
    _, _, test_records = split_by_actor(records)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    label_to_id = {x: i for i, x in enumerate(labels)}

    ds = RAVDESSDataset(test_records, label_to_id)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = EmotionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())

    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
