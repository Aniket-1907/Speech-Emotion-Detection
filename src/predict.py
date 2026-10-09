"""Predict emotion from an audio file using the saved model."""
import argparse
import numpy as np
import torch

from .data import PROJECT_ROOT, resolve_project_path
from .features import extract_feature
from .model import EmotionCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Audio path; relative paths use project root")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    args = parser.parse_args()

    audio_path = resolve_project_path(args.audio)
    checkpoint_path = resolve_project_path(args.checkpoint)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    model = EmotionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature = extract_feature(str(audio_path))
    x = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    print(f"Audio: {audio_path.name}")
    print("Predictions:")
    for index in np.argsort(probabilities)[::-1]:
        print(f"  {labels[index]:12s} {probabilities[index]:.4f}")
    best = int(np.argmax(probabilities))
    print(f"\nPredicted emotion: {labels[best]}")
    print(f"Top softmax score (not calibrated confidence): {probabilities[best]:.4f}")

if __name__ == "__main__":
    main()
