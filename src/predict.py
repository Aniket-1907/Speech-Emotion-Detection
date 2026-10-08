import argparse
import torch
import numpy as np

from .model import EmotionCNN
from .features import extract_feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    labels = checkpoint["labels"]

    model = EmotionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    spec = extract_feature(args.audio)
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    order = np.argsort(probs)[::-1]
    top = order[0]

    print(f"Predicted emotion: {labels[top]}")
    print(f"Confidence: {probs[top]:.4f}")
    print("\nProbabilities:")

    for i in order:
        print(f"{labels[i]:10s} {probs[i]:.4f}")

if __name__ == "__main__":
    main()
