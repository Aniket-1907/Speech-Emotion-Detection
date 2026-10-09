"""PyTorch Dataset for manifest records."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .features import extract_feature

class AudioEmotionDataset(Dataset):
    def __init__(self, records, label_to_id, augment=False):
        self.records = records
        self.label_to_id = label_to_id
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        spec = extract_feature(record.path)

        if self.augment:
            spec = spec.copy()
            # Simple SpecAugment masks on the time and frequency axes.
            if random.random() < 0.5 and spec.shape[1] > 5:
                width = random.randint(1, min(16, spec.shape[1] - 1))
                start = random.randint(0, spec.shape[1] - width)
                spec[:, start:start + width] = 0.0
            if random.random() < 0.5 and spec.shape[0] > 3:
                height = random.randint(1, min(10, spec.shape[0] - 1))
                start = random.randint(0, spec.shape[0] - height)
                spec[start:start + height, :] = 0.0

        if record.emotion not in self.label_to_id:
            raise ValueError(f"Unknown label '{record.emotion}' for {record.path}")

        x = torch.from_numpy(np.asarray(spec, dtype=np.float32)).unsqueeze(0)
        y = torch.tensor(self.label_to_id[record.emotion], dtype=torch.long)
        return x, y
