import random
import torch
from torch.utils.data import Dataset
from .features import extract_feature

class RAVDESSDataset(Dataset):
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
            # SpecAugment-style masking.
            spec = spec.copy()

            if random.random() < 0.5:
                width = random.randint(5, min(20, spec.shape[1]))
                start = random.randint(0, spec.shape[1] - width)
                spec[:, start:start + width] = 0

            if random.random() < 0.5:
                height = random.randint(3, min(12, spec.shape[0]))
                start = random.randint(0, spec.shape[0] - height)
                spec[start:start + height, :] = 0

        x = torch.from_numpy(spec).unsqueeze(0)
        y = self.label_to_id[record.emotion]

        return x, torch.tensor(y, dtype=torch.long)
