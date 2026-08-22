from __future__ import annotations

from pathlib import Path
import random
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover(root: str | Path):
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 class folders under {root}")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples = []
    for name in classes:
        for p in (root / name).rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                samples.append((str(p.relative_to(root)), class_to_idx[name]))
    if not samples:
        raise ValueError(f"No retinal images found below {root}")
    return samples, classes


def stratified_split(samples, seed=42, train_ratio=.70, val_ratio=.15):
    rng = random.Random(seed)
    groups = {}
    for item in samples:
        groups.setdefault(item[1], []).append(item)
    train, val, test = [], [], []
    for items in groups.values():
        rng.shuffle(items)
        n = len(items)
        a = max(1, int(n * train_ratio))
        b = max(a + 1, int(n * (train_ratio + val_ratio))) if n > 2 else n
        train.extend(items[:a]); val.extend(items[a:b]); test.extend(items[b:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    if not val or not test:
        raise ValueError("Dataset is too small for train/validation/test splitting")
    return train, val, test


class RetinaDataset(Dataset):
    def __init__(self, root, samples, image_size=224, train=False):
        self.root = Path(root)
        self.samples = samples
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)]
        ops += [transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel, label = self.samples[index]
        image = Image.open(self.root / rel).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long), rel
