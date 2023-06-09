"""
Diffusion Transformer - Dataset
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class DiffusionTransformerDataset(Dataset):
    """Dataset for diffusion transformer."""

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        split_file = os.path.join(self.root, f"{self.split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        samples.append((parts[0], int(parts[1])))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


def get_transforms(split, img_size=224):
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_dataloader(root, split, config):
    transform = get_transforms(split, config.img_size)
    dataset = DiffusionTransformerDataset(root, split, transform)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=True
    )

# update 1
