"""Image datasets, augmentations and models (needs the `vision` extra)."""

from __future__ import annotations

import os

import albumentations as A
import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import Dataset

from ..engine import target_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_paths(names, dirs) -> list[str]:
    """Find each image in the first folder that has it (train and test folders can differ)."""
    dirs = [d for d in dirs if d] or [""]
    out = []
    for name in names:
        name = str(name)
        for d in dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                out.append(path)
                break
        else:
            raise FileNotFoundError(f"image {name} not found in {dirs}")
    return out


def transforms(size: int, train: bool, augment: bool):
    ops = [A.Resize(size, size)]
    if train and augment:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        ]
    ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(ops)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: str) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.float32)


class ImageDataset(Dataset):
    def __init__(self, paths, y, transform, task):
        self.paths = paths
        self.y = None if y is None else target_tensor(task, y)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(image=load_rgb(self.paths[i]))["image"]
        item = {"image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()}
        item["target"] = self.y[i] if self.y is not None else torch.zeros(1)
        return item


class MaskDataset(Dataset):
    def __init__(self, paths, mask_paths, transform):
        self.paths = paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.mask_paths is not None:
            out = self.transform(image=img, mask=load_mask(self.mask_paths[i]))
            target = torch.from_numpy(out["mask"][None].astype(np.float32))
        else:
            out = self.transform(image=img)
            target = torch.zeros(1)
        image = torch.from_numpy(np.ascontiguousarray(out["image"].transpose(2, 0, 1))).float()
        return {"image": image, "target": target}


class ImageClassifier(nn.Module):
    def __init__(self, backbone: str, n_out: int, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        import timm

        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=n_out, drop_rate=drop_rate)

    def forward(self, image):
        return self.net(image)


class SegmentationNet(nn.Module):
    def __init__(self, arch: str, encoder: str, pretrained: bool = True, classes: int = 1):
        super().__init__()
        import segmentation_models_pytorch as smp

        cls = getattr(smp, arch)
        self.net = cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                       in_channels=3, classes=classes, activation=None)

    def forward(self, image):
        return self.net(image)
