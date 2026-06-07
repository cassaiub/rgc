import os
from typing import Optional
import torch
import torchvision
import numpy as np
import albumentations
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

def _grayscale_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


class Bent(datasets.ImageFolder):
    """ImageFolder subclass that loads images as single-channel grayscale."""

    def __init__(
        self,
        root: str,
        transform: Optional[torchvision.transforms.Compose] = None,
        target_transform: Optional[torchvision.transforms.Compose] = None,
    ) -> None:
        super().__init__(
            root=os.path.expanduser(root),
            transform=transform,
            target_transform=target_transform,
            loader=_grayscale_loader,
        )

    def __repr__(self) -> str:
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        return fmt_str

class Transforms:
    def __init__(self, transforms: albumentations.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]

def _wrap_transform(t):
    # GalaxyDataset was originally written for albumentations. For finetuning,
    # allow torchvision-style callables too (PIL -> Tensor).
    if t is None:
        return None
    if isinstance(t, albumentations.Compose):
        return Transforms(transforms=t)
    return t


class GalaxyDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        transform=None,
        test_transform=None,
        train_indices=None,
        test_indices=None,
    ):
        super(GalaxyDataset, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.test_transform = test_transform
        self.train_indices = train_indices
        self.test_indices = test_indices

    def setup(self, stage=None):
        entire_dataset_train = Bent(
            root=self.data_dir,
            transform=_wrap_transform(self.transform),
            target_transform=None,
        )

        entire_dataset_test = Bent(
            root=self.data_dir,
            transform=_wrap_transform(self.test_transform),
            target_transform=None,
        )

        self.train_dataset = torch.utils.data.Subset(entire_dataset_train, self.train_indices)
        self.test_dataset = torch.utils.data.Subset(entire_dataset_test, self.test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )