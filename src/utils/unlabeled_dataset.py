# -*- coding: utf-8 -*-

"""Dataset class for unlabeled data."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledDataset(Dataset):
    """Unlabeled dataset for BYOL pretraining."""

    _EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        data_path: str,
        transform: transforms.Compose = None,
        image_size: int = 151,
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.image_size = image_size
        self.paths = self._list_paths()
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found under data_path={self.data_path!r}. "
                "Update config.yaml -> data_params.data_path."
            )

    def _list_paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in Path(f"{self.data_path}").glob("**/*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._EXTS:
                continue
            paths.append(path)
        paths.sort()
        return paths

    def augment(self, image: Image) -> torch.Tensor:
        augmentations = transforms.Compose(
            [
                transforms.Pad((0, 0, 20, 20), fill=0),
                transforms.CenterCrop(self.image_size),
                transforms.RandomRotation(
                    360,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    expand=False,
                ),
                transforms.ToTensor(),
            ]
        )

        return augmentations(image)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as img:
            img = img.resize((self.image_size, self.image_size))
            x = self.transform(img) if self.transform else self.augment(img)
        return x

    def __len__(self) -> int:
        return len(self.paths)

