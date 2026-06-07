# -*- coding: utf-8 -*-

"""BYOL Trainer."""

import os

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from models.byol_pytorch import BYOL
from models.dstreeablelenet import DSteerableLeNet
from utils.unlabeled_dataset import UnlabeledDataset

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


class BYOLTrainer:
    """BYOL Trainer."""

    def __init__(
        self,
        image_size: int = 151,
        kernel_size: int = 5,
        N: int = 16,
        lr: float = 0.001,
        batch_size: int = 128,
        num_workers: int = 4,
        num_epochs: int = 10,
        device: str = "cuda",
        projection_size: int = 256,
        projection_hidden_size: int = 4096,
        moving_average_decay: float = 0.99,
        data_path: str = "data",
        results_folder: str = "results",
        wandb_cfg: dict | None = None,
    ) -> None:
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.N = N
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.moving_average_decay = moving_average_decay
        self.data_path = data_path
        self.results_folder = results_folder
        self.wandb_cfg = wandb_cfg or {}

        self.wandb_enabled = bool(self.wandb_cfg.get("enabled", False)) and wandb is not None
        self.wandb_log_every_steps = int(self.wandb_cfg.get("log_every_steps", 50))
        self.wandb_log_checkpoints = bool(self.wandb_cfg.get("log_checkpoints", False))
        self._wandb_run = None

        self.model = DSteerableLeNet(
            imsize=self.image_size, kernel_size=self.kernel_size, N=self.N
        ).to(self.device)

        self.learner = BYOL(
            self.model,
            image_size=self.image_size,
            hidden_layer="fc",
            projection_size=self.projection_size,
            projection_hidden_size=self.projection_hidden_size,
            augment_fn=self.augmentation(),
            augment_fn2=self.augmentation(gaussian_blur=True),
            moving_average_decay=self.moving_average_decay,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.lr)

        self.run_version = 0
        while os.path.exists(
            self.results_folder + "/models/byol/run_" + str(self.run_version)
        ):
            self.run_version += 1

        self.model_out_dir = (
            self.results_folder + "/models/byol/run_" + str(self.run_version)
        )

        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)

    def train(self) -> None:
        if self.wandb_enabled:
            self._wandb_run = wandb.init(
                project=str(self.wandb_cfg.get("project", "rgc-byol")),
                entity=self.wandb_cfg.get("entity", None),
                name=self.wandb_cfg.get("name", None),
                tags=self.wandb_cfg.get("tags", None),
                config={
                    "image_size": self.image_size,
                    "kernel_size": self.kernel_size,
                    "N": self.N,
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "num_epochs": self.num_epochs,
                    "device": self.device,
                    "projection_size": self.projection_size,
                    "projection_hidden_size": self.projection_hidden_size,
                    "moving_average_decay": self.moving_average_decay,
                    "data_path": self.data_path,
                    "results_folder": self.results_folder,
                    "run_version": self.run_version,
                },
            )

        train_loader = self.train_dataloader()
        best_loss = 1e10
        global_step = 0
        for epoch in range(self.num_epochs):
            self.learner.train()
            running_loss = 0.0

            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for _, images in loop:
                images = images.to(self.device)

                loss = self.learner(images)
                running_loss += loss.item()

                self.learner.update_moving_average()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loop.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item())

                if self.wandb_enabled and (global_step % self.wandb_log_every_steps == 0):
                    wandb.log(
                        {"train/loss_step": float(loss.item()), "epoch": epoch},
                        step=global_step,
                    )
                global_step += 1

            running_loss /= len(self.train_dataloader())
            if self.wandb_enabled:
                wandb.log({"train/loss_epoch": float(running_loss), "epoch": epoch}, step=global_step)

            self.learner.eval()
            last_path = self.model_out_dir + "/last.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": running_loss,
                },
                last_path,
            )
            if self.wandb_enabled and self.wandb_log_checkpoints:
                wandb.save(last_path)

            if running_loss < best_loss:
                best_loss = running_loss
                best_path = self.model_out_dir + "/best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": running_loss,
                    },
                    best_path,
                )
                if self.wandb_enabled and self.wandb_log_checkpoints:
                    wandb.save(best_path)

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_data = UnlabeledDataset(
            data_path=self.data_path,
            image_size=self.image_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train_loader

    def augmentation(self, gaussian_blur: bool = False) -> transforms.Compose:
        fn = [
            self.RandomApply(
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.8,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=151, scale=(0.3, 1.0), ratio=(0.8, 1.0)),
            transforms.RandomRotation(
                360,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False,
            ),
            transforms.Normalize((0.0033,), (0.0393,)),
        ]

        if gaussian_blur:
            fn.insert(
                2,
                self.RandomApply(
                    transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                    p=0.5,
                ),
            )

        return transforms.Compose(fn)

    class RandomApply(nn.Module):
        def __init__(self, fn: object, p: float):
            super().__init__()
            self.fn = fn
            self.p = p

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if torch.rand(1) > self.p:
                return x
            return self.fn(x)

