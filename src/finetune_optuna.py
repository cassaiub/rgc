# %%
from __future__ import annotations

from pathlib import Path
import os

# Project root
PROJECT_ROOT = Path.cwd()

# WandB directories
wandb_runs = PROJECT_ROOT / "wandb_runs"
wandb_cache = wandb_runs / ".cache"
wandb_artifacts = PROJECT_ROOT / "wandb_artifacts"
WANDB_DIR = PROJECT_ROOT / "wandb"
# Make sure folders exist
wandb_runs.mkdir(parents=True, exist_ok=True)
wandb_cache.mkdir(parents=True, exist_ok=True)
wandb_artifacts.mkdir(parents=True, exist_ok=True)
WANDB_DIR.mkdir(parents=True, exist_ok=True)

# Environment variables — must be before importing wandb!
os.environ["WANDB_DIR"] = str(wandb_runs)
os.environ["WANDB_CACHE_DIR"] = str(wandb_cache)
os.environ["WANDB_ARTIFACTS_DIR"] = str(wandb_artifacts)
os.environ["WANDB_DATA_DIR"] = str(WANDB_DIR)
# Now safe to import
import wandb
from pytorch_lightning.loggers import WandbLogger


# %%

import os
from typing import Optional

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from e2cnn import gspaces
from e2cnn import nn as e2nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchvision import datasets
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# %%
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

# %%
class Transforms:
    def __init__(self, transforms: albumentations.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


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
            transform=self.transform,
            target_transform=None,
        )

        entire_dataset_test = Bent(
            root=self.data_dir,
            transform=self.test_transform,
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

# %%
BYOL_CHECKPOINT = os.path.join(PROJECT_ROOT, "byol", "best.pt")
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "no_spurious_masked_png")
CLASS_NAMES = ["nat", "sfri", "sfrii", "wat"]
NUM_CLASSES = len(CLASS_NAMES)

# %%
class DSteerableLeNet(nn.Module):
    """
    Steerable CNN encoder for radio galaxy images.

    Architecture matches the BYOL checkpoint so weights load cleanly.
    """

    def __init__(self, imsize: int = 151, kernel_size: int = 5, N: int = 16) -> None:
        super().__init__()
        self.imsize = imsize
        self.kernel_size = kernel_size
        self.N = N

        z = 0.5 * (self.imsize - 2)
        z = int(0.5 * (z - 2))
        self._z = z

        self.r2_act = gspaces.FlipRot2dOnR2(self.N)

        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type = e2nn.FieldType(self.r2_act, 6 * [self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, self.imsize, margin=1)
        self.conv1 = e2nn.R2Conv(
            in_type, out_type, kernel_size=self.kernel_size, padding=1, bias=False,
        )
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        self.drop1 = e2nn.PointwiseDropout(out_type, p=0.5)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(
            in_type, out_type, kernel_size=self.kernel_size, padding=1, bias=False,
        )
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        self.drop2 = e2nn.PointwiseDropout(out_type, p=0.5)

        self.gpool = e2nn.GroupPooling(out_type)

        self.fc = nn.Linear(16 * z * z, 2048)
        self.dummy = nn.Parameter(torch.empty(0))

    @property
    def feature_dim(self) -> int:
        return 16 * self._z * self._z

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Conv backbone up to flattened features -> [B, feature_dim]."""
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.relu2(self.conv2(x))))
        x = self.gpool(x)
        x = x.tensor
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))

# %%
class BYOLDownstreamClassifier(pl.LightningModule):

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained_path: str | None = None,
        freeze_encoder: bool = False,
        image_size: int = 151,
        kernel_size: int = 5,
        N: int = 16,
        normalize_features: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = DSteerableLeNet(
            imsize=image_size,
            kernel_size=kernel_size,
            N=N
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded BYOL weights from {pretrained_path}")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False


        feat_dim = self.encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 84),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(84, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

        # ---------------------------
        # Metrics
        # ---------------------------

        # train
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # validation
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

        # test
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.confusion_matrix_metric = ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes
        )

        self.test_outputs = []

    def forward(self, x):

        features = self.encoder.forward_features(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        acc = self.train_accuracy(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # --------------------------------
    # Validation
    # --------------------------------

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        acc = self.val_accuracy(y_hat, y)
        prec = self.val_precision(y_hat, y)
        rec = self.val_recall(y_hat, y)
        f1 = self.val_f1(y_hat, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/precision", prec)
        self.log("val/recall", rec)
        self.log("val/f1_score", f1)

        return loss

    # --------------------------------
    # Test
    # --------------------------------

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.test_outputs.append(
            {"y_hat": y_hat.detach(), "y": y.detach()}
        )

        return loss

    def on_test_epoch_end(self):

        y_hats = torch.cat([o["y_hat"] for o in self.test_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self.test_outputs], dim=0)

        acc = self.test_accuracy(y_hats, y_true)
        precision = self.test_precision(y_hats, y_true)
        recall = self.test_recall(y_hats, y_true)
        f1 = self.test_f1(y_hats, y_true)

        self.log("test/acc", acc)
        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/f1", f1)

        y_true_np = y_true.cpu().numpy()
        y_pred_np = torch.argmax(y_hats, dim=1).cpu().numpy()

        report = classification_report(
            y_true_np,
            y_pred_np,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        self.logger.experiment.log(
            {"classification_report": wandb.Html(f"<pre>{report}</pre>")}
        )

        self.logger.experiment.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_np,
                    preds=y_pred_np,
                    class_names=CLASS_NAMES,
                )
            }
        )

        self.test_outputs.clear()

    # --------------------------------
    # Optimizer
    # --------------------------------

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=5,
            mode="min",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
# %%

def objective(trial):
    hparams = {
        "model_name": "byol_dsteerable_downstream",
        "learning_rate": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        "num_workers": 8,
        "max_epochs": 100,
        "freeze_encoder": False,
        "image_size": 151,
        "train_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((150, 150)),

                torchvision.transforms.RandomRotation(180),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),

                torchvision.transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),

                torchvision.transforms.ToTensor(),

                torchvision.transforms.Pad(padding=1, fill=0, padding_mode="constant"),

                torchvision.transforms.Normalize(
                    mean=[0.0032],
                    std=[0.0376],
                ),
            ]
        ),
        "test_transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((150, 150)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Pad(padding=1, fill=0, padding_mode="constant"),
                torchvision.transforms.Normalize(
                    mean=[0.0032],
                    std=[0.0376],
                ),
            ]
        )

    }

    entire_dataset = Bent(root=DATA_DIR)
    targets = [t for _, t in entire_dataset.samples]
    indices = np.arange(len(targets))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=targets,
        random_state=42,
        )


    data_module = GalaxyDataset(
        data_dir=DATA_DIR,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        transform=hparams["train_transform"],
        test_transform=hparams["test_transform"],
        train_indices=train_idx,
        test_indices=test_idx,
    )

    model = BYOLDownstreamClassifier(
        num_classes=NUM_CLASSES,
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        pretrained_path=BYOL_CHECKPOINT,
        freeze_encoder=hparams["freeze_encoder"],
        image_size=hparams["image_size"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        callbacks=[lr_monitor],
        accelerator="auto",
        log_every_n_steps=1,
        default_root_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
    )

    trainer.fit(model, datamodule=data_module)

    val_results = trainer.validate(model, datamodule=data_module)
    val_f1 = val_results[0]["val/f1_score"]

    print(f"Trial {trial.number} F1: {val_f1:.4f}")

    torch.cuda.empty_cache()

    return val_f1


study = optuna.create_study(
    direction="maximize",
    study_name="byol_single_optuna_1",
    storage="sqlite:///byol_single_split_1.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
pl.seed_everything(42)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# %%
study.best_params

