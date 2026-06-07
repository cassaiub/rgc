# %%
from __future__ import annotations

import os
from typing import Optional

PROJECT_ROOT = os.getcwd()

os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".torch")
os.environ["WANDB_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs")
os.environ["WANDB_CACHE_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs", ".cache")

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from albumentations.pytorch import ToTensorV2
from e2cnn import gspaces
from e2cnn import nn as e2nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
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
from sklearn.model_selection import StratifiedKFold


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
            transform=Transforms(transforms=self.transform),
            target_transform=None,
        )

        entire_dataset_test = Bent(
            root=self.data_dir,
            transform=Transforms(transforms=self.test_transform),
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
BYOL_CHECKPOINT = os.path.join(PROJECT_ROOT, "src", "rgc_multihead", "byol", "best.pt")
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
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-6,
        pretrained_path: str | None = None,
        freeze_encoder: bool = False,
        image_size: int = 151,
        kernel_size: int = 5,
        N: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Encoder ----
        self.encoder = DSteerableLeNet(
            imsize=image_size,
            kernel_size=kernel_size,
            N=N,
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded BYOL weights from {pretrained_path}")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ---- Classification Head ----
        feat_dim = self.encoder.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

        # ---- Metrics ----
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.test_outputs = []

    def forward(self, x):
        features = self.encoder.forward_features(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.test_outputs.append({
            "logits": logits.detach(),
            "targets": y.detach()
        })

    def on_test_epoch_end(self):

        logits = torch.cat([o["logits"] for o in self.test_outputs])
        targets = torch.cat([o["targets"] for o in self.test_outputs])

        acc = self.test_acc(logits, targets)
        self.log("test/acc", acc)

        preds = torch.argmax(logits, dim=1)

        # Confusion matrix (manual)
        cm = ConfusionMatrix(
            task="multiclass",
            num_classes=self.hparams.num_classes
        ).to(self.device)

        confusion = cm(preds, targets)

        print("\nConfusion Matrix:")
        print(confusion.cpu().numpy())

        self.test_outputs.clear()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
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
entire_dataset = Bent(root=DATA_DIR)
targets = [t for _, t in entire_dataset.samples]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_fold_indices = list(skf.split(range(len(entire_dataset)), targets))

# %%
def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-5, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 32)
    freeze_encoder = trial.suggest_categorical("freeze_encoder", [True, False])

    train_transform = albumentations.Compose([
        albumentations.PadIfNeeded(min_height=152, min_width=152, border_mode=0, value=0),
        albumentations.CenterCrop(height=151, width=151),
        albumentations.HorizontalFlip(),
        albumentations.Rotate(limit=180, border_mode=0, value=0, p=0.5),
        albumentations.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        albumentations.RandomBrightnessContrast(p=0.3),
        albumentations.Normalize(mean=(0.0032,), std=(0.0376,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = albumentations.Compose([
        albumentations.PadIfNeeded(min_height=152, min_width=152, border_mode=0, value=0),
        albumentations.CenterCrop(height=151, width=151),
        albumentations.Normalize(mean=(0.0032,), std=(0.0376,), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    max_epochs = 40
    fold_scores = []
    fold_scores_dict = {}

    print(f"\n🔎 Trial {trial.number} started")

    for fold_idx, (train_indices, val_indices) in enumerate(all_fold_indices):

        print(f"   Fold {fold_idx+1}/5")

        data_module = GalaxyDataset(
            data_dir=DATA_DIR,
            batch_size=batch_size,
            num_workers=4,
            transform=train_transform,
            test_transform=val_transform,
            train_indices=train_indices,
            test_indices=val_indices,
        )

        model = BYOLDownstreamClassifier(
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained_path=BYOL_CHECKPOINT,
            freeze_encoder=freeze_encoder,
            image_size=151,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision="16-mixed",
            deterministic=True,
        )

        data_module.setup()
        trainer.fit(model, datamodule=data_module)

        val_results = trainer.validate(model, datamodule=data_module)
        fold_f1 = val_results[0]["val/f1"]
        fold_acc = val_results[0]["val/acc"]

        # Class-wise F1 and accuracy (per-class, then mean across classes)
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                x, y = batch
                x, y = x.to(model.device), y.to(model.device)
                logits = model(x)
                all_logits.append(logits)
                all_targets.append(y)
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        preds = torch.argmax(logits, dim=1)

        f1_per_class = F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="none"
        ).to(model.device)(preds, targets)
        acc_per_class = Accuracy(
            task="multiclass", num_classes=NUM_CLASSES, average="none"
        ).to(model.device)(preds, targets)

        f1_per_class = f1_per_class.cpu().tolist()
        acc_per_class = acc_per_class.cpu().tolist()
        mean_f1_per_class = sum(f1_per_class) / len(f1_per_class)
        mean_acc_per_class = sum(acc_per_class) / len(acc_per_class)

        fold_scores_dict[fold_idx] = {
            "val/f1_macro": float(fold_f1),
            "val/acc_macro": float(fold_acc),
            "val/mean_f1_per_class": mean_f1_per_class,
            "val/mean_accuracy_per_class": mean_acc_per_class,
            "val/f1_per_class": {CLASS_NAMES[i]: f1_per_class[i] for i in range(NUM_CLASSES)},
            "val/accuracy_per_class": {CLASS_NAMES[i]: acc_per_class[i] for i in range(NUM_CLASSES)},
        }

        print(f"Fold F1: {fold_f1:.4f}  Acc: {fold_acc:.4f}  mean_f1_per_class: {mean_f1_per_class:.4f}  mean_acc_per_class: {mean_acc_per_class:.4f}")

        fold_scores.append(fold_f1)

        # 🔥 report intermediate result for pruning
        trial.report(fold_f1, step=fold_idx)

        if trial.should_prune():
            print("   ❌ Trial pruned")
            raise optuna.TrialPruned()

        torch.cuda.empty_cache()

    mean_score = sum(fold_scores) / len(fold_scores)

    print(f"✅ Trial {trial.number} Mean F1: {mean_score:.4f}")

    return mean_score

# %%
study = optuna.create_study(
    direction="maximize",
    study_name="byol_5fold",
    storage="sqlite:///byol_5fold.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
)
pl.seed_everything(42)
study.optimize(objective, n_trials=50, show_progress_bar=True)

# %%
study.best_params


