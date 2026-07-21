"""BYOL downstream classifier (LightningModule).

Model definition only. The training/orchestration entry point lives in
`src/scripts/downstream_trainer.py`, which imports `BYOLDownstreamClassifier`
from here.
"""

import os
import re

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from models.dsteerablelenet import DSteerableLeNet


class BYOLDownstreamClassifier(pl.LightningModule):
    """BYOL-pretrained encoder + head; saves .pt outputs for analysis."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        pretrained_path: str | None,
        freeze_encoder: bool,
        image_size: int = 151,
        experiment_root: str = ".",
        class_names_report: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["experiment_root", "class_names_report"])
        self.experiment_root = experiment_root
        self.class_names_report = (
            list(class_names_report) if class_names_report is not None else [str(i) for i in range(num_classes)]
        )

        self.encoder = DSteerableLeNet(imsize=image_size, kernel_size=5, N=16)
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded BYOL weights from {pretrained_path}")
        elif pretrained_path is not None:
            print(
                f"WARNING: BYOL checkpoint not found at {pretrained_path} — training from scratch."
            )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        feat_dim = self.encoder.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.001)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision_metric = Precision(task="multiclass", num_classes=num_classes)
        self.recall_metric = Recall(task="multiclass", num_classes=num_classes)
        self.f1_metric = F1Score(task="multiclass", num_classes=num_classes)
        self.confusion_matrix_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_outputs: list[dict] = []

    def forward(self, x):
        features = self.encoder.forward_features(x)
        features = F.normalize(features, dim=1)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_accuracy(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)
        prec = self.precision_metric(y_hat, y)
        rec = self.recall_metric(y_hat, y)
        f1 = self.f1_metric(y_hat, y)
        self.log("val/loss", loss, on_epoch=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/precision", prec, on_epoch=True, logger=True)
        self.log("val/recall", rec, on_epoch=True, logger=True)
        self.log("val/f1_score", f1, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_outputs.append({"y_hat": y_hat, "y": y})
        return loss

    def on_test_epoch_end(self):
        y_hats = torch.cat([o["y_hat"] for o in self.test_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self.test_outputs], dim=0)
        y_probs = torch.softmax(y_hats.float(), dim=1)
        y_pred = torch.argmax(y_hats, dim=1)

        acc = self.test_accuracy(y_hats, y_true)
        precision = self.precision_metric(y_hats, y_true)
        recall = self.recall_metric(y_hats, y_true)
        f1 = self.f1_metric(y_hats, y_true)
        self.log("test/acc", acc, on_epoch=True, logger=True)
        self.log("test/precision", precision, on_epoch=True, logger=True)
        self.log("test/recall", recall, on_epoch=True, logger=True)
        self.log("test/f1_score", f1, on_epoch=True, logger=True)

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        report = classification_report(
            y_true_np,
            y_pred_np,
            target_names=self.class_names_report,
            zero_division=0,
        )
        self.logger.experiment.log({"classification_report": wandb.Html(f"<pre>{report}</pre>")})
        self.logger.experiment.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_np,
                    preds=y_pred_np,
                    class_names=self.class_names_report,
                )
            }
        )

        datamodule = self.trainer.datamodule
        test_subset = datamodule.test_dataset
        test_paths = [test_subset.dataset.samples[idx][0] for idx in test_subset.indices]

        run_name = getattr(self.logger.experiment, "name", "test_run")
        run_id = getattr(self.logger.experiment, "id", "local")
        # Paths and W&B artifact names: only [a-zA-Z0-9._-] allowed for artifacts
        safe_run_name = re.sub(r"[^a-zA-Z0-9._-]", "_", str(run_name))
        output_dir = os.path.join(self.experiment_root, "saved_test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{safe_run_name}_{run_id}_test_outputs.pt")
        self.last_test_save_path = save_path

        num_classes = y_hats.shape[1]
        save_dict = {
            "logits": y_hats.detach().cpu().float(),
            "probs": y_probs.detach().cpu().float(),
            "y_true": y_true.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "class_names": [str(i) for i in range(num_classes)],
            "test_paths": test_paths,
        }
        torch.save(save_dict, save_path)

        artifact = wandb.Artifact(
            f"test_outputs_{safe_run_name}",
            type="dataset",
            description="BYOL downstream: logits, probs, labels, paths",
        )
        artifact.add_file(save_path)
        self.logger.experiment.log_artifact(artifact)

        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
