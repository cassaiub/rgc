import argparse
import datetime as _dt
import json
import os
import re
from pathlib import Path
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from torchvision import transforms as T

if __package__ is None or __package__ == "":
    # Allow running as: python3 models/downstream_trainer.py
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from models.dstreeablelenet import DSteerableLeNet
from utils.dataloader import Bent, GalaxyDataset


PROJECT_ROOT = os.environ.get("RGC_PROJECT_ROOT", os.getcwd())
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".torch")
os.environ["WANDB_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs")
os.environ["WANDB_CACHE_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs", ".cache")

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


def _resolve_path(root: str, p: str) -> str:
    if p is None:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(root, p))


def _wandb_entity_and_project(finetune_cfg: dict) -> tuple[str | None, str]:
    """W&B entity (team or user) must exist at wandb.ai; omit or null for your default logged-in scope.

    Project names cannot contain /. If `wandb_entity` is unset, `/` in `wandb_project` is folded into the
    project slug (e.g. `a/b` -> `a-b`) instead of guessing an entity — avoiding 404 when the prefix is only a label.
    """
    raw_ent = finetune_cfg.get("wandb_entity")
    entity: str | None = None
    if raw_ent is not None:
        ent_s = str(raw_ent).strip()
        if ent_s and ent_s.lower() not in ("none", "null"):
            entity = ent_s

    raw_proj = str(finetune_cfg.get("wandb_project", "rgc-downstream")).strip()
    project_name = raw_proj.replace("/", "-").replace("\\", "-")

    if entity is None and not project_name:
        project_name = "rgc-downstream"

    return (entity, project_name)


def _load_finetune_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "finetune" not in raw:
        raise KeyError("config.yaml must contain a top-level 'finetune' section.")
    return raw["finetune"]


def main(config_path: str = "config.yaml") -> None:
    cfg = _load_finetune_cfg(config_path)

    project_root = os.environ.get("RGC_PROJECT_ROOT", os.getcwd())
    experiment_root = _resolve_path(project_root, str(cfg.get("experiment_root", "finetune_runs")))
    os.makedirs(experiment_root, exist_ok=True)

    base_data_dir = _resolve_path(project_root, str(cfg["data_dir"]))
    if not os.path.isdir(base_data_dir):
        raise FileNotFoundError(
            f"finetune.data_dir does not exist: {base_data_dir!r}. "
            "Update config.yaml -> finetune.data_dir."
        )
    byol_checkpoint = cfg.get("byol_checkpoint", None)
    if byol_checkpoint is not None:
        byol_checkpoint = _resolve_path(project_root, str(byol_checkpoint))

    k_fold = int(cfg.get("k_fold", 1))
    test_size = float(cfg.get("test_size", 0.2))
    split_seed = int(cfg.get("split_seed", 42))

    model_slug = str(cfg.get("model_slug", "byol_downstream"))

    train_transform = T.Compose(
        [
            T.Resize((150, 150)),
            T.RandomRotation(90),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Pad(padding=1, fill=0, padding_mode="constant"),
            T.Normalize(mean=[0.0032], std=[0.0376]),
        ]
    )
    test_transform = T.Compose(
        [
            T.Resize((150, 150)),
            T.ToTensor(),
            T.Pad(padding=1, fill=0, padding_mode="constant"),
            T.Normalize(mean=[0.0032], std=[0.0376]),
        ]
    )

    wandb.login()

    # Single ImageFolder root (class folders inside)
    folder = "single_dataset"
    dataset_variant = "dataset"
    current_data_dir = base_data_dir

    entire_dataset = Bent(root=current_data_dir)
    if len(entire_dataset) == 0:
        raise FileNotFoundError(f"No images found under {current_data_dir!r}")

    num_classes = len(entire_dataset.classes)
    class_names_report = list(entire_dataset.classes)
    targets = [t for _, t in entire_dataset.samples]
    indices = np.arange(len(targets))
    targets_np = np.asarray(targets, dtype=np.int64)

    fold_splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    if k_fold and k_fold > 1:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=split_seed)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(indices, targets_np)):
            # skf.split returns indices into the provided arrays; convert to dataset indices
            fold_splits.append((fold_i, indices[train_idx], indices[test_idx]))
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=targets_np,
            random_state=split_seed,
        )
        fold_splits.append((0, train_idx, test_idx))

    # ---- k-fold aggregation ----
    fold_reports: list[dict] = []
    fold_confusions: list[np.ndarray] = []
    fold_run_names: list[str] = []

    log_dir = os.path.join(experiment_root, "kfold_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"kfold_{Path(current_data_dir).name}_{model_slug}_{log_stamp}.jsonl")

    for fold_i, train_idx, test_idx in fold_splits:
        run_name = (
            f"finetune_{Path(current_data_dir).name}_{model_slug}_fold{fold_i+1}of{len(fold_splits)}"
            if len(fold_splits) > 1
            else f"finetune_{Path(current_data_dir).name}_{model_slug}_1fold"
        )
        fold_run_names.append(run_name)

        run_hparams = {
            **cfg,
            "model_name": model_slug,
            "data_dir": current_data_dir,
            "subdataset_folder": folder,
            "dataset_variant": dataset_variant,
            "run_name": run_name,
            "byol_checkpoint": byol_checkpoint,
            "experiment_root": experiment_root,
            "k_fold": k_fold,
            "fold_index": int(fold_i),
        }

        print(f"\n{'#' * 50}")
        print(f"# EXPERIMENT: BYOL downstream finetune")
        print(f"# MODEL: {model_slug}")
        print(f"# DATA DIR: {current_data_dir}")
        print(f"# RUN: {run_name}")
        print(f"{'#' * 50}")

        data_module = GalaxyDataset(
            data_dir=current_data_dir,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            transform=train_transform,
            test_transform=test_transform,
            train_indices=train_idx,
            test_indices=test_idx,
        )

        model = BYOLDownstreamClassifier(
            num_classes=num_classes,
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            pretrained_path=byol_checkpoint,
            freeze_encoder=bool(cfg["freeze_encoder"]),
            image_size=int(cfg.get("image_size", 151)),
            experiment_root=experiment_root,
            class_names_report=class_names_report,
        )

        wb_entity, wb_project = _wandb_entity_and_project(cfg)
        wandb_logger = WandbLogger(
            name=run_name,
            entity=wb_entity,
            project=wb_project,
            log_model=False,
            save_code=True,
        )
        wandb_logger.experiment.config.update(run_hparams)

        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=int(cfg["max_epochs"]),
            logger=wandb_logger,
            callbacks=[lr_monitor],
            accelerator=str(cfg.get("accelerator", "gpu")),
            devices=int(cfg.get("devices", 1)),
            precision=str(cfg.get("precision", "32-true")),
            log_every_n_steps=int(cfg.get("log_every_n_steps", 1)),
            default_root_dir=os.path.join(experiment_root, "checkpoints_byol"),
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        # ---- fold-level aggregation (read test outputs saved by the module) ----
        test_out_path = getattr(model, "last_test_save_path", None)
        if not test_out_path or not os.path.isfile(test_out_path):
            raise FileNotFoundError("Expected test outputs .pt was not saved; cannot aggregate k-fold results.")

        out = torch.load(test_out_path, map_location="cpu")
        y_true_fold = out["y_true"].numpy()
        y_pred_fold = out["y_pred"].numpy()

        report_dict = classification_report(
            y_true_fold,
            y_pred_fold,
            target_names=class_names_report,
            zero_division=0,
            output_dict=True,
        )
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true_fold, y_pred_fold):
            cm[int(t), int(p)] += 1

        fold_reports.append(report_dict)
        fold_confusions.append(cm)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "fold_result",
                        "fold_index": int(fold_i),
                        "run_name": run_name,
                        "test_outputs_path": test_out_path,
                        "classification_report": report_dict,
                        "confusion_matrix": cm.tolist(),
                    }
                )
                + "\n"
            )

        del model
        torch.cuda.empty_cache()
        wandb.finish()

    # ---- compute averages ----
    avg_cm = np.mean(np.stack(fold_confusions, axis=0), axis=0)

    def _avg_report(reports: list[dict]) -> dict:
        # Average classwise precision/recall/f1/support + macro/weighted averages.
        keys = ["precision", "recall", "f1-score", "support"]
        out: dict = {}
        for k in class_names_report + ["macro avg", "weighted avg"]:
            if k not in reports[0]:
                continue
            out[k] = {}
            for m in keys:
                vals = [float(r[k].get(m, 0.0)) for r in reports]
                out[k][m] = float(np.mean(vals))
        out["accuracy"] = float(np.mean([float(r.get("accuracy", 0.0)) for r in reports]))
        return out

    avg_report = _avg_report(fold_reports)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "type": "kfold_average",
                    "k_fold": int(len(fold_splits)),
                    "run_names": fold_run_names,
                    "avg_confusion_matrix": avg_cm.tolist(),
                    "avg_classification_report": avg_report,
                }
            )
            + "\n"
        )

    print("\n=== K-FOLD SUMMARY ===")
    print(f"log_file: {log_path}")
    print(f"avg_accuracy: {avg_report.get('accuracy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
