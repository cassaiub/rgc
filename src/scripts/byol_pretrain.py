import os
import sys
import argparse
from pathlib import Path

import yaml

# This script lives in src/scripts/; put src/ on the path so `models`/`utils` import
# regardless of the working directory it is launched from.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.byol_trainer import BYOLTrainer


# ---- filesystem / cache roots (user-specified) ----
# Default to the directory you run the script from.
# Override with:  RGC_PROJECT_ROOT=/some/path python3 byol_pretrain.py ...
PROJECT_ROOT = os.environ.get("RGC_PROJECT_ROOT", os.getcwd())
SCRIPT_ROOT = Path(__file__).resolve().parent

os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".torch")
os.environ["WANDB_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs")
os.environ["WANDB_CACHE_DIR"] = os.path.join(PROJECT_ROOT, "wandb_runs", ".cache")

EXPERIMENT_ROOT = os.path.join(PROJECT_ROOT, "byol_pretrain")
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)


def _get(d: dict, k: str, default):
    v = d.get(k, default)
    return default if v is None else v


def main(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    exp = config.get("exp_params", {})
    model = config.get("model_params", {})
    data = config.get("data_params", {})
    logging = config.get("logging_params", {})
    wandb_cfg = logging.get("wandb", {}) if isinstance(logging, dict) else {}

    trainer = BYOLTrainer(
        image_size=int(_get(model, "image_size", 151)),
        kernel_size=int(_get(model, "kernel_size", 5)),
        N=int(_get(exp, "N", 16)),
        lr=float(_get(exp, "lr", 3e-4)),
        batch_size=int(_get(exp, "batch_size", 16)),
        num_workers=int(_get(exp, "num_workers", 4)),
        num_epochs=int(_get(exp, "num_epochs", 500)),
        device=str(_get(exp, "device", "cuda")),
        projection_size=int(_get(exp, "projection_size", 256)),
        projection_hidden_size=int(_get(exp, "proj_hidden_size", 4096)),
        moving_average_decay=float(_get(exp, "moving_average_decay", 0.99)),
        data_path=str(_get(data, "data_path", "./data/unlabeled/")),
        results_folder=str(_get(logging, "results_dir", "./results/")),
        wandb_cfg=wandb_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to BYOL pretrain YAML config (rgc_old schema).",
    )
    args = parser.parse_args()
    main(args.config)

