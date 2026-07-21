# RGC (Radha Gobinda Chandra): a radio AGN classifier
RGC is a Radio AGN Classifier named after Radha Gobinda Chandra (1878–1975), a Bangladeshi-Indian variable star observer from the first half of the twentieth century.

RGC uses convolutional neural networks to classify radio galaxies (galaxies that emit radiation in radio wavelengths). It incorporates the current deep learning models available in astronomical literature.

## Repository layout

- `src/` — model code and the two training entry points (`src/scripts/`), the shared `config.yaml`, and the training guide [`src/TRAINING.md`](src/TRAINING.md).
- `notebooks/` — evaluation (ROC / AUC / ECE), Grad-CAM / attention visualisation, and single-image inference.
- `preprocessing/` — notebooks that turn FITS sources into the PNG cutouts used for training.
- `examples/` — a standalone example (`5foldoptuna_byol_finetuning_example.py`) showing Optuna 5-fold BYOL fine-tuning.
- `catalogs/` — radio-galaxy catalogs used for labels (`first-2060.csv`, `sasmal_ml_catalog.tsv`).

Image cutouts are large and are **not** stored in git. Point training at your local copy via `config.yaml`
(`data_params.data_path` for pretraining, `finetune.data_dir` for fine-tuning). Cache and output roots default to
the working directory, or to `RGC_PROJECT_ROOT` if it is set.

See [`src/TRAINING.md`](src/TRAINING.md) for the full pretrain → fine-tune workflow.
