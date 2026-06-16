# RGC (Radha Gobinda Chandra): a radio AGN classifier
RGC is a Radio AGN Classifier named after Radha Gobinda Chandra (1878–1975), a Bangladeshi-Indian variable star observer from the first half of the twentieth century.

RGC uses convolutional neural networks to classify radio galaxies (galaxies that emit radiation in radio wavelengths). It incorporates the current deep learning models available in astronomical literature.

## Repository layout

- `src/`, `notebooks/`, `pre_processing/`, `examples/` — the deep-learning classifier (training, evaluation, inference).
- `scripts/` — the **catalog pipeline**: cross-matching, de-duplication, multiwavelength matching (AllWISE, DESI Legacy) and cutout downloaders. See [`scripts/README.md`](scripts/README.md).
- `catalogs/` — radio-galaxy catalogs and key data products (`raw/` holds source catalogs as published; `rgz_dr1/` holds Radio Galaxy Zoo DR1 tables).
- `docs/` — column dictionaries, matching notes and the dedup/consistency reports.

Image cutouts (~100 GB) are **not** stored in git; point the scripts at your local copy with the `RGC_DATA` environment variable.
