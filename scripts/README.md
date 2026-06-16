# Catalog pipeline (`scripts/`)

Tools for building, cross-matching, de-duplicating and visually inspecting the
radio-galaxy catalogs in `../catalogs/`, plus downloaders for the image cutouts.

## Conventions

- **Repo-relative paths.** Scripts locate the repo root from their own location
  (`REPO = Path(__file__).resolve().parents[N]`) and read/write catalogs under
  `REPO/catalogs/`. No absolute paths — clone anywhere and they work.
- **Image cutouts live outside git.** The cutout tree (`Data/`, ~100 GB) is
  *not* in the repo. Scripts that read/write cutouts use:

  ```bash
  export RGC_DATA=/path/to/your/Data      # defaults to <repo>/Data if unset
  ```

- **Environment.** `pip install -r ../requirements.txt` (astropy, astroquery,
  pandas, numpy, matplotlib, plus `datalab` for DESI and `legacystamps`).

## Analysis scripts

| script | what it does |
|--------|--------------|
| `crossmatch_catalogs.py` | Positional cross-match of two CSV catalogs (cone search via astropy). `--self` finds internal duplicates within one catalog; `--dedup-out` writes a de-duplicated catalog keeping the highest-vote row per group. |
| `deduplicate.py` | Merge confirmed cross-catalog duplicate groups (from `DUPLICATES_report.csv`) into one row each, filling nulls from twins. |
| `compare_duplicates.py` | Download FIRST cutouts for duplicate pairs and plot them side by side for visual checking. |
| `match_awise_catalog.py` | Cross-match the master catalog against AllWISE (IRSA) and pull the full counterpart row + derived WISE photometry. |
| `match_desi_legacy.py` | Cross-match against DESI Legacy (Astro Data Lab, q3c) and pull the full Tractor row + photo-z. |
| `plot_duplicate_pairs.py` | Project-specific inspection PDF (master ↔ RGZ pairs). |
| `plot_duplicate_pdf.py` | **General, reusable** inspection PDF: takes a duplicates CSV + one or two image folders (`--self` for one). |

## Downloaders (`download/`)

All write into `$RGC_DATA/...`, are resumable (skip existing files) and log per source.

| script | survey / target |
|--------|-----------------|
| `download_vla_first_cutouts.py` | VLA FIRST 1.4 GHz cutouts for the RGZ DR1 FIRST catalog (SkyView). |
| `download_first_cutouts_rgz.py`  | FIRST cutouts for the RGZ DR1 FIRST catalog (near-duplicate of the above; SkyView). |
| `download_wise_cutouts.py`       | WISE AllWISE single-band cutouts (IRSA IBE). |
| `download_wise_first_overlay.py` | AllWISE W1+W2 cutouts with FIRST radio contours overlaid, for host-galaxy ID. |
| `download_desi_cutouts_radec.py` | DESI Legacy g+r+z cutouts using the catalog RA/Dec (`legacystamps`). |
| `download_desi_cutouts_hostname.py` | DESI Legacy g+r+z cutouts for sources with a DESI `Host_name`. |

## Typical flow

```bash
export RGC_DATA=/media/.../Data

# 1. find internal duplicates and write a clean catalog
python crossmatch_catalogs.py --self --ra1 RA --dec1 Dec --radius 5 --dedup-out

# 2. visually inspect the flagged pairs
python plot_duplicate_pdf.py \
    --csv ../catalogs/master_radio_catalog_deduplicate_removed_duplicates.csv \
    --self --img-a "$RGC_DATA/master_radio_catalog/png" \
    --pattern "{widx}_{source_catalog}"

# 3. add multiwavelength counterparts
python match_awise_catalog.py
python match_desi_legacy.py
```

See `../docs/` for column dictionaries, matching notes and the dedup/consistency reports.
