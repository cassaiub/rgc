"""
Download Legacy Survey (DESI host) g+r+z FITS cutouts using the RA/Dec columns
from the master radio catalog (instead of parsing the DESI Host_name).

Output layout (under Data/checking_all_row_DESI_host_cutouts/):
    fits/   {widx}_desi.fits
    log/    download_radec.log

Resumable: skips rows whose FITS already exists in fits/.

Run with the .rgc2 venv:
    python download_desi_cutouts_radec.py
"""
import time
import os
from pathlib import Path

import legacystamps
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RGC_DATA", SCRIPT_DIR / "../../Data"))
CSV_PATH = SCRIPT_DIR / "../../catalogs/master_radio_catalog_deduplicate_removed.csv"
OUT_DIR = DATA_ROOT / "checking_all_row_DESI_host_cutouts"
FITS_DIR = OUT_DIR / "fits"
LOG_DIR = OUT_DIR / "log"

SIZE_DEG = 0.075
BANDS = "grz"
LAYER = "ls-dr9"
MAX_ATTEMPTS = 3
RETRY_DELAY = 5

for d in (FITS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "download_radec.log"


def download_desi_data(ra: float, dec: float, dest_path: Path) -> None:
    """Download a multi-band FITS cutout and save it as dest_path."""
    tmp_dir = FITS_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    downloaded = legacystamps.download(
        ra=ra,
        dec=dec,
        bands=BANDS,
        size=SIZE_DEG,
        mode="fits",
        layer=LAYER,
        ddir=str(tmp_dir),
    )
    Path(downloaded).replace(dest_path)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    todo = []
    for row in df.itertuples(index=False):
        widx = int(row.widx)
        # Skip rows already downloaded.
        if (FITS_DIR / f"{widx}_desi.fits").exists():
            continue
        todo.append((widx, row.RA, row.Dec))

    print(f"{len(todo)} of {len(df)} rows to download (missing FITS only).")

    n_ok = n_fail = n_skip = 0
    with LOG_PATH.open("a") as log_f:
        for widx, ra, dec in todo:
            dest = FITS_DIR / f"{widx}_desi.fits"

            # Skip rows without valid coordinates.
            if pd.isna(ra) or pd.isna(dec):
                msg = f"Missing RA/Dec: ra={ra} dec={dec}"
                print(f"[SKIP] {widx} | {msg}")
                log_f.write(f"{widx}_desi.fits - SKIP - {msg}\n")
                log_f.flush()
                n_skip += 1
                continue

            ra, dec = float(ra), float(dec)

            status, error_msg = "FAILED", ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    download_desi_data(ra, dec, dest)
                    status, error_msg = "OK", ""
                    print(f"[OK] {widx} | RA={ra:.6f} Dec={dec:.6f}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"[FAILED {attempt}/{MAX_ATTEMPTS}] {widx} | RA={ra:.6f} Dec={dec:.6f} | {error_msg}")
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(RETRY_DELAY)

            n_ok += status == "OK"
            n_fail += status == "FAILED"
            log_f.write(f"{widx}_desi.fits - {status} - {error_msg}\n")
            log_f.flush()

    print(f"Done. OK={n_ok} FAILED={n_fail} SKIP={n_skip}. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
