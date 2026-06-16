"""
Download Legacy Survey (DESI host) g+r+z FITS cutouts for sources with DESI Host_name.

Output layout (under Data/DESI_host_cutouts/):
    fits/   {widx}_desi.fits
    log/    download.log

Resumable: skips sources whose FITS already exists.

Run with the .rgc2 venv:
    python download_desi_cutouts_hostname.py
"""
import re
import time
import os
from pathlib import Path

import legacystamps
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RGC_DATA", SCRIPT_DIR / "../../Data"))
CSV_PATH = SCRIPT_DIR / "../../catalogs/master_radio_catalog_deduplicate_removed.csv"
OUT_DIR = DATA_ROOT / "DESI_host_cutouts/nondesi_desi_cutout"
FITS_DIR = OUT_DIR / "fits"
LOG_DIR = OUT_DIR / "log"

SIZE_DEG = 0.075
BANDS = "grz"
LAYER = "ls-dr9"
MAX_ATTEMPTS = 3
RETRY_DELAY = 5

for d in (FITS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "download.log"


def desi_name_to_coords(host_name: str) -> tuple[float | None, float | None]:
    """
    Convert DESI host name to decimal degrees.

    Supports:
      DESI JHHMMSS.ss+DDMMSS.s   (e.g. DESI J000111.20-002011.6)
      DESI Jddd.dddd+dd.dddd     (e.g. DESI J242.0188+43.1634)
    """
    coord_part = host_name.replace("DESI", "").strip()

    hms = re.match(
        r"J"
        r"(\d{2})(\d{2})(\d{2}\.\d+)"
        r"([+-])"
        r"(\d{2})(\d{2})(\d{2}\.\d+)",
        coord_part,
    )
    if hms:
        ra_h, ra_m, ra_s = map(float, hms.group(1, 2, 3))
        sign = -1 if hms.group(4) == "-" else 1
        dec_d, dec_m, dec_s = map(float, hms.group(5, 6, 7))
        ra = 15.0 * (ra_h + ra_m / 60 + ra_s / 3600)
        dec = sign * (dec_d + dec_m / 60 + dec_s / 3600)
        return ra, dec

    decimal = re.match(r"J(\d+\.\d+)([+-])(\d+\.\d+)", coord_part)
    if decimal:
        ra = float(decimal.group(1))
        sign = -1 if decimal.group(2) == "-" else 1
        dec = sign * float(decimal.group(3))
        return ra, dec

    return None, None


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
    desi_rows = df[df["Host_name"].astype(str).str.startswith("DESI", na=False)]
    todo = [
        (int(row.widx), str(row.Host_name).strip())
        for row in desi_rows.itertuples(index=False)
        if not (FITS_DIR / f"{int(row.widx)}_desi.fits").exists()
    ]

    print(f"{len(todo)} of {len(desi_rows)} DESI sources to download (missing FITS only).")

    n_ok = n_fail = n_skip = 0
    with LOG_PATH.open("a") as log_f:
        for widx, host_name in todo:
            dest = FITS_DIR / f"{widx}_desi.fits"
            ra, dec = desi_name_to_coords(host_name)
            if ra is None:
                msg = f"Could not parse Host_name: {host_name}"
                print(f"[SKIP] {widx} | {msg}")
                log_f.write(f"{widx}_desi.fits - SKIP - {msg}\n")
                log_f.flush()
                n_skip += 1
                continue

            status, error_msg = "FAILED", ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    download_desi_data(ra, dec, dest)
                    status, error_msg = "OK", ""
                    print(f"[OK] {widx} | {host_name}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"[FAILED {attempt}/{MAX_ATTEMPTS}] {widx} | {host_name} | {error_msg}")
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(RETRY_DELAY)

            n_ok += status == "OK"
            n_fail += status == "FAILED"
            log_f.write(f"{widx}_desi.fits - {status} - {error_msg}\n")
            log_f.flush()

    print(f"Done. OK={n_ok} FAILED={n_fail} SKIP={n_skip}. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
