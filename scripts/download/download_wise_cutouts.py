"""
Download WISE AllWISE W1 FITS cutouts for sources from the master radio catalog.

Output layout (under Data/WISE_cutouts/):
    fits/   {widx}_awise.fits
    log/    download.log

Resumable: skips sources whose FITS already exists.

Run with the .rgc2 venv:
    python download_wise_cutouts.py
"""
import time
import requests
import os
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ibe import IbeClass
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

def save_png(fits_path: Path, png_path: Path) -> None:
    """Save a ZScale-normalized grayscale PNG preview of a FITS file."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    data = np.nan_to_num(data, nan=0.0)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    plt.figure(figsize=(6, 6))
    plt.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Pixel value")
    plt.title(fits_path.stem)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RGC_DATA", SCRIPT_DIR / "../../Data"))
CSV_PATH = SCRIPT_DIR / "../../catalogs/master_radio_catalog_deduplicate_removed.csv"
OUT_DIR = DATA_ROOT / "WISE_cutouts"
FITS_DIR = OUT_DIR / "fits"
LOG_DIR = OUT_DIR / "log"

SIZE_DEG   = 0.0075
BAND       = "w3"
BAND_VALUE = 1        # integer used in IBE metadata table
MAX_ATTEMPTS = 3
RETRY_DELAY  = 5

METADATA_ROOT = "p3am_cdd"
URL_ROOT = f"https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/{METADATA_ROOT}"

for d in (FITS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "download.log"
# add near the top with other dirs
PNG_DIR = OUT_DIR / "png"

for d in (FITS_DIR, LOG_DIR, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

def get_tile_urls(ra: float, dec: float) -> list[str]:
    """Query IRSA IBE and return FITS URLs for matching AllWISE tiles."""
    position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    wise = IbeClass()
    metadata = wise.query_region(
        coordinate = position,
        mission    = "wise",
        dataset    = "allwise",
        table      = METADATA_ROOT,
        columns    = "band,coadd_id",
        width      = f"{SIZE_DEG}deg",
        intersect  = "COVERS",
    )
    if metadata is None or len(metadata) == 0:
        return []

    coadd_ids = metadata[metadata["band"] == BAND_VALUE]["coadd_id"]
    if len(coadd_ids) == 0:
        return []

    urls = []
    for coadd_id in coadd_ids:
        coaddgrp = coadd_id[:2]
        coadd_ra = coadd_id[:4]
        urls.append(
            f"{URL_ROOT}/{coaddgrp}/{coadd_ra}/{coadd_id}"
            f"/{coadd_id}-{BAND}-int-3.fits"
        )
    return urls


def download_awise_data(ra: float, dec: float, dest_path: Path) -> None:
    """Download the first matching AllWISE tile and save it as dest_path."""
    urls = get_tile_urls(ra, dec)
    if not urls:
        raise ValueError(f"No AllWISE tiles found for RA={ra}, Dec={dec}")

    # Use the first tile (COVERS guarantees it fully contains the region)
    response = requests.get(urls[0], stream=True, timeout=120)
    response.raise_for_status()

    tmp_path = dest_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp_path.replace(dest_path)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    todo = [
        (int(row.widx), float(row.RA), float(row.Dec))
        for row in df.itertuples(index=False)
        if not (FITS_DIR / f"{int(row.widx)}_awise.fits").exists()
    ]

    print(f"{len(todo)} of {len(df)} sources to download (missing FITS only).")

    n_ok = n_fail = 0
    with LOG_PATH.open("a") as log_f:
        for widx, ra, dec in todo:
            dest = FITS_DIR / f"{widx}_awise.fits"

            status, error_msg = "FAILED", ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    download_awise_data(ra, dec, dest)
                    
                    # Save PNG preview
                    png_path = PNG_DIR / f"{widx}_awise.png"
                    save_png(dest, png_path)
                    
                    status, error_msg = "OK", ""
                    print(f"[OK] {widx} | RA={ra}, Dec={dec}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"[FAILED {attempt}/{MAX_ATTEMPTS}] {widx} | RA={ra}, Dec={dec} | {error_msg}")
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(RETRY_DELAY)

            n_ok   += status == "OK"
            n_fail += status == "FAILED"
            log_f.write(f"{widx}_awise.fits - {status} - {error_msg}\n")
            log_f.flush()

    print(f"Done. OK={n_ok} FAILED={n_fail}. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()