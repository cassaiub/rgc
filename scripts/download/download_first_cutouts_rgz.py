"""
Download FIRST (1.4 GHz) cutouts for the de-duplicated catalogue.

Layout (all under Data/master_radio_catalog/):
    fits/   {widx}_{source_catalog}.fits
    png/    {widx}_{source_catalog}.png
    log/    download.log   (one line per source: OK / FAILED + error)

Naming uses widx + source_catalog provenance, e.g. 3_Lao+Mirabest.fits
Resumable: sources whose FITS already exists are skipped.

Run:
    python download_first_cutouts_rgz.py
"""
import os
import time
from pathlib import Path

from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import numpy as np
import pandas as pd
from PIL import Image

SURVEY = "VLA FIRST (1.4 GHz)"
PIXELS = (150, 150)
MAX_ATTEMPTS = 3
RETRY_DELAY = 5

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RGC_DATA", REPO / "Data"))
csv_path = REPO / "catalogs" / "rgz_dr1" / "DR1_FIRST_radio_classifications.csv"

master_dir = DATA / "DR1_FIRST_radio_classifications"
fits_dir = master_dir / "fits"
png_dir = master_dir / "png"
log_dir = master_dir / "log"
for d in (fits_dir, png_dir, log_dir):
    d.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "download.log"


def fits_to_png(input_fits_path: str, output_png_path: str):
    data = fits.getdata(input_fits_path)
    header = fits.getheader(input_fits_path)
    width, height = header["NAXIS1"], header["NAXIS2"]
    data = np.reshape(data, (height, width))
    data[np.isnan(data)] = np.nanmin(data)
    denom = np.nanmax(data) - np.nanmin(data)
    scaled = np.zeros_like(data) if denom == 0 else (data - np.nanmin(data)) / denom * 255
    Image.fromarray(scaled.astype(np.uint8), mode="L").save(output_png_path)


def celestial_capture(coord: SkyCoord, filename: str) -> None:
    image = SkyView.get_images(position=coord, survey=SURVEY,
                               coordinates="J2000", pixels=PIXELS)[0]
    image.writeto(filename, overwrite=True, output_verify="ignore")


def safe(label: str) -> str:
    return str(label).strip().replace(" ", "").replace("/", "-")


df = pd.read_csv(csv_path, low_memory=False)
coords = SkyCoord(ra=df["RA"].values * u.deg, dec=df["Dec"].values * u.deg, frame="icrs")

todo = [
    (coord, int(row.CatID), safe(row.RGZID))
    for coord, row in zip(coords, df.itertuples(index=False))
    if not (fits_dir / f"{int(row.CatID)}_{safe(row.RGZID)}.fits").exists()
]
print(f"{len(todo)} of {len(df)} sources to download (missing FITS only).")

n_ok = n_fail = 0
with log_path.open("a") as log_f:
    for coord, CatID, RGZID in todo:
        base = f"{CatID}_{RGZID}"
        fpath = fits_dir / f"{base}.fits"
        status, error_msg = "FAILED", ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                celestial_capture(coord, str(fpath))
                fits_to_png(str(fpath), str(png_dir / f"{base}.png"))
                status, error_msg = "OK", ""
                print(f"[OK] {base}")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"[FAILED {attempt}/{MAX_ATTEMPTS}] {base} | {error_msg}")
                if "list index out of range" in error_msg or "404" in error_msg:
                    break  # outside survey footprint -- no point retrying
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY)
        n_ok += status == "OK"
        n_fail += status == "FAILED"
        log_f.write(f"{base}.fits - {status} - {error_msg}\n")
        log_f.flush()

print(f"Done. OK={n_ok} FAILED={n_fail}. Log: {log_path}")
