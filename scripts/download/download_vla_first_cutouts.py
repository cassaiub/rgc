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


def fits_to_png(input_fits_path: str, output_png_path: str):
    data = fits.getdata(input_fits_path)
    header = fits.getheader(input_fits_path)

    width, height = header["NAXIS1"], header["NAXIS2"]
    data = np.reshape(data, (height, width))

    data[np.isnan(data)] = np.nanmin(data)

    denom = np.nanmax(data) - np.nanmin(data)
    if denom == 0:
        scaled = np.zeros_like(data)
    else:
        scaled = (data - np.nanmin(data)) / denom * 255

    image = Image.fromarray(scaled.astype(np.uint8), mode="L")
    image.save(output_png_path)


def celestial_capture(survey: str, coord: SkyCoord, filename: str, pixels=(150, 150)) -> None:
    image = SkyView.get_images(position=coord, survey=survey, coordinates="J2000", pixels=pixels)[0]
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    image.writeto(filename, overwrite=True, output_verify="ignore")


MAX_ATTEMPTS = 3
RETRY_DELAY = 5

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RGC_DATA", REPO / "Data"))
csv_path = REPO / "catalogs" / "rgz_dr1" / "DR1_FIRST_radio_classifications.csv"
catalog = pd.read_csv(csv_path)

coords = SkyCoord(
    ra=catalog["RA"].values * u.deg,
    dec=catalog["Dec"].values * u.deg,
    frame="icrs",
)

out_dir = DATA / "DR1_FIRST_radio_classifications"
out_fits_dir = out_dir / "fits"
out_png_dir = out_dir / "png"
out_fits_dir.mkdir(parents=True, exist_ok=True)
out_png_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "log" / "vla_download.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

total = len(catalog)
print(f"Catalog has {total} sources. Downloading missing VLA FIRST cutouts into {out_dir}.")

downloaded = skipped = failed = 0

with log_path.open("a") as log_f:
    for coord, row in zip(coords, catalog.itertuples(index=False)):
        base_name = f"{row.CatID}_{row.RGZID}"
        fits_path = out_fits_dir / f"{base_name}.fits"
        png_path = out_png_dir / f"{base_name}.png"

        # Skip if already downloaded.
        if fits_path.exists() and png_path.exists():
            skipped += 1
            continue

        status, error_msg = "FAILED", ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                celestial_capture("VLA FIRST (1.4 GHz)", coord, str(fits_path))
                fits_to_png(str(fits_path), str(png_path))
                status, error_msg = "OK", ""
                print(f"[OK] {base_name}")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"[FAILED {attempt}/{MAX_ATTEMPTS}] {base_name} | {error_msg}")
                if "list index out of range" in error_msg or "404" in error_msg:
                    break
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY)

        if status == "OK":
            downloaded += 1
        else:
            failed += 1
        log_f.write(f"{base_name}.fits - {status} - {error_msg}\n")
        log_f.flush()

print(
    f"Done. {downloaded} downloaded, {skipped} skipped (already present), "
    f"{failed} failed. Log written to {log_path}"
)
