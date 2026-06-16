#!/usr/bin/env python3
"""
Download AllWISE W1 + W2 image cutouts for every source in the master radio
catalog, sized to match the VLA/FIRST cutouts so they can be overplotted to
identify the host galaxy of each radio AGN.

Service : IRSA IBE  -> AllWISE Atlas coadds (p3am_cdd), native 1.375"/pixel
Bands   : W1 (3.4 um, PRIMARY HDU)  +  W2 (4.6 um, extension 'W2')
Size    : SIZE_ARCSEC on a side (default 270" = 150 FIRST pixels @ 1.8"/pix)
Output  : <OUT_DIR>/<widx>_awise.fits  (multi-extension FITS)

An overlay PNG is also written to <OUT_DIR>/png/<widx>_awise.png: the W1
background in greyscale with the FIRST radio map drawn as contours and the
catalog (RA,Dec) marked, for visual host-galaxy identification. Disable with
--no-png.

Resumable: rows whose output file already exists are skipped.
Failures : appended to <OUT_DIR>/_failed.csv (widx, ra, dec, reason).

Usage:
    python download_awise_cutouts.py                  # default 270 arcsec
    python download_awise_cutouts.py --size 300       # custom size in arcsec
    python download_awise_cutouts.py --bands 1 2 3 4  # more bands if wanted
    python download_awise_cutouts.py --no-png         # FITS only
"""

import argparse
import glob
import io
import os
import sys
import time
from pathlib import Path
import warnings

from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# FIRST cutout headers carry non-ASCII commentary cards and NaN padding;
# these warnings are harmless and would otherwise flood the log.
warnings.simplefilter("ignore", VerifyWarning)
warnings.simplefilter("ignore", AstropyUserWarning)

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from astroquery.ipac.irsa.ibe import Ibe

# ---------------------------------------------------------------------------
# Defaults (override the common ones on the command line)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RGC_DATA", REPO / "Data"))
CATALOG = str(REPO / "catalogs" / "master_radio_catalog_deduplicate_removed.csv")
OUT_DIR = str(DATA / "awise_cutouts")

# FIRST cutouts to overlay as radio contours; files are <widx>_<catalog>.fits
FIRST_DIR = str(DATA / "master_radio_catalog" / "fits")

# Radio contour levels = CONTOUR_SIGMA * (rms) * CONTOUR_STEPS
CONTOUR_SIGMA = 3.0
CONTOUR_STEPS = [1, 2, 4, 8, 16, 32, 64]

SIZE_ARCSEC = 270.0          # field of view per side; 150 px * 1.8"/px (FIRST)
BANDS = [1, 2]               # W1 -> primary HDU, the rest -> extensions

RA_COL, DEC_COL, ID_COL = "RA", "Dec", "widx"

# AllWISE Atlas intensity coadds on the IBE data server
IBE_DATA = "https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd"

MAX_RETRIES = 4              # per HTTP request
RETRY_WAIT = 5.0            # seconds, grows linearly with attempt number
SLEEP_BETWEEN = 0.3        # polite pause between sources (seconds)
HTTP_TIMEOUT = 120          # seconds per request


def cutout_url(coadd_id, band, ra, dec, size_arcsec):
    """Build an IBE cutout URL for one AllWISE Atlas band.

    The data path is derived from the coadd_id, e.g. coadd_id '0000p000_ac51'
    lives under .../p3am_cdd/00/0000/0000p000_ac51/.
    """
    c = coadd_id
    fname = f"{c}-w{band}-int-3.fits"
    return (
        f"{IBE_DATA}/{c[:2]}/{c[:4]}/{c}/{fname}"
        f"?center={ra},{dec}deg&size={size_arcsec}arcsec&gzip=false"
    )


def get_fits(url):
    """GET a FITS file with retries; return an astropy HDUList."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            if r.status_code == 200 and r.content:
                return fits.open(io.BytesIO(r.content))
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT * attempt)
    raise RuntimeError(f"download failed after {MAX_RETRIES} tries: {last}")


def find_coadd(ra, dec):
    """Return the coadd_id of the AllWISE Atlas tile most centered on (ra,dec).

    Raises LookupError if the position has no AllWISE coverage.
    """
    coord = SkyCoord(ra, dec, unit="deg")
    tbl = Ibe.query_region(
        coordinate=coord, mission="wise", dataset="allwise",
        table="p3am_cdd", most_centered=True,
    )
    if tbl is None or len(tbl) == 0:
        raise LookupError("no AllWISE coverage")
    # most_centered returns one row per band, all on the same coadd tile
    return str(tbl[0]["coadd_id"])


def build_cutout(widx, ra, dec, size_arcsec, bands):
    """Download the requested bands and assemble one multi-extension HDUList."""
    coadd_id = find_coadd(ra, dec)

    primary = None
    extensions = []
    for band in bands:
        url = cutout_url(coadd_id, band, ra, dec, size_arcsec)
        with get_fits(url) as hdul:
            data = hdul[0].data.copy()
            hdr = hdul[0].header.copy()
        if data is None:
            raise RuntimeError(f"empty data for W{band}")
        hdr["EXTNAME"] = f"W{band}"
        hdr["WIDX"] = (int(widx), "master catalog widx")
        hdr["RA_OBJ"] = (float(ra), "catalog RA (deg)")
        hdr["DEC_OBJ"] = (float(dec), "catalog Dec (deg)")
        hdr["COADD_ID"] = (coadd_id, "AllWISE Atlas coadd id")
        hdr["CUTSIZE"] = (size_arcsec, "cutout size on a side (arcsec)")
        if primary is None:
            primary = fits.PrimaryHDU(data=data, header=hdr)
        else:
            extensions.append(fits.ImageHDU(data=data, header=hdr))
    return fits.HDUList([primary, *extensions])


def find_first(widx):
    """Return the path to this widx's FIRST cutout, or None if absent."""
    hits = glob.glob(os.path.join(FIRST_DIR, f"{widx}_*.fits"))
    return hits[0] if hits else None


def save_overlay_png(out_png, w1_hdu, ra, dec, widx, first_path):
    """W1 greyscale background + FIRST radio contours + catalog host marker."""
    w1 = np.asarray(w1_hdu.data, dtype=float)
    w1_wcs = WCS(w1_hdu.header)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=w1_wcs)
    norm = ImageNormalize(w1, interval=ZScaleInterval())
    ax.imshow(w1, origin="lower", cmap="gray", norm=norm)

    title = f"widx {widx} - AllWISE W1"
    if first_path:
        with fits.open(first_path) as fh:
            fdata = np.squeeze(np.asarray(fh[0].data, dtype=float))
            # FIRST headers can carry >2 WCS axes; keep the celestial pair
            fwcs = WCS(fh[0].header).celestial
        _, _, std = sigma_clipped_stats(fdata, sigma=3.0, maxiters=5)
        levels = [CONTOUR_SIGMA * std * s for s in CONTOUR_STEPS]
        ax.contour(fdata, levels=levels, colors="red", linewidths=0.6,
                   transform=ax.get_transform(fwcs))
        title += " + FIRST contours"
    else:
        title += " (no FIRST cutout found)"

    ax.scatter(ra, dec, transform=ax.get_transform("world"),
               marker="+", s=120, color="cyan", linewidths=1.2,
               label="catalog position")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.6)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--size", type=float, default=SIZE_ARCSEC,
                    help="cutout size per side in arcsec (default %(default)s)")
    ap.add_argument("--bands", type=int, nargs="+", default=BANDS,
                    help="WISE bands; first one is the primary HDU (default 1 2)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download even if the output file exists")
    ap.add_argument("--no-png", dest="png", action="store_false",
                    help="skip the FIRST-contour overlay PNGs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png")
    if args.png:
        os.makedirs(png_dir, exist_ok=True)
    fail_log = os.path.join(args.out_dir, "_failed.csv")

    df = pd.read_csv(args.catalog)
    for col in (RA_COL, DEC_COL, ID_COL):
        if col not in df.columns:
            sys.exit(f"column '{col}' not found in {args.catalog}")

    n = len(df)
    print(f"{n} sources | bands W{args.bands} | size {args.size}\" "
          f"-> {args.out_dir}")

    done = skipped = failed = 0
    for i, row in df.iterrows():
        widx = int(row[ID_COL])
        ra, dec = float(row[RA_COL]), float(row[DEC_COL])
        out = os.path.join(args.out_dir, f"{widx}_awise.fits")
        out_png = os.path.join(png_dir, f"{widx}_awise.png")

        png_ok = (not args.png) or os.path.exists(out_png)
        if os.path.exists(out) and png_ok and not args.overwrite:
            skipped += 1
            continue

        try:
            # Reuse the FITS if it already exists (only the PNG is missing).
            if os.path.exists(out) and not args.overwrite:
                hdul = fits.open(out)
            else:
                hdul = build_cutout(widx, ra, dec, args.size, args.bands)
                hdul.writeto(out, overwrite=True)
            if args.png:
                save_overlay_png(out_png, hdul[0], ra, dec, widx,
                                 find_first(widx))
            hdul.close()
            done += 1
            print(f"[{i + 1}/{n}] widx {widx}: OK")
        except Exception as e:  # noqa: BLE001 - log and keep going
            failed += 1
            print(f"[{i + 1}/{n}] widx {widx}: FAIL {e}")
            new = not os.path.exists(fail_log)
            with open(fail_log, "a") as fh:
                if new:
                    fh.write("widx,ra,dec,reason\n")
                fh.write(f'{widx},{ra},{dec},"{str(e).replace(chr(34), chr(39))}"\n')

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. downloaded={done} skipped={skipped} failed={failed}")
    if failed:
        print(f"Failures logged to {fail_log} "
              f"(re-run the script to retry them; successes are skipped).")


if __name__ == "__main__":
    main()
