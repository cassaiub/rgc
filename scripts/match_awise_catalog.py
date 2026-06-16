#!/usr/bin/env python3
"""
Cross-match every radio source in the master radio catalog against the AllWISE
Source Catalog (IRSA) and pull the FULL AllWISE row of the counterpart (all 298
columns), plus a few convenience quantities that are NOT in the catalog.

For each radio (RA, Dec) it does a cone search of allwise_p3as_psd, keeps the
NEAREST AllWISE source within --radius (default 6"), and records:

  - bookkeeping: widx, in_ra, in_dec, z_cat (input redshift), n_match, sep_arcsec
    (the input redshift is written as `z_cat` because AllWISE already has its own
    column named `z` -- the unit-sphere position component)
  - EVERY AllWISE column: profile-fit W1-W4 mags + uncertainties + SNR, raw
    fluxes/sky, aperture photometry (8 apertures), variability/repeatability,
    proper motion, all quality/contamination flags, the built-in 2MASS J/H/K
    match and the 2MASS XSC galaxy parameters, etc.
  - derived extras (computed here, not in the catalog):
      f_w1_mjy .. f_w4_mjy   flux densities in mJy (from the Vega zero-points)
      w1_w2                  W1-W2 colour
      agn_wise               Stern+2012 WISE-AGN flag (W1-W2 >= 0.8)

NOTE: AllWISE is a photometric catalog and has NO redshift. Redshift for these
sources lives in the input catalog's own `z` column; this script carries it
through so everything is in one table.

Output : <out>  (CSV, default alongside the catalog). Sources with no AllWISE
         counterpart get a row with n_match=0 and blank photometry, so they are
         not retried. Genuine query errors go to <out>.errors.csv for retry.

Resumable: re-running skips widx already present in the output.

Usage:
    python match_awise_catalog.py
    python match_awise_catalog.py --radius 3
    python match_awise_catalog.py --all-matches   # one row per match, not just nearest
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "catalogs" / "master_radio_catalog_deduplicate_removed.csv")
OUT = str(REPO / "catalogs" / "master_radio_catalog_awise_match.csv")

RA_COL, DEC_COL, ID_COL, Z_COL = "RA", "Dec", "widx", "z"
AWISE_TABLE = "allwise_p3as_psd"

# WISE Vega zero-magnitude flux densities (Jy); flux = ZP * 10**(-mag/2.5)
WISE_ZP_JY = {1: 309.540, 2: 171.787, 3: 31.674, 4: 8.363}

MAX_RETRIES = 4
RETRY_WAIT = 5.0
SLEEP_BETWEEN = 0.2


def mag_to_mjy(band, mag):
    """Vega mag -> flux density in mJy, NaN-safe."""
    if mag is None or not np.isfinite(mag):
        return np.nan
    return WISE_ZP_JY[band] * 10.0 ** (-mag / 2.5) * 1e3


def cone_search(coord, radius_arcsec):
    """Cone search (all columns) with retries; returns an astropy Table."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Irsa.query_region(
                coord, catalog=AWISE_TABLE, spatial="Cone",
                radius=radius_arcsec * u.arcsec, columns="*",
            )
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
    raise RuntimeError(f"query failed after {MAX_RETRIES} tries: {last}")


def _clean(v):
    """Catalog scalar -> plain value: masked -> NaN, bytes -> str."""
    if v is None or v is np.ma.masked or np.ma.is_masked(v):
        return np.nan
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return v


def _f(v):
    """Float-or-NaN."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def row_from_match(widx, ra, dec, z, m, sep, n_match, all_cols):
    """Build one output record: bookkeeping + every AllWISE column + extras."""
    rec = {"widx": widx, "in_ra": ra, "in_dec": dec, "z_cat": z,
           "n_match": n_match, "sep_arcsec": float(sep) if sep is not None else np.nan}
    for c in all_cols:
        rec[c] = _clean(m[c]) if m is not None else np.nan
    # Derived extras (not in the catalog).
    for b in (1, 2, 3, 4):
        rec[f"f_w{b}_mjy"] = mag_to_mjy(b, _f(rec.get(f"w{b}mpro"))) if m is not None else np.nan
    w1, w2 = _f(rec.get("w1mpro")), _f(rec.get("w2mpro"))
    rec["w1_w2"] = (w1 - w2) if (np.isfinite(w1) and np.isfinite(w2)) else np.nan
    rec["agn_wise"] = bool(np.isfinite(rec["w1_w2"]) and rec["w1_w2"] >= 0.8)
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--radius", type=float, default=6.0,
                    help="match radius in arcsec (default %(default)s)")
    ap.add_argument("--all-matches", action="store_true",
                    help="write every match within radius, not just the nearest")
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    for col in (RA_COL, DEC_COL, ID_COL):
        if col not in df.columns:
            sys.exit(f"column '{col}' not found in {args.catalog}")
    has_z = Z_COL in df.columns

    # Authoritative, fixed column list so every row is aligned (even no-match rows).
    all_cols = list(Irsa.list_columns(AWISE_TABLE).keys())
    print(f"fetching ALL {len(all_cols)} AllWISE columns (+ derived extras)")

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["widx"])["widx"].astype(int))
        print(f"resuming: {len(done)} widx already in {args.out}")

    err_log = args.out + ".errors.csv"
    n = len(df)
    matched = unmatched = failed = 0

    for i, row in df.iterrows():
        widx = int(row[ID_COL])
        if widx in done:
            continue
        ra, dec = float(row[RA_COL]), float(row[DEC_COL])
        z = float(row[Z_COL]) if has_z and pd.notna(row[Z_COL]) else np.nan
        coord = SkyCoord(ra, dec, unit="deg")

        try:
            tbl = cone_search(coord, args.radius)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i+1}/{n}] widx {widx}: ERROR {e}")
            new = not os.path.exists(err_log)
            with open(err_log, "a") as fh:
                if new:
                    fh.write("widx,ra,dec,reason\n")
                fh.write(f'{widx},{ra},{dec},"{str(e)[:200]}"\n')
            time.sleep(SLEEP_BETWEEN)
            continue

        recs = []
        if tbl is None or len(tbl) == 0:
            recs.append(row_from_match(widx, ra, dec, z, None, None, 0, all_cols))
            unmatched += 1
            print(f"[{i+1}/{n}] widx {widx}: no match")
        else:
            seps = coord.separation(
                SkyCoord(tbl["ra"], tbl["dec"], unit="deg")).arcsec
            order = np.argsort(seps)
            n_match = len(tbl)
            picks = order if args.all_matches else order[:1]
            for j in picks:
                recs.append(row_from_match(widx, ra, dec, z, tbl[int(j)],
                                           seps[int(j)], n_match, all_cols))
            matched += 1
            print(f"[{i+1}/{n}] widx {widx}: {n_match} match(es), "
                  f"nearest {seps[order[0]]:.2f}\"")

        # Append incrementally so progress survives interruption.
        out_df = pd.DataFrame(recs)
        header = not os.path.exists(args.out)
        out_df.to_csv(args.out, mode="a", header=header, index=False)
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. matched={matched} unmatched={unmatched} failed={failed}")
    print(f"Output: {args.out}")
    if failed:
        print(f"Query errors in {err_log}; re-run to retry (done widx skipped).")


if __name__ == "__main__":
    main()
