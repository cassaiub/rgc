#!/usr/bin/env python3
"""
Cross-match every radio source in the master radio catalog against the DESI
Legacy Imaging Surveys (Tractor catalog) via the NOIRLab Astro Data Lab, and
pull the FULL Tractor row PLUS the FULL photo-z row for the counterpart.

This is the DESI analogue of match_awise_catalog.py. The big addition over
AllWISE: DESI Legacy has PHOTO-Z (Zhou et al. 2021) and, where available, a
matched SDSS/DESI spec-z -- so this is where the "redshift" information lives.

For each radio (RA,Dec) it runs a q3c cone search of ls_dr9.tractor (DR9 by
default; --dr dr10 also works), keeps the NEAREST source within --radius
(default 3"), LEFT JOINs the photo_z table on ls_id, and records EVERY column
of both tables. The tractor table already provides converted magnitudes
(mag_*), dereddened mags/fluxes (dered_*), colors (g_r, w1_w2, ...), shape
parameters, quality/mask flags and Gaia cross-match info, so no derived
photometry is computed here -- the raw catalog values are passed straight
through. Duplicate join keys (ls_id/brickid/objid/release) are taken from the
tractor side only, so column names never collide.

Bookkeeping columns added in front of the catalog columns:
  widx, in_ra, in_dec, z_cat (input redshift), n_match, sep_arcsec

Output : <out> (CSV). No-counterpart sources get a row with n_match=0 (catalog
         columns blank) so they are not retried. Query errors go to
         <out>.errors.csv for retry.
Resumable: re-running skips widx already in the output.

Usage:
    python match_desi_legacy.py
    python match_desi_legacy.py --radius 6        # match the AllWISE run
    python match_desi_legacy.py --dr dr10         # use DR10 (adds i-band)
    python match_desi_legacy.py --all-matches     # one row per match in radius
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from dl import queryClient as qc

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "catalogs" / "master_radio_catalog_deduplicate_removed.csv")
OUT = str(REPO / "catalogs" / "master_radio_catalog_desi_match.csv")

RA_COL, DEC_COL, ID_COL, Z_COL = "RA", "Dec", "widx", "z"

# photo_z columns that also exist in tractor -> take them from tractor only.
PZ_DUP_KEYS = {"ls_id", "brickid", "objid", "release"}

MAX_RETRIES = 4
RETRY_WAIT = 5.0
SLEEP_BETWEEN = 0.15
QUERY_TIMEOUT = 120        # seconds; Data Lab can stall transiently, so cap + retry


def run_query(sql):
    """Run a Data Lab query with retries; return a DataFrame."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            csv = qc.query(sql=sql, fmt="csv", timeout=QUERY_TIMEOUT)
            return pd.read_csv(io.StringIO(csv))
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
    raise RuntimeError(f"query failed after {MAX_RETRIES} tries: {last}")


def photo_z_select(dr):
    """Return the 'p.col' fragment for every non-duplicate photo_z column."""
    schema = f"ls_{dr}"
    cols = list(pd.read_csv(io.StringIO(
        qc.query(sql=f"SELECT * FROM {schema}.photo_z LIMIT 0", fmt="csv"))).columns)
    extra = [c for c in cols if c not in PZ_DUP_KEYS]
    return "".join(f", p.{c}" for c in extra)


def all_columns(dr, psel):
    """Column list (and order) of t.* + the selected photo_z columns."""
    schema = f"ls_{dr}"
    sql = (f"SELECT t.*{psel} FROM {schema}.tractor t "
           f"LEFT JOIN {schema}.photo_z p ON t.ls_id = p.ls_id LIMIT 0")
    return list(pd.read_csv(io.StringIO(qc.query(sql=sql, fmt="csv"))).columns)


def build_sql(dr, ra, dec, radius_arcsec, psel):
    schema = f"ls_{dr}"
    return (
        f"SELECT t.*{psel}\n"
        f"FROM {schema}.tractor t\n"
        f"LEFT JOIN {schema}.photo_z p ON t.ls_id = p.ls_id\n"
        f"WHERE q3c_radial_query(t.ra, t.dec, {ra}, {dec}, {radius_arcsec/3600.0})"
    )


def row_from_match(widx, ra, dec, z, m, sep, n_match, all_cols):
    """Build one output record: bookkeeping + every catalog column of match `m`."""
    rec = {"widx": widx, "in_ra": ra, "in_dec": dec, "z_cat": z,
           "n_match": n_match, "sep_arcsec": float(sep) if sep is not None else np.nan}
    for c in all_cols:
        rec[c] = m[c] if m is not None else np.nan
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dr", default="dr9", choices=["dr9", "dr10"],
                    help="Legacy Survey data release (default %(default)s)")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="match radius in arcsec (default %(default)s; "
                         "use 6 to match the AllWISE run)")
    ap.add_argument("--all-matches", action="store_true",
                    help="write every match within radius, not just the nearest")
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    for col in (RA_COL, DEC_COL, ID_COL):
        if col not in df.columns:
            sys.exit(f"column '{col}' not found in {args.catalog}")
    has_z = Z_COL in df.columns

    # Resolve the full column list once so every row (match or not) is aligned.
    psel = photo_z_select(args.dr)
    all_cols = all_columns(args.dr, psel)
    print(f"fetching ALL {len(all_cols)} columns from {args.dr} tractor+photo_z")

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["widx"])["widx"].astype(int))
        print(f"resuming: {len(done)} widx already in {args.out}")

    err_log = args.out + ".errors.csv"
    n = len(df)
    print(f"{n} sources | {args.dr} | radius {args.radius}\" -> {args.out}")
    matched = unmatched = failed = 0

    for i, row in df.iterrows():
        widx = int(row[ID_COL])
        if widx in done:
            continue
        ra, dec = float(row[RA_COL]), float(row[DEC_COL])
        z = float(row[Z_COL]) if has_z and pd.notna(row[Z_COL]) else np.nan

        try:
            res = run_query(build_sql(args.dr, ra, dec, args.radius, psel))
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
        if len(res) == 0:
            recs.append(row_from_match(widx, ra, dec, z, None, None, 0, all_cols))
            unmatched += 1
            print(f"[{i+1}/{n}] widx {widx}: no match")
        else:
            coord = SkyCoord(ra, dec, unit="deg")
            seps = coord.separation(
                SkyCoord(res["ra"].values, res["dec"].values, unit="deg")).arcsec
            order = np.argsort(seps)
            n_match = len(res)
            picks = order if args.all_matches else order[:1]
            for j in picks:
                recs.append(row_from_match(widx, ra, dec, z,
                                           res.iloc[int(j)], seps[int(j)], n_match, all_cols))
            matched += 1
            zc = res.iloc[int(order[0])].get("z_phot_mean")
            print(f"[{i+1}/{n}] widx {widx}: {n_match} match(es), "
                  f"nearest {seps[order[0]]:.2f}\" z_phot={zc}")

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
