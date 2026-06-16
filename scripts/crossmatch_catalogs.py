#!/usr/bin/env python3
"""
Positional cross-match between two CSV catalogs.

Give it two CSV files and the RA/Dec column names of each (they need NOT share
any other columns -- only sky position is used). It does a cone search of
catalog A against catalog B and splits catalog A into:

  * DUPLICATES  -- A-sources that have >=1 B-source within --radius. Each row
                   keeps all of A's columns plus:
                     n_match    how many B-sources fell within the radius
                     sep_arcsec separation to the NEAREST B-source (arcsec)
                     match_*    every column of the matched B-source
                                (the nearest one; or one row per match with
                                 --all-matches)
  * UNIQUE      -- A-sources with NO B-source within --radius (A columns only).

Optionally (--unique-b) it also writes the B-sources that have no A counterpart.

Matching uses astropy search_around_sky, so multiple B-matches per A-source are
found and counted, not just the nearest.

Usage:
    python crossmatch_catalogs.py A.csv B.csv
    python crossmatch_catalogs.py A.csv B.csv \\
        --ra1 RA --dec1 Dec --ra2 RAdeg --dec2 DEdeg --radius 3
    python crossmatch_catalogs.py A.csv B.csv --all-matches --unique-b

Defaults are set for this project:
    A = master_radio_catalog_deduplicate_removed.csv (RA/Dec)
    B = Lao/FIRST_bt_table1.csv                       (RAdeg/DEdeg)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky

REPO = Path(__file__).resolve().parents[1]
A_DEFAULT = str(REPO / "catalogs" / "master_radio_catalog_deduplicate_removed.csv")
B_DEFAULT = A_DEFAULT




def load_coords(df, ra_col, dec_col, path):
    """Return (SkyCoord, finite-mask) for the given RA/Dec columns."""
    for c in (ra_col, dec_col):
        if c not in df.columns:
            sys.exit(f"column '{c}' not found in {path}\n"
                     f"  available: {', '.join(map(str, df.columns))}")
    ra = pd.to_numeric(df[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df[dec_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    # Build SkyCoord only on the finite rows; bad rows are treated as unmatched.
    coord = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    return coord, good


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_a", nargs="?", default=A_DEFAULT,
                    help="catalog A (the one that gets split; default master)")
    ap.add_argument("csv_b", nargs="?", default=B_DEFAULT,
                    help="catalog B (matched against; default FIRST_bt_table1)")
    ap.add_argument("--ra1", default="RA", help="RA column in A (default %(default)s)")
    ap.add_argument("--dec1", default="Dec", help="Dec column in A (default %(default)s)")
    ap.add_argument("--ra2", default="RAdeg", help="RA column in B (default %(default)s)")
    ap.add_argument("--dec2", default="DEdeg", help="Dec column in B (default %(default)s)")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="cone-search radius in arcsec (default %(default)s)")
    ap.add_argument("--self", dest="self_match", action="store_true",
                    help="self-match A against itself to find internal duplicates "
                         "(ignores csv_b; uses A's RA/Dec; excludes each row "
                         "matching itself)")
    ap.add_argument("--dedup-out", dest="dedup_out", nargs="?", const="",
                    default=None,
                    help="self-match only: write a deduplicated catalog with one "
                         "row per group of sources within --radius, keeping the "
                         "highest-vote entry. Optionally give an output path; "
                         "default is alongside A as <prefix>_dedup.csv")
    ap.add_argument("--vote-col", default="N_votes",
                    help="column used to pick the kept row per group, highest wins "
                         "(default %(default)s; ties broken by N_total then row order)")
    ap.add_argument("--all-matches", action="store_true",
                    help="one row per A-B match in DUPLICATES, not just the nearest")
    ap.add_argument("--unique-b", action="store_true",
                    help="also write B-sources with no A counterpart")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: alongside catalog A)")
    ap.add_argument("--prefix", default=None,
                    help="output filename prefix (default: A's basename)")
    args = ap.parse_args()

    # In self-match mode B *is* A: same file, same RA/Dec columns.
    if args.self_match:
        args.csv_b = args.csv_a
        args.ra2, args.dec2 = args.ra1, args.dec1

    a = pd.read_csv(args.csv_a)
    b = a if args.self_match else pd.read_csv(args.csv_b)
    print(f"A: {len(a)} rows  {args.csv_a}")
    if args.self_match:
        print("SELF-MATCH mode: finding internal duplicates within A")
    else:
        print(f"B: {len(b)} rows  {args.csv_b}")

    ca, good_a = load_coords(a, args.ra1, args.dec1, args.csv_a)
    cb, good_b = load_coords(b, args.ra2, args.dec2, args.csv_b)
    if (~good_a).any():
        print(f"  A: {int((~good_a).sum())} rows have non-finite RA/Dec -> unique")
    if (~good_b).any():
        print(f"  B: {int((~good_b).sum())} rows have non-finite RA/Dec -> ignored")

    # Positional indices (into the full A/B frames) of the finite-coord rows.
    a_pos = np.flatnonzero(good_a)
    b_pos = np.flatnonzero(good_b)

    # All A-B pairs within the radius. idx_a/idx_b index into ca/cb (the finite
    # subsets); sep2d is the separation of each pair.
    idx_a, idx_b, sep2d, _ = search_around_sky(ca, cb, args.radius * u.arcsec)
    sep_arcsec = sep2d.arcsec

    if args.self_match:
        # ca and cb are identical here, so every source trivially matches itself
        # at sep=0 (idx_a == idx_b). Drop those self-pairs; what remains are
        # genuine internal duplicates. Both directions of a pair are kept so that
        # n_match / has_match are correct for *every* duplicated source (each one
        # then appears once in DUPLICATES with its nearest neighbour).
        keep = idx_a != idx_b
        idx_a, idx_b, sep_arcsec = idx_a[keep], idx_b[keep], sep_arcsec[keep]

    # Number of B-matches for each finite A-source, and its nearest B-match.
    n_match = np.zeros(len(ca), dtype=int)
    nearest_b = np.full(len(ca), -1, dtype=int)     # idx into cb
    nearest_sep = np.full(len(ca), np.nan)
    # Sort pairs by A then by separation so the first pair per A is the nearest.
    o = np.lexsort((sep_arcsec, idx_a))
    for k in o:
        ia = idx_a[k]
        n_match[ia] += 1
        if nearest_b[ia] < 0:           # first (=nearest) pair seen for this A
            nearest_b[ia] = idx_b[k]
            nearest_sep[ia] = sep_arcsec[k]

    # Map finite-subset counts back onto the full A frame (bad-coord rows -> 0).
    n_match_full = np.zeros(len(a), dtype=int)
    n_match_full[a_pos] = n_match
    has_match = n_match_full > 0

    b_pref = b.add_prefix("match_")     # avoid column-name collisions with A

    # ---- DUPLICATES (A-sources with a B match) --------------------------------
    dup_rows = []
    matched_b_full = set()              # full-B indices that matched any A
    for ia_local in np.flatnonzero(n_match > 0):
        a_full = a_pos[ia_local]
        base = a.iloc[a_full].to_dict()
        if args.all_matches:
            pairs = [(idx_b[k], sep_arcsec[k]) for k in o if idx_a[k] == ia_local]
        else:
            pairs = [(nearest_b[ia_local], nearest_sep[ia_local])]
        for jb_local, sep in pairs:
            b_full = b_pos[jb_local]
            matched_b_full.add(int(b_full))
            rec = dict(base)
            rec["n_match"] = int(n_match[ia_local])
            rec["sep_arcsec"] = float(sep)
            rec.update(b_pref.iloc[b_full].to_dict())
            dup_rows.append(rec)
    dup_df = pd.DataFrame(dup_rows)

    # ---- UNIQUE (A-sources with no B match) -----------------------------------
    uniq_df = a.loc[~has_match].copy()

    # ---- output paths ---------------------------------------------------------
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv_a))
    os.makedirs(outdir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.csv_a))[0]
    dup_path = os.path.join(outdir, f"{prefix}_duplicates.csv")
    uniq_path = os.path.join(outdir, f"{prefix}_unique.csv")

    dup_df.to_csv(dup_path, index=False)
    uniq_df.to_csv(uniq_path, index=False)

    n_dup_sources = int(has_match.sum())          # distinct A-sources with a match
    print(f"\nradius = {args.radius}\"")
    print(f"DUPLICATES: {n_dup_sources} A-sources matched "
          f"({len(dup_df)} rows written) -> {dup_path}")
    print(f"UNIQUE    : {len(uniq_df)} A-sources unmatched -> {uniq_path}")

    # ---- DEDUP (self-match only): one row per group, keep highest-vote --------
    if args.dedup_out is not None:
        if not args.self_match:
            sys.exit("--dedup-out only makes sense with --self")

        # Union-find over the full A frame: link every pair within the radius so
        # transitive duplicates (A~B, B~C) collapse into one group.
        parent = np.arange(len(a))

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:        # path compression
                parent[x], x = root, parent[x]
            return root

        for la, lb in zip(idx_a, idx_b):
            ra_, rb_ = find(a_pos[la]), find(a_pos[lb])
            if ra_ != rb_:
                parent[rb_] = ra_

        roots = np.array([find(i) for i in range(len(a))])

        # Pick the kept row per group: highest --vote-col, then N_total, then the
        # first occurrence. Singletons (most rows) are kept unchanged.
        order_cols, ascending = [], []
        if args.vote_col in a.columns:
            order_cols.append(args.vote_col); ascending.append(False)
        else:
            print(f"  warning: --vote-col '{args.vote_col}' not in catalog; "
                  f"keeping first row of each group")
        if "N_total" in a.columns:
            order_cols.append("N_total"); ascending.append(False)

        work = a.copy()
        work["_root"] = roots
        work["_orig"] = np.arange(len(a))
        if order_cols:
            work = work.sort_values(order_cols + ["_orig"],
                                    ascending=ascending + [True], kind="mergesort")
        else:
            work = work.sort_values("_orig", kind="mergesort")
        kept = work.drop_duplicates("_root", keep="first").sort_values("_orig")
        dedup_df = kept.drop(columns=["_root", "_orig"])

        dedup_path = args.dedup_out or os.path.join(
            outdir, f"{prefix}_dedup.csv")
        dedup_df.to_csv(dedup_path, index=False)
        print(f"DEDUP     : {len(a)} rows -> {len(dedup_df)} unique sources "
              f"({len(a) - len(dedup_df)} removed) -> {dedup_path}")

    if args.unique_b:
        uniq_b = b.loc[[i for i in b_pos if int(i) not in matched_b_full]].copy()
        # B rows with bad coords never match, so they are unique too.
        uniq_b = pd.concat([uniq_b, b.loc[~good_b]]).sort_index()
        uniq_b_path = os.path.join(outdir, f"{prefix}_unique_B.csv")
        uniq_b.to_csv(uniq_b_path, index=False)
        print(f"UNIQUE_B  : {len(uniq_b)} B-sources unmatched -> {uniq_b_path}")


if __name__ == "__main__":
    main()
