#!/usr/bin/env python3
"""
Visual inspection PDF for cross-match duplicates.

For each row of the duplicates CSV it places the MASTER cutout and the matched
RGZ cutout side by side ("a pair"). Pairs are laid out 3 columns per row,
several rows per page, into a multi-page PDF.

Master png  : Data/master_radio_catalog/png/{widx}_{source_catalog}.png
RGZ    png  : Data/DR1_FIRST_radio_classifications/png/{idx}_{match_RGZID}.png
              (the {idx}_ prefix is not in the CSV, so we build an
               RGZID -> filename lookup by scanning the directory once.)

Rows whose RGZ source has no png are skipped and reported.

Usage:
    python plot_duplicate_pairs.py
    python plot_duplicate_pairs.py --limit 60        # quick preview
    python plot_duplicate_pairs.py --csv X.csv --out Y.pdf
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RGC_DATA", REPO / "Data"))
CSV_DEFAULT = str(REPO / "catalogs" / "master_radio_catalog_deduplicate_removed_duplicates.csv")
MDIR = str(DATA / "master_radio_catalog" / "png")
RDIR = str(DATA / "DR1_FIRST_radio_classifications" / "png")
OUT_DEFAULT = str(REPO / "catalogs" / "duplicate_pairs_inspection.pdf")

COLS = 3          # pairs across the page
ROWS_PER_PAGE = 4  # pairs down the page  -> 12 pairs / page


def build_rgz_lookup(rdir):
    """RGZID -> filename, parsed from '{idx}_{RGZID}.png'."""
    lut = {}
    for f in os.listdir(rdir):
        if f.endswith(".png"):
            base = f[:-4]
            rgzid = base.split("_", 1)[1] if "_" in base else base
            lut[rgzid] = f
    return lut


def show(ax, path, title):
    if path and os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, "(no image)", ha="center", va="center",
                fontsize=8, color="0.5", transform=ax.transAxes)
    ax.set_title(title, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rgz = build_rgz_lookup(RDIR)

    # Build the list of pairs that have BOTH images.
    pairs, missing = [], 0
    for _, r in df.iterrows():
        m = f"{r['widx']}_{r['source_catalog']}.png"
        mpath = os.path.join(MDIR, m)
        rid = str(r["match_RGZID"]).strip()
        rfile = rgz.get(rid)
        rpath = os.path.join(RDIR, rfile) if rfile else None
        if rpath is None or not os.path.exists(mpath):
            missing += 1
            continue
        sep = r.get("sep_arcsec", float("nan"))
        pairs.append({
            "mpath": mpath, "rpath": rpath,
            "mtitle": f"MASTER  {r['source_catalog']} {r.get('FIRST_ID','')}",
            "rtitle": f"RGZ  {rid}\nsep={sep:.2f}\"",
        })

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"pairs with both images : {len(pairs)}")
    print(f"rows skipped (no rgz png): {missing}")
    if not pairs:
        print("nothing to plot.")
        return

    per_page = COLS * ROWS_PER_PAGE
    n_pages = (len(pairs) + per_page - 1) // per_page

    with PdfPages(args.out) as pdf:
        for p in range(n_pages):
            chunk = pairs[p * per_page:(p + 1) * per_page]
            fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
            # Each pair = 2 image columns -> total image-cols = COLS*2
            for i, pr in enumerate(chunk):
                row = i // COLS
                col = i % COLS
                base = row * (COLS * 2) + col * 2
                axm = fig.add_subplot(ROWS_PER_PAGE, COLS * 2, base + 1)
                axr = fig.add_subplot(ROWS_PER_PAGE, COLS * 2, base + 2)
                show(axm, pr["mpath"], pr["mtitle"])
                show(axr, pr["rpath"], pr["rtitle"])
            fig.suptitle(
                f"Duplicate cross-match pairs  (page {p+1}/{n_pages})  "
                f"master | RGZ",
                fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  page {p+1}/{n_pages}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
