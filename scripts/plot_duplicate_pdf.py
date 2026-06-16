#!/usr/bin/env python3
"""
General visual-inspection PDF for a duplicates CSV (reusable).

Takes a duplicates CSV produced by crossmatch_catalogs.py and lays each matched
pair side by side -- the SOURCE cutout next to its MATCH cutout -- 3 pairs across
and 4 down per A4-landscape page, into a multi-page PDF.

It works for both cases:

  * cross-match (two catalogs): give --img-a and --img-b (one folder each).
  * self-match  (one catalog) : give --self and a single --img-a; both panels
                                are pulled from that same folder.

Filenames are built from CSV columns via a template (--pattern). For the MATCH
panel the template defaults to the same one with every {col} -> {match_col}, so a
self-match of a catalog whose pngs are "{widx}_{source_catalog}.png" needs only:

    --pattern "{widx}_{source_catalog}"

If the on-disk filenames carry a leading index that is NOT in the CSV (e.g.
"{idx}_{RGZID}.png" where idx is unknown), pass --scan-a / --scan-b: the folder
is scanned once, each filename's leading "<number>_" is stripped, and the
template result is matched against the remainder.

Rows whose SOURCE or MATCH cutout cannot be found are skipped and counted.

Usage:
    # self-match of the master catalog (the immediate use case)
    python plot_duplicates_pdf.py \\
        --csv  catalogs/master_radio_catalog_deduplicate_removed_duplicates.csv \\
        --self --img-a "$RGC_DATA/master_radio_catalog/png" \\
        --pattern "{widx}_{source_catalog}"

    # cross-match between two catalogs with different png folders/patterns
    python plot_duplicates_pdf.py --csv X_duplicates.csv \\
        --img-a A/png --pattern        "{widx}_{source_catalog}" \\
        --img-b B/png --pattern-match  "{match_RGZID}" --scan-b

    python plot_duplicates_pdf.py ... --limit 60      # quick preview
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

COLS = 3            # pairs across the page
ROWS_PER_PAGE = 4   # pairs down the page  -> 12 pairs / page


def derive_match_pattern(pattern):
    """Turn '{widx}_{source_catalog}' into '{match_widx}_{match_source_catalog}'."""
    return re.sub(r"\{([^}]+)\}", r"{match_\1}", pattern)


def build_scan_lookup(folder):
    """key -> filename, where key is the filename minus a leading '<number>_'."""
    lut = {}
    for f in os.listdir(folder):
        if not f.lower().endswith(".png"):
            continue
        base = f[:-4]
        # strip a single leading "<digits>_" prefix if present
        key = re.sub(r"^\d+_", "", base)
        lut[key] = f
    return lut


def resolve_path(row, pattern, folder, scan_lut):
    """Format the template against the row and return an existing png path or None."""
    try:
        key = pattern.format(**row)
    except KeyError:
        return None
    if scan_lut is not None:
        fname = scan_lut.get(key)
        return os.path.join(folder, fname) if fname else None
    path = os.path.join(folder, key + ".png")
    return path if os.path.exists(path) else None


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="duplicates CSV")
    ap.add_argument("--img-a", dest="img_a", required=True,
                    help="folder of SOURCE-side cutouts")
    ap.add_argument("--img-b", dest="img_b", default=None,
                    help="folder of MATCH-side cutouts (omit with --self)")
    ap.add_argument("--self", dest="self_match", action="store_true",
                    help="self-match: MATCH panel uses --img-a too")
    ap.add_argument("--pattern", default="{widx}_{source_catalog}",
                    help="filename template (no .png) for the SOURCE panel, "
                         "using CSV column names (default %(default)s)")
    ap.add_argument("--pattern-match", dest="pattern_match", default=None,
                    help="template for the MATCH panel "
                         "(default: --pattern with every {col} -> {match_col})")
    ap.add_argument("--scan-a", dest="scan_a", action="store_true",
                    help="resolve SOURCE filenames by scanning --img-a and "
                         "stripping a leading '<number>_' prefix")
    ap.add_argument("--scan-b", dest="scan_b", action="store_true",
                    help="same as --scan-a but for the MATCH folder")
    ap.add_argument("--label-a", dest="label_a", default=None,
                    help="header label for the SOURCE panel")
    ap.add_argument("--label-b", dest="label_b", default=None,
                    help="header label for the MATCH panel")
    ap.add_argument("--out", default=None,
                    help="output PDF (default: <csv>_pairs_inspection.pdf)")
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    args = ap.parse_args()

    img_b = args.img_a if args.self_match else args.img_b
    if img_b is None:
        ap.error("give --img-b, or use --self to reuse --img-a for both panels")

    pattern_b = args.pattern_match or derive_match_pattern(args.pattern)
    label_a = args.label_a or ("source" if args.self_match else "A")
    label_b = args.label_b or ("duplicate" if args.self_match else "B")

    df = pd.read_csv(args.csv)
    lut_a = build_scan_lookup(args.img_a) if args.scan_a else None
    lut_b = build_scan_lookup(img_b) if args.scan_b else None

    pairs, missing = [], 0
    for _, r in df.iterrows():
        row = r.to_dict()
        pa = resolve_path(row, args.pattern, args.img_a, lut_a)
        pb = resolve_path(row, pattern_b, img_b, lut_b)
        if pa is None or pb is None:
            missing += 1
            continue
        sep = row.get("sep_arcsec", float("nan"))
        try:
            sep_s = f"sep={float(sep):.2f}\""
        except (TypeError, ValueError):
            sep_s = ""
        pairs.append({
            "pa": pa, "pb": pb,
            "ta": f"{label_a}  {args.pattern.format(**row)}",
            "tb": f"{label_b}  {pattern_b.format(**row)}\n{sep_s}",
        })

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"pairs with both images : {len(pairs)}")
    print(f"rows skipped (missing png): {missing}")
    if not pairs:
        print("nothing to plot.")
        return

    out = args.out or re.sub(r"(_duplicates)?\.csv$", "", args.csv) + \
        "_pairs_inspection.pdf"

    per_page = COLS * ROWS_PER_PAGE
    n_pages = (len(pairs) + per_page - 1) // per_page

    with PdfPages(out) as pdf:
        for p in range(n_pages):
            chunk = pairs[p * per_page:(p + 1) * per_page]
            fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
            for i, pr in enumerate(chunk):
                row_i, col_i = divmod(i, COLS)
                base = row_i * (COLS * 2) + col_i * 2
                axa = fig.add_subplot(ROWS_PER_PAGE, COLS * 2, base + 1)
                axb = fig.add_subplot(ROWS_PER_PAGE, COLS * 2, base + 2)
                show(axa, pr["pa"], pr["ta"])
                show(axb, pr["pb"], pr["tb"])
            fig.suptitle(
                f"Duplicate pairs  (page {p+1}/{n_pages})  "
                f"{label_a} | {label_b}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  page {p+1}/{n_pages}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
