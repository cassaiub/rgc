"""
Download FIRST cutouts for duplicate-source pairs and plot them side by side
so they can be visually checked as the same object or not.

Reuses the SkyView download approach from download.py.
Run:  python compare_duplicates.py
"""
from pathlib import Path

from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_PAIRS = 20
PIXELS = (150, 150)
SURVEY = "VLA FIRST (1.4 GHz)"

REPO = Path(__file__).resolve().parents[1]
dup_csv = REPO / "catalogs" / "DUPLICATES_report.csv"
out_dir = REPO / "duplicate_pair_check"
fits_dir = out_dir / "fits"
out_dir.mkdir(parents=True, exist_ok=True)
fits_dir.mkdir(parents=True, exist_ok=True)


def celestial_capture(coord: SkyCoord, filename: Path) -> None:
    image = SkyView.get_images(
        position=coord, survey=SURVEY, coordinates="J2000", pixels=PIXELS
    )[0]
    filename.parent.mkdir(parents=True, exist_ok=True)
    image.writeto(filename, overwrite=True, output_verify="ignore")


def load_image(coord: SkyCoord, filename: Path):
    """Download (cached) and return a 2D array, or None on failure."""
    try:
        if not filename.exists():
            celestial_capture(coord, filename)
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        data = np.reshape(np.asarray(data, dtype=float),
                          (header["NAXIS2"], header["NAXIS1"]))
        data[np.isnan(data)] = np.nanmin(data)
        return data
    except Exception as e:
        print(f"  [FAILED] {filename.name} | {e}")
        return None


def stretch(data):
    """Percentile clip for display contrast."""
    if data is None:
        return None
    lo, hi = np.nanpercentile(data, [1, 99.5])
    if hi <= lo:
        lo, hi = np.nanmin(data), np.nanmax(data)
    return np.clip((data - lo) / (hi - lo + 1e-12), 0, 1)


# --- select 20 duplicate groups of size 2 (clean pairs) ---
dup = pd.read_csv(dup_csv)
pair_groups = [g for _, g in dup.groupby("dup_group") if len(g) == 2]
pair_groups = pair_groups[:N_PAIRS]
print(f"Selected {len(pair_groups)} pairs for comparison")

log_rows = []
for k, g in enumerate(pair_groups, 1):
    g = g.reset_index(drop=True)
    a, b = g.iloc[0], g.iloc[1]
    gid = int(a["dup_group"])
    print(f"[{k:02d}/{len(pair_groups)}] group {gid}: {a['Catalog']} vs {b['Catalog']}")

    ca = SkyCoord(ra=a["RA"] * u.deg, dec=a["Dec"] * u.deg, frame="icrs")
    cb = SkyCoord(ra=b["RA"] * u.deg, dec=b["Dec"] * u.deg, frame="icrs")
    sep = ca.separation(cb).arcsec

    img_a = load_image(ca, fits_dir / f"g{gid}_a_{a['Catalog']}.fits")
    img_b = load_image(cb, fits_dir / f"g{gid}_b_{b['Catalog']}.fits")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6))
    for ax, img, row in zip(axes, [img_a, img_b], [a, b]):
        if img is not None:
            ax.imshow(stretch(img), origin="lower", cmap="inferno")
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        zval = "nan" if pd.isna(row["z"]) else f"{row['z']:.4f}"
        fval = "nan" if pd.isna(row["Flux_FIRST_1400MHz"]) else f"{row['Flux_FIRST_1400MHz']:.1f}"
        ax.set_title(f"{row['Catalog']}  {row['FIRST_ID']}\n"
                     f"RA={row['RA']:.5f} Dec={row['Dec']:.5f}\n"
                     f"z={zval}  S_FIRST={fval} mJy", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Group {gid}  |  separation = {sep:.2f}\"", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile = out_dir / f"pair_{k:02d}_group{gid}.png"
    fig.savefig(outfile, dpi=110)
    plt.close(fig)

    log_rows.append({
        "pair": k, "dup_group": gid, "sep_arcsec": round(float(sep), 3),
        "catA": a["Catalog"], "catB": b["Catalog"],
        "FIRST_ID_A": a["FIRST_ID"], "FIRST_ID_B": b["FIRST_ID"],
        "zA": a["z"], "zB": b["z"],
        "img": outfile.name,
        "downloaded": (img_a is not None) and (img_b is not None),
    })

pd.DataFrame(log_rows).to_csv(out_dir / "pair_check_log.csv", index=False)

# --- contact sheet: all pairs on one page ---
ok = [r for r in log_rows if r["downloaded"]]
n = len(pair_groups)
fig, axes = plt.subplots(n, 2, figsize=(6, 2.6 * n))
if n == 1:
    axes = axes[None, :]
for ax_row, g in zip(axes, pair_groups):
    g = g.reset_index(drop=True)
    for ax, (_, row) in zip(ax_row, g.iterrows()):
        gid = int(row["dup_group"])
        side = "a" if row.name == 0 else "b"
        fp = fits_dir / f"g{gid}_{side}_{row['Catalog']}.fits"
        try:
            data = fits.getdata(fp)
            h = fits.getheader(fp)
            data = np.reshape(np.asarray(data, float), (h["NAXIS2"], h["NAXIS1"]))
            ax.imshow(stretch(data), origin="lower", cmap="inferno")
        except Exception:
            ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"g{gid} {row['Catalog']}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(out_dir / "contact_sheet.png", dpi=110)
plt.close(fig)

print(f"\nDone. {len(ok)}/{n} pairs downloaded successfully.")
print(f"Side-by-side PNGs + contact_sheet.png + pair_check_log.csv in: {out_dir}")
