from pathlib import Path
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import pandas as pd
import astropy.units as u
import argparse


def celestial_capture(survey: str, coord: SkyCoord, filename: str, pixels=(150, 150)) -> None:
    """
    Capture a celestial image using the SkyView service.
    Saves FITS files safely even if header contains unprintable characters.
    """
    # Download image from SkyView
    image = SkyView.get_images(position=coord, survey=survey, coordinates="J2000", pixels=pixels)[0]

    # Ensure output directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Write FITS ignoring header verification errors
    image.writeto(filename, overwrite=True, output_verify="ignore")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Download FITS images from SkyView")
parser.add_argument("--limit", type=int, default=None, 
                    help="Limit number of files to download (for testing)")
parser.add_argument("--start", type=int, default=0,
                    help="Start from this index (for resuming)")
args = parser.parse_args()

# Load catalog
script_dir = Path(__file__).parent
csv_path = script_dir / "DR1_FIRST_radio_classifications.csv"
lao = pd.read_csv(csv_path)

# Apply limit if specified
if args.limit:
    lao = lao.iloc[args.start:args.start + args.limit]
    print(f"Limited to {len(lao)} files (starting from index {args.start})")

# Build SkyCoord (RA/DEC in degrees)
coords = SkyCoord(
    ra=lao["RA"].values * u.deg,
    dec=lao["Dec"].values * u.deg,
    frame="icrs",
)

out_dir = script_dir / "DR1_FIRST_fits"
log_path = script_dir / "download_DR1_FIRST.log"

# Ensure output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

with log_path.open("w") as log_f:
    for idx, (coord, row) in enumerate(zip(coords, lao.itertuples(index=False))):
        radio_name = row.RGZID

        # Build filename using RGZID exactly as in the catalog
        base_name = f"{radio_name}"
        fname = out_dir / f"{base_name}.fits"

        # Skip if file already exists
        if fname.exists():
            status = "SKIPPED"
            error_msg = "File already exists"
            print(f"[SKIPPED] {base_name} (already exists)")
        else:
            try:
                celestial_capture(
                    survey="VLA FIRST (1.4 GHz)", 
                    coord=coord,
                    filename=str(fname),
                )
                status = "OK"
                error_msg = ""
                print(f"[OK] {base_name}")

            except Exception as e:
                status = "FAILED"
                error_msg = str(e)
                print(f"[FAILED] {base_name} | {error_msg}")

        # Log: file name - status - error
        log_f.write(f"{base_name}.fits - {status} - {error_msg}\n")
        log_f.flush()
