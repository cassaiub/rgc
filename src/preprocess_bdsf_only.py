#!/usr/bin/env python3

import os
import bdsf
import numpy as np
from pathlib import Path
from astropy.io import fits
from PIL import Image


# ==========================================================
# ---------------- FITS → PNG ------------------------------
# ==========================================================

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


# ==========================================================
# ---------------- APPLY PNG MASK --------------------------
# ==========================================================

def apply_mask_png(image_path: str, mask_path: str, output_path: str):
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image)
    mask_array = np.array(mask)

    if image_array.shape != mask_array.shape:
        print(f"Shape mismatch: {image_path}")
        return

    mask_binary = (mask_array > 0).astype(np.uint8)
    masked = np.where(mask_binary == 1, image_array, 0)

    Image.fromarray(masked.astype(np.uint8), mode="L").save(output_path)


# ==========================================================
# ---------------- MAIN PIPELINE ---------------------------
# ==========================================================

def process_dataset(
    raw_fits_root: str,
    raw_png_root: str,
    spurious_mask_fits_root: str,
    no_spurious_mask_fits_root: str,
    spurious_mask_png_root: str,
    no_spurious_mask_png_root: str,
    spurious_masked_png_root: str,
    no_spurious_masked_png_root: str,
    freq: float,
    beam: tuple,
    threshold_island: int,
    threshold_pixel: int,
    rms_box: tuple,
    atrous_do: bool,
    rms_map: bool,
    mean_map: str,
    flag_maxsize_bm: int,
    neighbor_pixel_dist: int,
):

    for folder in os.listdir(raw_fits_root):

        raw_fits_folder = os.path.join(raw_fits_root, folder)
        if not os.path.isdir(raw_fits_folder):
            continue

        print(f"\n========== Processing folder: {folder} ==========")

        # ---- Create all subfolders ----
        raw_png_folder = os.path.join(raw_png_root, folder)
        spurious_mask_fits_folder = os.path.join(spurious_mask_fits_root, folder)
        no_spurious_mask_fits_folder = os.path.join(no_spurious_mask_fits_root, folder)
        spurious_mask_png_folder = os.path.join(spurious_mask_png_root, folder)
        no_spurious_mask_png_folder = os.path.join(no_spurious_mask_png_root, folder)
        spurious_masked_png_folder = os.path.join(spurious_masked_png_root, folder)
        no_spurious_masked_png_folder = os.path.join(no_spurious_masked_png_root, folder)

        for p in [
            raw_png_folder,
            spurious_mask_fits_folder,
            no_spurious_mask_fits_folder,
            spurious_mask_png_folder,
            no_spurious_mask_png_folder,
            spurious_masked_png_folder,
            no_spurious_masked_png_folder,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)

        # ======================================================
        # Process each FITS file
        # ======================================================

        for image_file in os.listdir(raw_fits_folder):

            if not image_file.endswith(".fits"):
                continue

            print(f"Processing: {image_file}")

            image_path = os.path.join(raw_fits_folder, image_file)
            stem = Path(image_file).stem

            try:
                # ----------------------------------------------
                # 1️⃣ Run PyBDSF ONCE
                # ----------------------------------------------
                image = bdsf.process_image(
                    image_path,
                    frequency=freq,
                    beam=beam,
                    thresh_isl=threshold_island,
                    thresh_pix=threshold_pixel,
                    rms_box=rms_box,
                    atrous_do=atrous_do,
                    rms_map=rms_map,
                    mean_map=mean_map,
                    flag_maxsize_bm=flag_maxsize_bm,
                )

                # ----------------------------------------------
                # 2 Export SPURIOUS mask (all islands)
                # ----------------------------------------------
                spurious_mask_fits = os.path.join(
                    spurious_mask_fits_folder,
                    stem + "_mask.fits"
                )

                image.export_image(
                    img_type="island_mask",
                    outfile=spurious_mask_fits,
                    clobber=True,
                    mask_dilation=1,
                )

                # ----------------------------------------------
                # 3 Find central island
                # ----------------------------------------------
                island_positions = []

                for isl in image.islands:
                    island_id = isl.island_id
                    mask_active = isl.mask_active
                    y_slice, x_slice = isl.bbox

                    if mask_active.shape == image.ch0_arr.shape:
                        mask_active = mask_active[y_slice, x_slice]

                    coords = np.argwhere(~mask_active.astype(bool))
                    if coords.size == 0:
                        continue

                    y0, x0 = np.median(coords, axis=0)
                    xpos, ypos = x0 + x_slice.start, y0 + y_slice.start

                    island_positions.append((island_id, xpos, ypos))

                if len(island_positions) == 0:
                    continue

                image_center = np.array([
                    image.ch0_arr.shape[1] / 2,
                    image.ch0_arr.shape[0] / 2
                ])

                distances = [
                    (iid, np.linalg.norm(np.array([x, y]) - image_center))
                    for iid, x, y in island_positions
                ]

                central_island_id = min(distances, key=lambda x: x[1])[0]

                # ----------------------------------------------
                # 4️⃣ Determine neighbors
                # ----------------------------------------------
                neighbor_dict = {}
                for id1, x1, y1 in island_positions:
                    neighbors = []
                    for id2, x2, y2 in island_positions:
                        if id1 == id2:
                            continue
                        if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= neighbor_pixel_dist:
                            neighbors.append(id2)
                    neighbor_dict[id1] = neighbors

                central_neighbors = neighbor_dict.get(central_island_id, [])

                # ----------------------------------------------
                # 5 Build NON-SPURIOUS mask
                # ----------------------------------------------
                original_pyRank = image.pyrank.copy()
                keep_mask = np.zeros_like(original_pyRank, dtype=bool)

                for isl in image.islands:

                    if isl.island_id != central_island_id and \
                       isl.island_id not in central_neighbors:
                        continue

                    y_slice, x_slice = isl.bbox
                    mask_active = isl.mask_active

                    if mask_active.shape == image.ch0_arr.shape:
                        mask_active = mask_active[y_slice, x_slice]

                    keep_mask[y_slice, x_slice] |= (~mask_active).astype(bool)

                image.pyrank = np.where(
                    keep_mask,
                    original_pyRank,
                    -1
                ).astype(original_pyRank.dtype)

                no_spurious_mask_fits = os.path.join(
                    no_spurious_mask_fits_folder,
                    stem + "_mask.fits"
                )

                image.export_image(
                    img_type="island_mask",
                    outfile=no_spurious_mask_fits,
                    clobber=True,
                    mask_dilation=1,
                )

                image.pyrank = original_pyRank

                # ----------------------------------------------
                # 6 Convert RAW FITS → PNG
                # ----------------------------------------------
                raw_png_path = os.path.join(raw_png_folder, stem + ".png")
                fits_to_png(image_path, raw_png_path)

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue

        # ======================================================
        # Convert masks FITS → PNG
        # ======================================================

        for mask_file in os.listdir(spurious_mask_fits_folder):
            if mask_file.endswith(".fits"):
                fits_to_png(
                    os.path.join(spurious_mask_fits_folder, mask_file),
                    os.path.join(spurious_mask_png_folder, mask_file.replace(".fits", ".png"))
                )

        for mask_file in os.listdir(no_spurious_mask_fits_folder):
            if mask_file.endswith(".fits"):
                fits_to_png(
                    os.path.join(no_spurious_mask_fits_folder, mask_file),
                    os.path.join(no_spurious_mask_png_folder, mask_file.replace(".fits", ".png"))
                )

        # ======================================================
        # Apply masks
        # ======================================================

        for image_file in os.listdir(raw_png_folder):

            if not image_file.endswith(".png"):
                continue

            stem = Path(image_file).stem
            raw_png_path = os.path.join(raw_png_folder, image_file)

            spurious_mask_png = os.path.join(spurious_mask_png_folder, stem + "_mask.png")
            no_spurious_mask_png = os.path.join(no_spurious_mask_png_folder, stem + "_mask.png")

            if os.path.exists(spurious_mask_png):
                apply_mask_png(
                    raw_png_path,
                    spurious_mask_png,
                    os.path.join(spurious_masked_png_folder, stem + "_masked.png")
                )

            if os.path.exists(no_spurious_mask_png):
                apply_mask_png(
                    raw_png_path,
                    no_spurious_mask_png,
                    os.path.join(no_spurious_masked_png_folder, stem + "_masked.png")
                )


# ==========================================================
# ---------------- RUN -------------------------------------
# ==========================================================

if __name__ == "__main__":

    root_dataset_path = "/media/plato/shahal/final_rgc/newdata_set/new_dataset/"

    process_dataset(
        raw_fits_root=os.path.join(root_dataset_path, "raw_fits"),
        raw_png_root=os.path.join(root_dataset_path, "raw_png"),
        spurious_mask_fits_root=os.path.join(root_dataset_path, "spurious_mask_fits"),
        no_spurious_mask_fits_root=os.path.join(root_dataset_path, "no_spurious_mask_fits"),
        spurious_mask_png_root=os.path.join(root_dataset_path, "spurious_mask_png"),
        no_spurious_mask_png_root=os.path.join(root_dataset_path, "no_spurious_mask_png"),
        spurious_masked_png_root=os.path.join(root_dataset_path, "spurious_masked_png"),
        no_spurious_masked_png_root=os.path.join(root_dataset_path, "no_spurious_masked_png"),
        beam=(5.4/3600, 5.4/3600, 0.0),
        freq=1.4e9,
        threshold_island=3,
        threshold_pixel=5,
        rms_box=(25, 8),
        atrous_do=True,
        rms_map=False,
        mean_map="zero",
        flag_maxsize_bm=50,
        neighbor_pixel_dist=30,
    )