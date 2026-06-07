from astropy.io import fits
import numpy as np
from PIL import Image
import os
import tqdm
input_folder_path = "./20k_images/rgz20k_fits_raw"
output_folder_path = "./20k_images/rgz20k_png_raw"

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

os.makedirs(output_folder_path, exist_ok=True)

for file in tqdm.tqdm(os.listdir(input_folder_path)):
    if file.endswith(".fits"):
        fits_to_png(os.path.join(input_folder_path, file), os.path.join(output_folder_path, file.replace(".fits", ".png")))

print("Conversion complete")