import os
import qim3d
import shutil
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage import io
import scipy.ndimage as ndi
from tifffile import imwrite,imread
from joblib import Parallel, delayed


from src import _PATH_DATA


def create_mask(im, intensity_threshold, iteration):
    h, _, _ = ndi.center_of_mass(im)
    h = int(h)

    bottom = 0
    for i in range(max(h - (20 + iteration), 0), 0, -1):
        if im[i, :, :].max() < intensity_threshold:
            bottom = i
            break

    top = im.shape[0]
    for i in range(min(h + (20 + iteration), im.shape[0]), im.shape[0]):
        if im[i, :, :].max() < intensity_threshold:
            top = i
            break

    mask = np.zeros(im.shape)
    mask[bottom:top, :, :] = 1

    im_avg = ndi.convolve(im, np.ones((3, 3, 3)) / (3**3))

    if im[mask == 1].max() < im_avg.max():
        im[mask == 1] = 0
        return create_mask(im, intensity_threshold, iteration + 1)
    return mask


def cut(im):
    mask = create_mask(im, 70, 0)
    im[mask == 0] = 0
    im[im <= 40] = 0

    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)

    out = ndi.rotate(im, angle1, axes=(1, 0))
    out = ndi.rotate(out, angle2, axes=(1, 2))
    out = ndi.rotate(out, angle3, axes=(0, 2))

    ax2_top = out.shape[2]
    ax2_bottom = 0
    ax1_top = out.shape[1]
    ax1_bottom = 0
    ax0_top = out.shape[0]
    ax0_bottom = 0

    for i in range(out.shape[2]):
        if np.any(out[:, :, i] >= 75):
            ax2_bottom = i
            break
    for i in range(out.shape[2] - 1, -1, -1):
        if np.any(out[:, :, i] >= 75):
            ax2_top = i
            break
    for i in range(out.shape[1]):
        if np.any(out[:, i, :] >= 75):
            ax1_bottom = i
            break
    for i in range(out.shape[1] - 1, -1, -1):
        if np.any(out[:, i, :] >= 75):
            ax1_top = i
            break
    for i in range(out.shape[0]):
        if np.any(out[i, :, :] >= 75):
            ax0_bottom = i
            break
    for i in range(out.shape[0] - 1, -1, -1):
        if np.any(out[i, :, :] >= 75):
            ax0_top = i
            break

    cut = out[ax0_bottom:ax0_top, ax1_bottom:ax1_top, ax2_bottom:ax2_top]
    return cut

    
def main(image_path):
    im = qim3d.io.load(image_path)
    vol = cut(cut(im))
    
    # temp = np.zeros((256,256,256))
    # if vol.shape[0]//2 * 2 != vol.shape[0]:
    #     x_start = temp.shape[0]//2 - vol.shape[0]//2
    #     x_end = temp.shape[0]//2 + vol.shape[0]//2+1
    # else:
    #     x_start = temp.shape[0]//2 - vol.shape[0]//2
    #     x_end = temp.shape[0]//2 + vol.shape[0]//2
    
    # if vol.shape[1]//2 * 2 != vol.shape[1]:
    #     y_start = temp.shape[1]//2 - vol.shape[1]//2
    #     y_end = temp.shape[1]//2 + vol.shape[1]//2+1
    # else:
    #     y_start = temp.shape[1]//2 - vol.shape[1]//2
    #     y_end = temp.shape[1]//2 + vol.shape[1]//2
    
    # if vol.shape[2]//2 * 2 != vol.shape[2]:
    #     z_start = temp.shape[2]//2 - vol.shape[2]//2
    #     z_end = temp.shape[2]//2 + vol.shape[2]//2+1
    # else:
    #     z_start = temp.shape[2]//2 - vol.shape[2]//2
    #     z_end = temp.shape[2]//2 + vol.shape[2]//2
    
    # temp[x_start:x_end,y_start:y_end,z_start:z_end] = vol
    # vol = temp
    
    temp = image_path.split("/")
    temp[-2] = temp[-2] + "_clean"
    new_filename = "/".join(temp)
    imwrite(new_filename, vol)


if __name__ == "__main__":
    os.makedirs(f"data/bugnist_256/SL_clean/", exist_ok=True)

    np.random.seed(42)
    
    image_paths = glob("data/bugnist_256/SL/*.tif")
    Parallel(n_jobs=-1)(
        delayed(main)(image_path)
        for image_path in tqdm(image_paths, unit="image", desc="cutting images")
    )

    # for size in ["256", "128"]:
    #     image_paths = glob.glob(f"../../data/bugnist_{size}/**/*.tif")
    #     Parallel(n_jobs=-1)(
    #         delayed(cut)(image_path)
    #         for image_path in tqdm(image_paths, unit="image", desc="cutting images")
    #     )

    files = glob(f"{_PATH_DATA}/bugnist_256/SL_clean/*.tif")
    temp = {}
    for file in files:
        temp["/".join(file.split("/")[-4:])] = imread(file).shape
    df = pd.DataFrame(data={"file_name":files})
    df["file_name"] = df.file_name.apply(lambda x: "/".join(x.split("/")[-4:]))
    df["size"] = df.file_name.apply(lambda x: temp[x])
    df.to_csv("cut_sizes.csv",index=False)
