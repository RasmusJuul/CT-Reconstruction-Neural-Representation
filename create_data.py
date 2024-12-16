import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from tqdm import tqdm
import tifffile
import os
from glob import glob
import pandas as pd
import h5py

from src import _PATH_DATA

def generate_points(num_points, x_range, y_range, min_dist, rng):
    # generate vector points for the starting positions of every fibre.
    points = []
    num_tries = 0
    while len(points) < num_points:
        new_point = np.array(
            [rng.uniform(x_range[0], x_range[1]), rng.uniform(y_range[0], y_range[1])]
        )
        if all(np.linalg.norm(new_point - point) >= min_dist for point in points):
            points.append(new_point)
        num_tries += 1
        if num_tries > 10000:
            print("failed after 10000 tries")
            break

    return np.array(points)


# create a single cylinder. This is called in the below function
def create_cylinder(volume_size, radius, length, orientation, start_point):
    # Initialize the volume with zeros
    volume = np.zeros(volume_size, dtype=int)

    # Ensure orientation is a numpy array of floats
    orientation = np.array(orientation, dtype=float)

    # Ensure start_point is a numpy array of integers
    start_point = np.array(start_point, dtype=int)

    # Ensure radius and length are integers
    radius = int(radius)
    length = int(length)

    # Generate a grid of points
    x = np.arange(volume_size[0])
    y = np.arange(volume_size[1])
    z = np.arange(volume_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Shift grid to align with start_point
    X -= start_point[0]
    Y -= start_point[1]
    Z -= start_point[2]

    # Calculate distance from each point to the axis of the cylinder
    distances = np.sqrt(
        (Y * orientation[2] - Z * orientation[1]) ** 2
        + (Z * orientation[0] - X * orientation[2]) ** 2
        + (X * orientation[1] - Y * orientation[0]) ** 2
    )
    # Set points inside the cylinder to 1
    volume[(distances <= radius)] = 1

    return volume


# create a bundle of fibres by calling the above function in a loop
def create_fibre_bundle(volume_size, radius, length, n_fibres, misalignment, rng):

    nvs = (volume_size[0], volume_size[1], 30)
    volume_out = np.ones(nvs, dtype=np.float32)*0.1
    points_out = generate_points(
        n_fibres,
        (1, volume_size[0]-1),
        (1, volume_size[0]-1),
        radius * 2,
        rng,
    )
    sp1 = points_out[:, 0].reshape(-1, 1)
    sp2 = points_out[:, 1].reshape(-1, 1)

    sp3 = np.zeros((sp2.shape[0], 1))
    start_points = np.concatenate((sp1, sp2, sp3), 1)

    for n1 in range(sp2.shape[0]):
        orientation = np.clip(
            (0, 0, 1) + (rng.rand(3) * misalignment - misalignment / 2), -1, 1
        )
        this_start_points = start_points[n1, :]

        this_cylinder_volume = create_cylinder(
            nvs, radius, length, orientation, this_start_points
        )
        volume_out += this_cylinder_volume
    volume_out = np.clip(volume_out, 0, 1)
    mask = create_circular_mask(volume_out.shape[:2])
    volume_out = apply_mask(volume_out,mask)
    return volume_out

def create_circular_mask(image_shape, center=None, radius=None):
    h, w = image_shape[:2]
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

def apply_mask(image, mask):
    masked_image = image.copy()
    masked_image[~mask] = 0
    return masked_image


def save_file(mode, folder_name, count, i, vol):
    num_subfolders = max(1, int(count / 100))
    subfolder = str(i % num_subfolders).zfill(3)

    tifffile.imwrite(
        f"data/{folder_name}/{mode}/{subfolder}/fiber_{str(i).zfill(5)}.tif",
        vol,
    )


def main(mode, folder_name, count, i, seed):
    rng = np.random.RandomState(seed=seed)
    vol_shape_zxy = np.array([256, 256, 256])
    # vol_shape_yxz = np.flip(vol_shape_zxy)
    # Example usage:
    radius = 4  # in um
    length = 400
    orientation = (0, 0, 1)  # (x,y,z)
    start_point = (0, 0, 0)
    misalignment = 0.01 #rng.rand(1)[0]  # 0-1
    n_fibres = 1000 #rng.randint(75, 125)
    # create
    # if making training data in a loop, start here (remove random seed for different volumes)
    cylinder_volume = create_fibre_bundle(
        vol_shape_zxy, radius, length, n_fibres, misalignment, rng
    )
    cylinder_volume = sp.ndimage.zoom(cylinder_volume, (1, 1, 10), order=1)
    vol = np.ascontiguousarray(np.transpose(cylinder_volume, (2, 1, 0)))
    vol = vol[22:278,:,:]
    save_file(mode, folder_name, count, i, vol)


def create_csv(split):
    folder_name = f"synthetic_fibers"

    file_path = glob(f"{folder_name}/{split}/**/*.tif", root_dir=_PATH_DATA)
    file_path.sort()

    df = pd.DataFrame()

    df["file_path"] = file_path

    df["index"] = df.file_path.apply(lambda x: x.split("/")[-1][4:-4])
    df.set_index("index", inplace=True)

    df.to_csv(f"{_PATH_DATA}/{folder_name}/{split}.csv", index=False, encoding="utf-8")


def create_hdf5_dataset(files, hdf5_path):
    # Open the first image to get the shape
    vol_shape = tifffile.imread(f"{_PATH_DATA}/{files[0]}").shape

    # Create HDF5 file
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Create a dataset in the file
        dataset = hdf5_file.create_dataset(
            "volumes", (len(files), *vol_shape), dtype="float"
        )

        # Loop through all images and save them to the dataset
        for i, file in tqdm(
            enumerate(files), unit="vol", desc="Saving volumes to hdf5 file"
        ):
            vol = tifffile.imread(f"{_PATH_DATA}/{file}")
            # vol -= vol.min()
            # vol = vol / vol.max()
            vol = vol.transpose(2, 1, 0)
            dataset[i] = vol


if __name__ == "__main__":
    folder_name = "synthetic_fibers"

    os.makedirs(f"{_PATH_DATA}/{folder_name}/train", exist_ok=True)
    os.makedirs(f"{_PATH_DATA}/{folder_name}/test", exist_ok=True)
    # os.makedirs(f"{_PATH_DATA}/{folder_name}/validation", exist_ok=True)

    count_train = 100
    count_test = 10
    # count_validation = 2

    for i in range(max(1, int(count_train / 100))):
        os.makedirs(
            f"{_PATH_DATA}/{folder_name}/train/{str(i).zfill(3)}", exist_ok=True
        )
    for i in range(max(1, int(count_test / 100))):
        os.makedirs(f"{_PATH_DATA}/{folder_name}/test/{str(i).zfill(3)}", exist_ok=True)
    # for i in range(max(1,int(count_validation / 100))):
    #     os.makedirs(f"data/{folder_name}/validation/{str(i).zfill(3)}", exist_ok=True)

    rng_train = np.random.SeedSequence(42).generate_state(count_train)
    rng_test = np.random.SeedSequence(1337).generate_state(count_test)
    # rng_validation = np.random.SeedSequence(1997).generate_state(count_validation)

    Parallel(n_jobs=-1)(
        delayed(main)("train", folder_name, count_train, i, rng_train[i])
        for i in tqdm(
            range(count_train), unit="image", desc="creating training fiber volumes"
        )
    )
    create_csv("train")
    files = pd.read_csv(
        f"{_PATH_DATA}/{folder_name}/train.csv", header=0
    ).file_path.to_list()
    create_hdf5_dataset(files, f"{_PATH_DATA}/{folder_name}/train.hdf5")
    Parallel(n_jobs=-1)(
        delayed(main)("test", folder_name, count_test, i, rng_test[i])
        for i in tqdm(
            range(count_test), unit="image", desc="creating testing fiber volumes"
        )
    )
    create_csv("test")
    files = pd.read_csv(
        f"{_PATH_DATA}/{folder_name}/test.csv", header=0
    ).file_path.to_list()
    create_hdf5_dataset(files, f"{_PATH_DATA}/{folder_name}/test.hdf5")
    # Parallel(n_jobs=-1)(delayed(main)
    #                     ("validation",folder_name,count_validation,i,rng_validation[i])
    #                     for i in tqdm(range(count_validation), unit="image", desc="creating validation fiber volumes"))
    # create_csv("validation")
