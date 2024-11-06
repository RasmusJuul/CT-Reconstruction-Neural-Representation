# -*- coding: utf-8 -*-
import os
import math
import torch
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from tifffile import tifffile
from torch.utils.data import DataLoader

from src import _PATH_DATA


def collate_fn(batch):
    points = batch[0]
    targets = batch[1]
    return points, targets, None


def collate_fn_imagefit(batch):
    points = batch[0]
    targets = batch[1]
    img_idxs = batch[2]
    return points, targets, img_idxs


def collate_fn_raygan(batch):
    points = batch[0]
    targets = batch[1]
    position = batch[2]
    start_points = batch[3]
    end_points = batch[4]
    real_ray = batch[5]
    real_position = batch[6]
    real_start_points = batch[7]
    real_end_points = batch[8]
    return points, targets, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points

class Geometry(torch.nn.Module):
    def __init__(
        self,
        source_pos,
        detector_pos,
        detector_size,
        detector_pixel_size,
        object_shape,
        beam_type="cone",
    ):
        """
        Parameters
        ----------
        source_pos : torch.Tensor
            ray source position x,y,z position in relation to sample
        detector_pos : torch.Tensor
            center of detector x,y,z position in relation to sample
        detector_size : Sequence[int]
            Shape of detector in pixels e.g. (300,300)
        detector_pixel_size : Sequence[torch.Tensor]
            Two vectors defining the size of the pixels
            u : the vector from detector pixel (0,0) to (0,1)
            v : the vector from detector pixel (0,0) to (1,0)
        object_shape: Sequence[int]
            shape of object in um, i.e. an object which has the dimensions 200x200x300um has the shape [200,200,300]
        beam_type : str
            Which beam type to use for calculating projection. Default 'cone'
        """
        super(Geometry, self).__init__()
        self.source_pos = source_pos
        self.detector_pos = detector_pos
        self.detector_size = detector_size
        self.detector_pixel_size = detector_pixel_size
        self.u_vec = detector_pixel_size[:3]
        self.v_vec = detector_pixel_size[3:]
        self.beam_type = beam_type
        self.object_shape = object_shape

        # Convert into relative coordinates
        self.source_pos[0] /= self.object_shape[0] / 2
        self.source_pos[1] /= self.object_shape[1] / 2
        self.source_pos[2] /= self.object_shape[2] / 2

        self.detector_pos[0] /= self.object_shape[0] / 2
        self.detector_pos[1] /= self.object_shape[1] / 2
        self.detector_pos[2] /= self.object_shape[2] / 2

        self.u_vec[0] /= self.object_shape[0] / 2
        self.u_vec[1] /= self.object_shape[1] / 2
        self.u_vec[2] /= self.object_shape[2] / 2

        self.v_vec[0] /= self.object_shape[0] / 2
        self.v_vec[1] /= self.object_shape[1] / 2
        self.v_vec[2] /= self.object_shape[2] / 2

        self.detector_pixel_coordinates = self.create_grid(
            self.detector_pos,
            self.u_vec,
            self.v_vec,
            self.detector_size[0],
            self.detector_size[1],
        )

        self.rays = self.detector_pixel_coordinates - self.source_pos

        self.start_points, self.end_points, self.valid_rays = self.intersect_cube(
            self.source_pos.repeat(self.detector_size[0] * self.detector_size[1]).view(
                -1, 3
            ),
            self.rays.view(-1, 3),
        )

        self.start_points = self.start_points[self.valid_rays]
        self.end_points = self.end_points[self.valid_rays]

    def intersect_cube(self, ray_origins, ray_directions):
        """
        Calculate the intersection points of a ray with a cube in 3D space.

        This function assumes that the cube is centered at the origin and has a side length of 2 (from -1 to 1 on all axes).
        The rays are defined by starting points and directions stored in PyTorch tensors.

        Parameters:
        ray_origins (torch.Tensor): A tensor of shape (N, 3) where N is the number of rays, and each ray is defined by its origin (x, y, z).
        ray_directions (torch.Tensor): A tensor of shape (N, 3) where N is the number of rays, and each ray is defined by its direction (dx, dy, dz).

        Returns:
        entry_points (torch.Tensor): A tensor of shape (N, 3) representing the entry points of the rays on the cube.
        exit_points (torch.Tensor): A tensor of shape (N, 3) representing the exit points of the rays from the cube.
        valid_rays (torch.Tensor): A tensor of shape (N,) where each element is a boolean indicating whether the corresponding ray intersects the cube.
        """
        # Define the cube boundaries
        cube_min = -1
        cube_max = 1

        # Calculate the intersection t value for each axis
        t_min = (cube_min - ray_origins) / ray_directions
        t_max = (cube_max - ray_origins) / ray_directions

        # Reorder t_min and t_max for each axis
        t_min, t_max = torch.where(t_min > t_max, t_max, t_min), torch.where(
            t_min > t_max, t_min, t_max
        )

        # Get the maximum of t_min and the minimum of t_max
        t_near = torch.max(t_min, dim=1).values
        t_far = torch.min(t_max, dim=1).values

        # Create a tensor to store whether each ray intersects the cube
        valid_rays = t_near < t_far
        valid_rays = valid_rays & (t_far >= cube_min)

        # Calculate the intersection points for the valid rays
        entry_points = ray_origins + t_near.unsqueeze(-1) * ray_directions
        exit_points = ray_origins + t_far.unsqueeze(-1) * ray_directions

        # Return the intersection points and the valid rays tensor
        return entry_points, exit_points, valid_rays

    def create_grid(self, detector_pos, u_vec, v_vec, u_size, v_size):
        # Initialize the grid
        detector_pixels = torch.zeros((u_size, v_size, 3))

        # Calculate the starting point of the grid
        start_pos = (
            detector_pos
            - (u_size // 2) * u_vec
            - (v_size // 2) * v_vec
            + u_vec / 2
            + v_vec / 2
        )

        # Create ranges for u and v
        u_range = torch.arange(u_size).view(-1, 1, 1)
        v_range = torch.arange(v_size).view(1, -1, 1)

        # Fill the grid using broadcasting and vectorized operations
        detector_pixels = start_pos + u_range * u_vec + v_range * v_vec

        return detector_pixels


class CTpoints(torch.utils.data.Dataset):
    def __init__(self, args_dict, noisy_points=False):

        self.args = args_dict
        data_path = f"{_PATH_DATA}/{self.args['general']['data_path']}"

        positions = np.load(f"{data_path}_positions.npy")
        self.projections = np.load(f"{data_path}_projections.npy")

        if self.args["training"]["noise_level"] != None:
            self.projections += (
                np.random.normal(
                    loc=0, scale=self.projections.mean(), size=self.projections.shape
                )
                * self.args["training"]["noise_level"]
            )
            self.projections[self.projections < 0] = 0

        if "filaments_volumes" in data_path:
            with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][0,:,:,:]).permute(2,1,0)
        else:
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol.permute(2,1,0)

        self.detector_size = self.projections[0, :, :].shape
        detector_pos = torch.tensor(positions[:, 3:6])
        detector_pixel_size = torch.tensor(positions[:, 6:])

        source_pos = torch.tensor(positions[:, :3])
        object_shape = self.vol.shape

        self.end_points = [None] * positions.shape[0]
        self.start_points = [None] * positions.shape[0]
        self.valid_rays = [None] * positions.shape[0]

        for i in tqdm(range(positions.shape[0]), desc="Generating points from rays"):
            geometry = Geometry(
                source_pos[i],
                detector_pos[i],
                self.detector_size,
                detector_pixel_size[i],
                object_shape,
            )
            self.end_points[i] = geometry.end_points
            self.start_points[i] = geometry.start_points
            self.valid_rays[i] = geometry.valid_rays

        self.end_points = torch.cat(self.end_points).view(-1, 3)
        self.start_points = torch.cat(self.start_points).view(-1, 3)
        self.valid_rays = torch.cat(self.valid_rays)

        self.noisy = noisy_points

    def sample_points(self, start_points, end_points, num_points):
        """
        Parameters
        ----------
        num_points : int
            Number of points sampled per ray
        """
        # Compute the step size for each ray by dividing the total distance by the number of points
        step_size = (end_points - start_points) / (num_points - 1)

        # Create a tensor 'steps' of shape (num_points, 1, 1)
        # This tensor represents the step indices for each point along the ray
        steps = torch.arange(num_points).unsqueeze(-1).unsqueeze(-1)

        # Compute the coordinates of each point along the ray by adding the start point to the product of the step size and the step indices
        # This uses broadcasting to perform the computation for all points and all rays at once
        points = start_points + step_size * steps

        # Permute the dimensions of the points tensor to match the expected output shape
        return points.permute(1, 0, 2), step_size

    def __len__(self):
        return self.start_points.shape[0]

    def __getitems__(self, idx):
        end_points = self.end_points[idx]
        start_points = self.start_points[idx]
        points, step_size = self.sample_points(
            start_points, end_points, self.args["training"]["num_points"]
        )
        targets = torch.tensor(self.projections.flatten()[self.valid_rays][idx])

        if self.noisy:
            noise = (torch.rand(points.shape) - 0.5) * 0.5
            points = points + (noise * step_size[:, None, :])
            points = torch.clamp(points, -1.0, 1.0)

        points = points.contiguous().to(dtype=torch.float)
        targets = targets.contiguous().to(dtype=torch.float)
        return points, targets


class CTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args_dict,
    ):
        """
        Initializes the Data Module.

        Parameters
        ----------
        args_dict : Dict
            Dictionary with arguments
        """
        super().__init__()
        self.args = args_dict
        self.batch_size = self.args["training"]["batch_size"]

    def setup(self, stage=None):
        """
        Setup datasets for different stages (e.g., 'fit', 'test').

        Args:
        - stage (str, optional): The stage for which data needs to be set up. Defaults to None.
        """
        if stage == "fit" or stage is None:
            if self.args["training"]["noisy_points"]:
                self.train_dataset = CTpoints(self.args, noisy_points=True)
                self.validation_dataset = CTpoints(self.args, noisy_points=False)
            else:
                self.train_dataset = CTpoints(self.args, noisy_points=False)
                self.validation_dataset = CTpoints(self.args, noisy_points=False)
        elif stage == "test":
            self.test_dataset = CTpoints(self.args, noisy_points=False)

    def train_dataloader(self, shuffle=True, notebook=False):
        """
        Returns the training data loader.
        """
        if notebook:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                shuffle=shuffle,
                drop_last=True,
                collate_fn=collate_fn,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=5,
            collate_fn=collate_fn,
        )

    def val_dataloader(self, notebook=False):
        """
        Returns the validation data loader.
        """
        if notebook:
            return DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn,
            )
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=5,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )


class CTpointsWithRays(torch.utils.data.Dataset):
    def __init__(self, args_dict, noisy_points=False):

        self.args = args_dict
        data_path = f"{_PATH_DATA}/{self.args['general']['data_path']}"

        positions = np.load(f"{data_path}_positions.npy")
        self.projections = np.load(f"{data_path}_projections.npy")

        if self.args["training"]["noise_level"] != None:
            self.projections += (
                np.random.normal(
                    loc=0, scale=self.projections.mean(), size=self.projections.shape
                )
                * self.args["training"]["noise_level"]
            )
            self.projections[self.projections < 0] = 0

        if "filaments_volumes" in data_path:
            with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        else:
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol.permute(2,1,0)

        self.detector_size = self.projections[0, :, :].shape
        detector_pos = torch.tensor(positions[:, 3:6])
        detector_pixel_size = torch.tensor(positions[:, 6:])

        source_pos = torch.tensor(positions[:, :3])
        object_shape = self.vol.shape

        self.end_points = [None] * positions.shape[0]
        self.start_points = [None] * positions.shape[0]
        self.valid_rays = [None] * positions.shape[0]
        self.positions = [None] * positions.shape[0]

        for i in tqdm(range(positions.shape[0]), desc="Generating points from rays"):
            geometry = Geometry(
                source_pos[i],
                detector_pos[i],
                self.detector_size,
                detector_pixel_size[i],
                object_shape,
            )
            self.end_points[i] = geometry.end_points
            self.start_points[i] = geometry.start_points
            self.valid_rays[i] = geometry.valid_rays
            self.positions[i] = torch.from_numpy(np.array([positions[i]] * geometry.start_points.shape[0]))

        self.end_points = torch.cat(self.end_points).view(-1, 3)
        self.start_points = torch.cat(self.start_points).view(-1, 3)
        self.valid_rays = torch.cat(self.valid_rays)
        self.positions = torch.cat(self.positions)

        if "FiberDataset" in self.args['general']['ray_data_path']:
            self.real_ray_path = f"{_PATH_DATA}/FiberDataset/combined_interpolated_points.hdf5"
        else:
            self.real_ray_path = self.args['general']['ray_data_path']
        self.real_positions = None
        self.real_start_points = None
        self.real_end_points = None
        self.real_rays = None
        

        self.noisy = noisy_points

    def sample_points(self, start_points, end_points, num_points):
        """
        Parameters
        ----------
        num_points : int
            Number of points sampled per ray
        """
        # Compute the step size for each ray by dividing the total distance by the number of points
        step_size = (end_points - start_points) / (num_points - 1)

        # Create a tensor 'steps' of shape (num_points, 1, 1)
        # This tensor represents the step indices for each point along the ray
        steps = torch.arange(num_points).unsqueeze(-1).unsqueeze(-1)

        # Compute the coordinates of each point along the ray by adding the start point to the product of the step size and the step indices
        # This uses broadcasting to perform the computation for all points and all rays at once
        points = start_points + step_size * steps

        # Permute the dimensions of the points tensor to match the expected output shape
        return points.permute(1, 0, 2), step_size

    def __len__(self):
        return self.start_points.shape[0]

    def __getitems__(self, idx):
        end_points = self.end_points[idx]
        start_points = self.start_points[idx]
        position = self.positions[idx]
        points, step_size = self.sample_points(
            start_points, end_points, self.args["training"]["num_points"]
        )
        targets = torch.tensor(self.projections.flatten()[self.valid_rays][idx])

        if self.noisy:
            noise = (torch.rand(points.shape) - 0.5) * 0.5
            points = points + (noise * step_size[:, None, :])
            points = torch.clamp(points, -1.0, 1.0)

        points = points.contiguous().to(dtype=torch.float)
        targets = targets.contiguous().to(dtype=torch.float)
        position = position.contiguous().to(dtype=torch.float)
        start_points = start_points.contiguous().to(dtype=torch.float)
        end_points = end_points.contiguous().to(dtype=torch.float)

        if self.real_rays == None:
            self.real_positions = h5py.File(self.real_ray_path, 'r')["position"]
            self.real_start_points = h5py.File(self.real_ray_path, 'r')["start_point"]
            self.real_end_points = h5py.File(self.real_ray_path, 'r') ["end_point"]
            self.real_rays = h5py.File(self.real_ray_path, 'r')["ray"]

        sample_idx = np.sort(np.random.choice(self.real_rays.shape[0],self.args["training"]["batch_size"],replace=False))
        real_ray = torch.from_numpy(self.real_rays[sample_idx]).contiguous().to(dtype=torch.float)
        real_position = torch.from_numpy(self.real_positions[sample_idx]).contiguous().to(dtype=torch.float)
        real_start_points = torch.from_numpy(self.real_start_points[sample_idx]).contiguous().to(dtype=torch.float)
        real_end_points = torch.from_numpy(self.real_end_points[sample_idx]).contiguous().to(dtype=torch.float)
        
        return points, targets, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points

class CTRayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args_dict,
    ):
        """
        Initializes the Data Module.

        Parameters
        ----------
        args_dict : Dict
            Dictionary with arguments
        """
        super().__init__()
        self.args = args_dict
        self.batch_size = self.args["training"]["batch_size"]

    def setup(self, stage=None):
        """
        Setup datasets for different stages (e.g., 'fit', 'test').

        Args:
        - stage (str, optional): The stage for which data needs to be set up. Defaults to None.
        """
        if stage == "fit" or stage is None:
            if self.args["training"]["noisy_points"]:
                self.train_dataset = CTpointsWithRays(self.args, noisy_points=True)
                self.validation_dataset = CTpointsWithRays(self.args, noisy_points=False)
            else:
                self.train_dataset = CTpointsWithRays(self.args, noisy_points=False)
                self.validation_dataset = CTpointsWithRays(self.args, noisy_points=False)
        elif stage == "test":
            self.test_dataset = CTpointsWithRays(self.args, noisy_points=False)

    def train_dataloader(self, shuffle=True, notebook=False):
        """
        Returns the training data loader.
        """
        if notebook:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                shuffle=shuffle,
                drop_last=True,
                collate_fn=collate_fn_raygan,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            # persistent_workers=True,
            prefetch_factor=5,
            collate_fn=collate_fn_raygan,
        )

    def val_dataloader(self, notebook=False):
        """
        Returns the validation data loader.
        """
        if notebook:
            return DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                collate_fn=collate_fn_raygan,
            )
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=5,
            collate_fn=collate_fn_raygan,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn_raygan,
        )

class Imagefit(torch.utils.data.Dataset):
    def __init__(self, args_dict, split="train"):

        if "/tmp/" in args_dict['general']['data_path']:
            self.file_path = f"{args_dict['general']['data_path']}"
        else:
            self.file_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"

        if ".hdf5" in self.file_path:
            self.number_of_volumes = h5py.File(f"{self.file_path}", "r")["volumes"].shape[0]
        else:
            files = pd.read_csv(f"{self.file_path}/{split}.csv", header=0)
            self.image_paths = files.file_path.to_list()
            self.number_of_volumes = len(self.image_paths)
            
        self.volume_sidelength = args_dict["model"]["volume_sidelength"]

        self.mgrid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, self.volume_sidelength[0]),
                torch.linspace(-1, 1, self.volume_sidelength[1]),
                torch.linspace(-1, 1, self.volume_sidelength[2]),
                indexing="ij",
            ),
            dim=-1,
        )
        # self.mgrid = self.mgrid.view(-1,self.volume_sidelength,3)
        self.dataset = None

    def __len__(self):
        return self.number_of_volumes * self.mgrid.shape[2]

    def __getitem__(self, idx):
        img_idx = idx // self.mgrid.shape[2]
        grid_idx = idx % self.mgrid.shape[2]

        points = self.mgrid[grid_idx]

        if self.dataset is None:
            self.dataset = h5py.File(f"{self.file_path}/train.hdf5", "r")["volumes"]

        targets = torch.tensor(self.dataset[img_idx][grid_idx])

        return points, targets, img_idx

    def __getitems__(self, idx):
        idx = torch.tensor(idx)
        img_idxs = idx // self.mgrid.shape[2]
        grid_idxs = idx % self.mgrid.shape[2]

        points = self.mgrid[:, :, grid_idxs]
        targets = torch.zeros(
            self.volume_sidelength[0],
            self.volume_sidelength[1],
            *idx.shape,
            dtype=torch.float32,
        )
        if self.dataset is None:
            if ".hdf5" in self.file_path:
                self.dataset = h5py.File(f"{self.file_path}", "r")["volumes"]
            else:
                self.dataset = h5py.File(f"{self.file_path}/train_small.hdf5", "r")["volumes"]

        for img_idx in img_idxs.unique():
            temp = torch.tensor(
                self.dataset[img_idx][:, :, grid_idxs[img_idxs == img_idx]]
            )
            if temp.dtype != torch.float32:
                temp = temp.to(dtype=torch.float32)
            if len(temp.shape) < 3:
                temp = temp.unsqueeze(dim=-1)
            targets[:, :, img_idxs == img_idx] = temp

        targets = targets.to(dtype=torch.float).permute(2, 1, 0).contiguous()
        points = points.to(dtype=torch.float).permute(2, 1, 0, 3).contiguous()
        return points, targets, img_idxs


class ImagefitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args_dict,
    ):
        """
        Initializes the BugNISTDataModule.

        Parameters
        ----------
        args_dict : Dict
            Dictionary with arguments
        """
        super().__init__()
        self.args = args_dict
        self.batch_size = self.args["training"]["batch_size"]

    def setup(self, stage=None):
        """
        Setup datasets for different stages (e.g., 'fit', 'test').

        Args:
        - stage (str, optional): The stage for which data needs to be set up. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = Imagefit(self.args, split="train")
            # self.validation_dataset = Imagefit(self.args, split="train")

        elif stage == "test":
            self.test_dataset = Imagefit(self.args, split="test")

    def train_dataloader(self, shuffle=True):
        """
        Returns the training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
            # drop_last=True,
            persistent_workers=True,
            prefetch_factor=10,
            collate_fn=collate_fn_imagefit,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.
        """
        return DataLoader(
            # self.validation_dataset,
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=5,
            collate_fn=collate_fn_imagefit,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn_imagefit,
        )
