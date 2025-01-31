# -*- coding: utf-8 -*-
import os
import math
import torch
import h5py
import psutil
import qim3d
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
    return points, targets


def collate_fn_imagefit(batch):
    points = batch[0]
    targets = batch[1]
    return points, targets

def collate_fn_raygan(batch):
    points = batch[0]
    targets = batch[1]
    start_points = batch[2]
    end_points = batch[3]
    real_ray = batch[4]
    real_start_points = batch[5]
    real_end_points = batch[6]
    return points, targets, start_points, end_points, real_ray, real_start_points, real_end_points

def collate_fn_neuralgan(batch):
    points = batch[0]
    targets = batch[1]
    start_points = batch[2]
    end_points = batch[3]
    real_ray = batch[4]
    real_start_points = batch[5]
    real_end_points = batch[6]
    real_slices = batch[7]
    embedding = batch[8]
    return points, targets, start_points, end_points, real_ray, real_start_points, real_end_points, real_slices, embedding

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
        # Initialization code remains the same
        super(Geometry, self).__init__()
        self.source_pos = source_pos
        self.detector_pos = detector_pos
        self.detector_size = detector_size
        self.detector_pixel_size = detector_pixel_size
        self.u_vec = detector_pixel_size[:3]
        self.v_vec = detector_pixel_size[3:]
        self.beam_type = beam_type
        self.object_shape = torch.tensor(object_shape)  # Convert torch.Size to torch.Tensor
        
        self.detector_pixel_coordinates = self.create_grid(
            self.detector_pos,
            self.u_vec,
            self.v_vec,
            self.detector_size[0],
            self.detector_size[1],
        )

        if self.beam_type == "parallel":
            self.source_pos = self.create_grid(
                self.source_pos,
                self.u_vec,
                self.v_vec,
                self.detector_size[0],
                self.detector_size[1],
            )
            self.rays = self.detector_pixel_coordinates - self.source_pos
            
            self.start_points, self.end_points, self.valid_rays = self.intersect_box(
                self.source_pos.view(-1, 3),
                self.rays.view(-1, 3),
            )
            
        elif self.beam_type == "cone":
            self.rays = self.detector_pixel_coordinates - self.source_pos
            
            self.start_points, self.end_points, self.valid_rays = self.intersect_box(
                self.source_pos.repeat(self.detector_size[0] * self.detector_size[1]).view(
                    -1, 3
                ),
                self.rays.view(-1, 3),
            )

        self.start_points = self.start_points[self.valid_rays]
        self.end_points = self.end_points[self.valid_rays]

    def intersect_box(self, ray_origins, ray_directions):
        """
        Calculate the intersection points of a ray with a box in 3D space,
        and normalize them to the range [-1, 1].
        
        Parameters:
        ray_origins (torch.Tensor): Tensor of shape (N, 3) representing ray origins.
        ray_directions (torch.Tensor): Tensor of shape (N, 3) representing ray directions.
    
        Returns:
        entry_points (torch.Tensor): Normalized entry points of the rays on the box.
        exit_points (torch.Tensor): Normalized exit points of the rays from the box.
        valid_rays (torch.Tensor): Boolean tensor indicating whether the rays intersect the box.
        """
        # Define the box boundaries in physical coordinates
        box_min = -self.object_shape / 2
        box_max = self.object_shape / 2
    
        # Calculate the intersection t values for each axis
        t_min = (box_min - ray_origins) / ray_directions
        t_max = (box_max - ray_origins) / ray_directions
    
        # Reorder t_min and t_max for each axis
        t_min, t_max = torch.where(t_min > t_max, t_max, t_min), torch.where(
            t_min > t_max, t_min, t_max
        )
    
        # Get the maximum of t_min and the minimum of t_max
        t_near = torch.max(t_min, dim=1).values
        t_far = torch.min(t_max, dim=1).values
    
        # Create a tensor to store whether each ray intersects the box
        valid_rays = t_near < t_far
        valid_rays = valid_rays & (t_far >= 0)
    
        # Calculate the intersection points for the valid rays
        entry_points = ray_origins + t_near.unsqueeze(-1) * ray_directions
        exit_points = ray_origins + t_far.unsqueeze(-1) * ray_directions
    
        # Normalize points to the range [-1, 1]
        normalization_factor = self.object_shape / 2
        entry_points_normalized = entry_points / normalization_factor
        exit_points_normalized = exit_points / normalization_factor

        # Set the coordinates of the size 1 axis to 0
        size_1_axis = (self.object_shape == 1).nonzero(as_tuple=True)[0]
        entry_points_normalized[:, size_1_axis] = 0
        exit_points_normalized[:, size_1_axis] = 0
    
        return entry_points_normalized, exit_points_normalized, valid_rays

    def create_grid(self, detector_pos, u_vec, v_vec, u_size, v_size):
        # Initialize the grid
        detector_pixels = torch.zeros((u_size, v_size, 3))
    
        # Calculate the starting point of the grid
        start_pos = (
            detector_pos
            - (u_size // 2) * u_vec
            - (v_size // 2) * v_vec
            + (u_vec / 2 if u_size % 2 == 0 else 0)
            + (v_vec / 2 if v_size % 2 == 0 else 0)
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
        elif "synthetic_fibers" in data_path:
            with h5py.File(f"{_PATH_DATA}/synthetic_fibers/test.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        elif "bugnist_256" in data_path:
            with h5py.File(f"{_PATH_DATA}/bugnist_256/SL_cubed.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        elif "Task07_Pancreas" in data_path:
            with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,49:50,:]
        elif "pasta" in data_path:
            self.vol = torch.zeros((1120,1120,1912))
        else:
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol.permute(2,1,0)

        self.detector_size = self.projections[0, :, :].shape
        detector_pos = torch.tensor(positions[:, 3:6])
        detector_pixel_size = torch.tensor(positions[:, 6:])

        source_pos = torch.tensor(positions[:, :3])
        object_shape = torch.tensor(self.vol.shape)

        self.end_points = [None] * positions.shape[0]
        self.start_points = [None] * positions.shape[0]
        self.valid_rays = [None] * positions.shape[0]
        self.detector_pixel_coordinates = [None] * positions.shape[0]

        for i in tqdm(range(positions.shape[0]), desc="Generating points from rays"):
            geometry = Geometry(
                source_pos[i],
                detector_pos[i],
                self.detector_size,
                detector_pixel_size[i],
                object_shape,
                beam_type=self.args['general']['beam_type'],
            )
            self.end_points[i] = geometry.end_points
            self.start_points[i] = geometry.start_points
            self.valid_rays[i] = geometry.valid_rays
            self.detector_pixel_coordinates[i] = geometry.detector_pixel_coordinates

        self.end_points = torch.cat(self.end_points).view(-1, 3)
        self.start_points = torch.cat(self.start_points).view(-1, 3)
        self.valid_rays = torch.cat(self.valid_rays)
        self.detector_pixel_coordinates = torch.cat(self.detector_pixel_coordinates)

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
        self.slices = self.args['training']['slices']
        data_path = f"{_PATH_DATA}/{self.args['general']['data_path']}"
        self.file_path = data_path
        self.noisy = noisy_points

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
        elif "synthetic_fibers" in data_path:
            with h5py.File(f"{_PATH_DATA}/synthetic_fibers/test.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        elif "bugnist_256" in data_path:
            with h5py.File(f"{_PATH_DATA}/bugnist_256/SL_cubed.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        elif "Task07_Pancreas" in data_path:
            with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,49:50,:]
        else:
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol.permute(2,1,0)

        self.detector_size = self.projections[0, :, :].shape
        detector_pos = torch.tensor(positions[:, 3:6])
        detector_pixel_size = torch.tensor(positions[:, 6:])

        source_pos = torch.tensor(positions[:, :3])
        object_shape = torch.tensor(self.vol.shape)

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
                beam_type=self.args['general']['beam_type'],
            )
            self.end_points[i] = geometry.end_points
            self.start_points[i] = geometry.start_points
            self.valid_rays[i] = geometry.valid_rays

        self.end_points = torch.cat(self.end_points).view(-1, 3)
        self.start_points = torch.cat(self.start_points).view(-1, 3)
        self.valid_rays = torch.cat(self.valid_rays)
        
        self.real_ray_path = self.args['general']['ray_data_path']
        
        # Get the total system memory
        total_memory = psutil.virtual_memory().total / (1024 ** 3)
        if total_memory > 140:
            print("loading ray data into ram")
            self.real_start_points = h5py.File(self.real_ray_path, 'r')["start_point"][:]
            self.real_end_points = h5py.File(self.real_ray_path, 'r')["end_point"][:]
            self.real_rays = h5py.File(self.real_ray_path, 'r')["ray"][:]
            self.loaded_rays_into_ram = True
        else:
            # self.real_positions = None
            self.real_start_points = None
            self.real_end_points = None
            self.real_rays = None
            self.loaded_rays_into_ram = False

        
        
        self.dataset = None
        self.counter_for_slices = 0

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
        start_points = start_points.contiguous().to(dtype=torch.float)
        end_points = end_points.contiguous().to(dtype=torch.float)

        if not self.loaded_rays_into_ram:
            self.real_start_points = h5py.File(self.real_ray_path, 'r')["start_point"]
            self.real_end_points = h5py.File(self.real_ray_path, 'r')["end_point"]
            self.real_rays = h5py.File(self.real_ray_path, 'r')["ray"]

        sample_idx = np.sort(np.random.choice(self.real_rays.shape[0],self.args["training"]["batch_size"],replace=False))
        real_ray = torch.from_numpy(self.real_rays[sample_idx]).contiguous().to(dtype=torch.float)
        real_start_points = torch.from_numpy(self.real_start_points[sample_idx]).contiguous().to(dtype=torch.float)
        real_end_points = torch.from_numpy(self.real_end_points[sample_idx]).contiguous().to(dtype=torch.float)

        if self.slices:
            # Loading of slices
            if self.dataset is None:
                if "bugnist_256" in self.file_path:
                    self.dataset = h5py.File(f"{'/'.join(self.real_ray_path.split('/')[:-1])}/SL_cubed_clean.hdf5", "r")["volumes"]
                    # don't sample the training volume so avoid soldat_16_000 (idx 198), manual for now
                    self.volume_idxs = np.append(np.arange(198),np.arange(self.dataset.shape[0])[199:])
                elif "filaments_volumes" in self.file_path:
                    self.dataset = h5py.File(f"{'/'.join(self.real_ray_path.split('/')[:-1])}/filaments_volumes.hdf5", "r")["volumes"]
                    file_number = int(self.args['general']['data_path'].split("_")[-1])
                    self.volume_idxs = np.append(np.arange(file_number),np.arange(self.dataset.shape[0])[file_number+1:])
                elif "synthetic_fibers" in self.file_path:
                    self.dataset = h5py.File(f"{'/'.join(self.real_ray_path.split('/')[:-1])}/train.hdf5", "r")["volumes"]
                    self.volume_idxs = np.arange(self.dataset.shape[0])

            embedding = torch.tensor([[np.random.rand(1)[0],0,0],
                                      [0,np.random.rand(1)[0],0],
                                      [0,0,np.random.rand(1)[0]]], dtype=torch.float)
            slice_volume = torch.from_numpy(self.dataset[self.volume_idxs[self.counter_for_slices]]).permute(2,1,0)
            slice_volume_shape = slice_volume.shape

            # take the random slice in each direction from a volume
            real_slices = torch.stack((slice_volume[int(slice_volume_shape[0]*embedding[0,0]),:,:],
                                       slice_volume[:,int(slice_volume_shape[1]*embedding[1,1]),:],
                                       slice_volume[:,:,int(slice_volume_shape[2]*embedding[2,2])])
                                     ).unsqueeze(dim=1).contiguous().to(dtype=torch.float)
           
            
            # Increase counter, and reset when counter reaches the number of volumes available
            self.counter_for_slices += 1
            if self.counter_for_slices == self.volume_idxs.shape[0]:
                self.counter_for_slices = 0
            return points, targets, start_points, end_points, real_ray, real_start_points, real_end_points, real_slices, embedding
        
        
        return points, targets, start_points, end_points, real_ray, real_start_points, real_end_points

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
        
        if self.args['training']['slices']:
            if notebook:
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.args["training"]["num_workers"],
                    pin_memory=True,
                    shuffle=shuffle,
                    drop_last=True,
                    collate_fn=collate_fn_neuralgan,
                )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                shuffle=shuffle,
                drop_last=True,
                persistent_workers=True,
                prefetch_factor=20,
                collate_fn=collate_fn_neuralgan,
            )
        else:
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
                persistent_workers=True,
                prefetch_factor=20,
                collate_fn=collate_fn_raygan,
            )
            

    def val_dataloader(self, notebook=False):
        """
        Returns the validation data loader.
        """
        if self.args['training']['slices']:
            if notebook:
                return DataLoader(
                    self.validation_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.args["training"]["num_workers"],
                    pin_memory=True,
                    collate_fn=collate_fn_neuralgan,
                )
            return DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                num_workers=self.args["training"]["num_workers"],
                pin_memory=True,
                prefetch_factor=10,
                collate_fn=collate_fn_neuralgan,
                persistent_workers=True,
            )
        else:
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
                prefetch_factor=10,
                collate_fn=collate_fn_raygan,
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
            collate_fn=collate_fn_neuralgan,
        )

class Imagefit(torch.utils.data.Dataset):
    def __init__(self, args_dict, split="train"):

        self.file_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.vol = torch.tensor(qim3d.io.load(self.file_path)).permute(2,1,0)
        self.vol -= self.vol.min()
        self.vol = self.vol/self.vol.max()
        self.volume_sidelength = self.vol.shape

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

    def __len__(self):
        return self.mgrid.shape[2]

    def __getitems__(self, idx):
        idx = torch.tensor(idx)
        points = self.mgrid[:, :, idx]
        targets = self.vol[:,:,idx]

        targets = targets.to(dtype=torch.float).contiguous()
        points = points.to(dtype=torch.float).contiguous()
        return points, targets


class ImagefitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args_dict,
    ):
        """
        Initializes the DataModule.

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
