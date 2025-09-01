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
    ray_dir = batch[2]
    return points, targets, ray_dir


def collate_fn_imagefit(batch):
    points = batch[0]
    targets = batch[1]
    return points, targets

def collate_fn_raygan(batch):
    points = batch[0]
    targets = batch[1]
    start = batch[2]
    end = batch[3]
    real_ray = batch[4]
    real_start = batch[5]
    real_end = batch[6]
    valid = batch[7]
    return points, targets, start, end, real_ray, real_start, real_end, valid


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
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,125:126]).permute(1,2,0)
        elif "bugnist_256" in data_path:
            with h5py.File(f"{_PATH_DATA}/bugnist_256/SL_cubed.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
        elif "Task07_Pancreas" in data_path:
            with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
                self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,49:50,:]
                # self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,:,:]
        elif "plenoptic" in data_path:
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol
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
        
        # === Auto-compute max_steps ===
        dataset_size = len(self)  # total rays
        batch_size = self.args["training"]["batch_size"]
        num_epochs = self.args["training"]["num_epochs"]

        self.max_steps = num_epochs * math.ceil(dataset_size / batch_size)

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
    
        # Ray direction (normalize to unit length)
        ray_dirs = end_points - start_points
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True).clamp_min(1e-8)
    
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
        ray_dirs = ray_dirs.contiguous().to(dtype=torch.float)

        return points, targets, ray_dirs



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


# class CTpointsWithRays(torch.utils.data.Dataset):
#     def __init__(self, args_dict, noisy_points=False):

#         self.args = args_dict
#         data_path = f"{_PATH_DATA}/{self.args['general']['data_path']}"
#         self.file_path = data_path
#         self.noisy = noisy_points

#         positions = np.load(f"{data_path}_positions.npy")
#         self.projections = np.load(f"{data_path}_projections.npy")

#         if self.args["training"]["noise_level"] != None:
#             self.projections += (
#                 np.random.normal(
#                     loc=0, scale=self.projections.mean(), size=self.projections.shape
#                 )
#                 * self.args["training"]["noise_level"]
#             )
#             self.projections[self.projections < 0] = 0

#         if "filaments_volumes" in data_path:
#             with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
#                 self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
#         elif "synthetic_fibers" in data_path:
#             with h5py.File(f"{_PATH_DATA}/synthetic_fibers/test.hdf5", 'r') as f:
#                 self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
#         elif "bugnist_256" in data_path:
#             with h5py.File(f"{_PATH_DATA}/bugnist_256/SL_cubed.hdf5", 'r') as f:
#                 self.vol = torch.from_numpy(f["volumes"][int(data_path.split("_")[-1]),:,:,:]).permute(2,1,0)
#         elif "Task07_Pancreas" in data_path:
#             with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
#                 self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,49:50,:]
#         else:
#             vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
#             vol -= vol.min()
#             vol = vol / vol.max()
#             self.vol = vol.permute(2,1,0)

#         self.detector_size = self.projections[0, :, :].shape
#         detector_pos = torch.tensor(positions[:, 3:6])
#         detector_pixel_size = torch.tensor(positions[:, 6:])

#         source_pos = torch.tensor(positions[:, :3])
#         object_shape = torch.tensor(self.vol.shape)

#         self.end_points = [None] * positions.shape[0]
#         self.start_points = [None] * positions.shape[0]
#         self.valid_rays = [None] * positions.shape[0]

#         for i in tqdm(range(positions.shape[0]), desc="Generating points from rays"):
#             geometry = Geometry(
#                 source_pos[i],
#                 detector_pos[i],
#                 self.detector_size,
#                 detector_pixel_size[i],
#                 object_shape,
#                 beam_type=self.args['general']['beam_type'],
#             )
#             self.end_points[i] = geometry.end_points
#             self.start_points[i] = geometry.start_points
#             self.valid_rays[i] = geometry.valid_rays

#         self.end_points = torch.cat(self.end_points).view(-1, 3)
#         self.start_points = torch.cat(self.start_points).view(-1, 3)
#         self.valid_rays = torch.cat(self.valid_rays)
        
#         self.real_ray_path = self.args['general']['ray_data_path']
        
#         # Get the total system memory
#         total_memory = psutil.virtual_memory().total / (1024 ** 3)
#         if total_memory > 140:
#             print("loading ray data into ram")
#             self.real_start_points = h5py.File(self.real_ray_path, 'r')["start_point"][:]
#             self.real_end_points = h5py.File(self.real_ray_path, 'r')["end_point"][:]
#             self.real_rays = h5py.File(self.real_ray_path, 'r')["ray"][:]
#             self.loaded_rays_into_ram = True
#         else:
#             # self.real_positions = None
#             self.real_start_points = None
#             self.real_end_points = None
#             self.real_rays = None
#             self.loaded_rays_into_ram = False
        

#     def sample_points(self, start_points, end_points, num_points):
#         """
#         Parameters
#         ----------
#         num_points : int
#             Number of points sampled per ray
#         """
#         # Compute the step size for each ray by dividing the total distance by the number of points
#         step_size = (end_points - start_points) / (num_points - 1)

#         # Create a tensor 'steps' of shape (num_points, 1, 1)
#         # This tensor represents the step indices for each point along the ray
#         steps = torch.arange(num_points).unsqueeze(-1).unsqueeze(-1)

#         # Compute the coordinates of each point along the ray by adding the start point to the product of the step size and the step indices
#         # This uses broadcasting to perform the computation for all points and all rays at once
#         points = start_points + step_size * steps

#         # Permute the dimensions of the points tensor to match the expected output shape
#         return points.permute(1, 0, 2), step_size

#     def __len__(self):
#         return self.start_points.shape[0]

#     def __getitems__(self, idx):
#         end_points = self.end_points[idx]
#         start_points = self.start_points[idx]
#         points, step_size = self.sample_points(
#             start_points, end_points, self.args["training"]["num_points"]
#         )
#         targets = torch.tensor(self.projections.flatten()[self.valid_rays][idx])

#         if self.noisy:
#             noise = (torch.rand(points.shape) - 0.5) * 0.5
#             points = points + (noise * step_size[:, None, :])
#             points = torch.clamp(points, -1.0, 1.0)

#         points = points.contiguous().to(dtype=torch.float)
#         targets = targets.contiguous().to(dtype=torch.float)
#         start_points = start_points.contiguous().to(dtype=torch.float)
#         end_points = end_points.contiguous().to(dtype=torch.float)

#         if not self.loaded_rays_into_ram:
#             self.real_start_points = h5py.File(self.real_ray_path, 'r')["start_point"]
#             self.real_end_points = h5py.File(self.real_ray_path, 'r')["end_point"]
#             self.real_rays = h5py.File(self.real_ray_path, 'r')["ray"]

#         sample_idx = np.sort(np.random.choice(self.real_rays.shape[0],self.args["training"]["batch_size"],replace=False))
#         real_ray = torch.from_numpy(self.real_rays[sample_idx]).contiguous().to(dtype=torch.float)
#         real_start_points = torch.from_numpy(self.real_start_points[sample_idx]).contiguous().to(dtype=torch.float)
#         real_end_points = torch.from_numpy(self.real_end_points[sample_idx]).contiguous().to(dtype=torch.float)
        
#         return points, targets, start_points, end_points, real_ray, real_start_points, real_end_points


class CTpointsWithRays(torch.utils.data.Dataset):
    def __init__(self, args_dict, noisy_points=False):
        self.args = args_dict
        data_path = f"{_PATH_DATA}/{self.args['general']['data_path']}"
        self.file_path = data_path
        self.noisy = noisy_points

        ###############################
        # 1. Load measured data and generate measured rays
        ###############################
        # Load positions and projections for the measured rays
        positions = np.load(f"{data_path}_positions.npy")
        self.projections = np.load(f"{data_path}_projections.npy")
        
        if self.args["training"]["noise_level"] is not None:
            self.projections += np.random.normal(
                loc=0, 
                scale=self.projections.mean(), 
                size=self.projections.shape
            ) * self.args["training"]["noise_level"]
            self.projections[self.projections < 0] = 0

        # Load volume
        if "filaments_volumes" in data_path:
            with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
                self.vol = torch.from_numpy(
                    f["volumes"][int(data_path.split("_")[-1]), :, :, :]
                ).permute(2,1,0)
        elif "synthetic_fibers" in data_path:
            with h5py.File(f"{_PATH_DATA}/synthetic_fibers/test.hdf5", 'r') as f:
                self.vol = torch.from_numpy(
                    f["volumes"][int(data_path.split("_")[-1]), :, : ,125:126]
                ).permute(1,2,0)
        elif "bugnist_256" in data_path:
            with h5py.File(f"{_PATH_DATA}/bugnist_256/SL_cubed.hdf5", 'r') as f:
                self.vol = torch.from_numpy(
                    f["volumes"][int(data_path.split("_")[-1]), :, :, :]
                ).permute(2,1,0)
        elif "Task07_Pancreas" in data_path:
            with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
                self.vol = torch.from_numpy(
                    f["volumes"][100+int(data_path.split("_")[-1]), :, :, :]
                ).permute(1,2,0)[:,49:50,:]
                # self.vol = torch.from_numpy(f["volumes"][100+int(data_path.split("_")[-1]),:,:,:]).permute(1,2,0)[:,:,:]
        else:
            import tifffile
            vol = torch.tensor(tifffile.imread(f"{data_path}.tif"))
            vol -= vol.min()
            vol = vol / vol.max()
            self.vol = vol.permute(2,1,0)
        
        # Use the measured projections to define detector size.
        self.detector_size = self.projections[0, :, :].shape

        # Positions file is assumed to contain columns:
        #   [:3]  => source position
        #   [3:6] => detector position
        #   [6:]  => detector pixel size
        measured_source_pos = torch.tensor(positions[:, :3])
        measured_detector_pos = torch.tensor(positions[:, 3:6])
        measured_detector_pixel_size = torch.tensor(positions[:, 6:])
        object_shape = torch.tensor(self.vol.shape)

        measured_start_list = []
        measured_end_list   = []
        measured_valid_list = []  # for debugging or later use if needed
        measured_targets_list = []  # this will store the valid detector values
        
        # Loop over each projection
        for i in tqdm(range(positions.shape[0]), desc="Generating rays from projections"):
            geometry = Geometry(
                measured_source_pos[i],
                measured_detector_pos[i],
                self.detector_size,
                measured_detector_pixel_size[i],
                object_shape,
                beam_type=self.args['general']['beam_type'],
            )
            # geometry.valid_rays is a boolean tensor for the detector grid of projection i
            measured_start_list.append(geometry.start_points)
            measured_end_list.append(geometry.end_points)
            measured_valid_list.append(geometry.valid_rays)
            
            # Get the i-th projection, flatten it, and index it with the valid mask
            proj_i = torch.tensor(self.projections[i]).flatten()  # shape: (detector_size[0]*detector_size[1],)
            valid_mask_i = geometry.valid_rays  # should be a boolean tensor of the same length
            measured_targets_list.append(proj_i[valid_mask_i])
        
        # Now, concatenate the lists.
        self.measured_start_points = torch.cat(measured_start_list).view(-1, 3)
        self.measured_end_points   = torch.cat(measured_end_list).view(-1, 3)
        # Optionally, you can keep a global valid mask if needed:
        self.measured_valid_mask = torch.cat(measured_valid_list)  # shape will match the targets below
        
        # The measured targets are now built from the valid pixels of each projection.
        self.measured_targets = torch.cat(measured_targets_list)

        ###############################
        # 2. Load the measured “real” rays (for adversarial training)
        ###############################
        self.measured_ray_path = self.args['general']['ray_data_path']
        total_memory = psutil.virtual_memory().total / (1024 ** 3)
        if total_memory > 140:
            print("Loading measured ray data into RAM")
            with h5py.File(self.measured_ray_path, 'r') as f:
                self.measured_real_start_points = torch.from_numpy(f["start_point"][:]).float()
                self.measured_real_end_points = torch.from_numpy(f["end_point"][:]).float()
                self.measured_real_rays = torch.from_numpy(f["ray"][:]).float()
            self.measured_loaded_rays_into_ram = True
        else:
            self.measured_loaded_rays_into_ram = False

        ###############################
        # 3. Load extra rays from a second positions file (no detector value)
        ###############################
        # Expecting an extra positions file in args, e.g. args["general"]["extra_positions_path"]
        if self.args['general']['extra_positions_path'] is not None:
            extra_positions = np.load(f"{_PATH_DATA}/{self.args['general']['extra_positions_path']}")
            extra_source_pos = torch.tensor(extra_positions[:, :3])
            extra_detector_pos = torch.tensor(extra_positions[:, 3:6])
            extra_detector_pixel_size = torch.tensor(extra_positions[:, 6:])
            extra_start_list = []
            extra_end_list = []
            extra_valid_list = []  # These extra rays do not have associated detector measurements.
            for i in tqdm(range(extra_positions.shape[0]), desc="Generating extra rays via Geometry"):
                geometry = Geometry(
                    extra_source_pos[i],
                    extra_detector_pos[i],
                    self.detector_size,  # assume same as measured
                    extra_detector_pixel_size[i],
                    object_shape,
                    beam_type=self.args['general']['beam_type'],
                )
                extra_start_list.append(geometry.start_points)
                extra_end_list.append(geometry.end_points)
                extra_valid_list.append(torch.zeros(geometry.start_points.shape[0], dtype=torch.bool))
            self.extra_start_points = torch.cat(extra_start_list).view(-1, 3)
            self.extra_end_points = torch.cat(extra_end_list).view(-1, 3)
            self.extra_valid_mask = torch.cat(extra_valid_list)
            self.num_extra = self.extra_start_points.shape[0]
            # For extra rays (without detector measurements) assign a dummy target value (e.g. –1)
            self.extra_targets = -torch.ones(self.num_extra).float()
            
            ###############################
            # 4. Load the extra “real” rays (paired with the extra rays)
            ###############################
            self.extra_ray_path = self.args['general']['extra_ray_data_path']
            if total_memory > 140:
                print("Loading extra ray data into RAM")
                with h5py.File(self.extra_ray_path, 'r') as f:
                    self.extra_real_start_points = torch.from_numpy(f["start_point"][:]).float()
                    self.extra_real_end_points = torch.from_numpy(f["end_point"][:]).float()
                    self.extra_real_rays = torch.from_numpy(f["ray"][:]).float()
                self.extra_loaded_rays_into_ram = True
            else:
                self.extra_loaded_rays_into_ram = False

        # Total dataset length is measured + extra rays.
        self.total_length = self.measured_start_points.shape[0]

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
        return self.total_length

    def __getitems__(self, idxs):
        """
        Vectorized batch loading for measured rays plus extra rays.
        
        Args:
            idxs (sequence): A sequence (e.g. list or array) of indices (into the measured rays).
            
        Returns:
            points:         Tensor of shape (B_total, num_points, 3) sampled along the rays.
            targets:        Tensor of shape (B_total,) with the detector target (or dummy for extra rays).
            start_points:   Tensor of shape (B_total, 3) ray start coordinates.
            end_points:     Tensor of shape (B_total, 3) ray end coordinates.
            real_rays:      Tensor of shape (B_total, R, L) containing one (or R) corresponding real rays per sample.
            real_starts:    Tensor of shape (B_total, 3) corresponding real-ray start points.
            real_ends:      Tensor of shape (B_total, 3) corresponding real-ray end points.
            valid_mask:     Tensor of shape (B_total,) that is 1.0 for measured rays and 0.0 for extra rays.
            
        Here B_total = (number of measured samples) + (number of extra samples).
        The measured samples are those indicated by idxs, and the extra samples are randomly drawn.
        """
    
        # Parameters.
        num_pts = self.args["training"]["num_points"]
        # Set extra_count from your config
        if self.args['general']['extra_positions_path'] is not None:
            extra_count = self.args["training"]["extra_batch_size"]
        else:
            extra_count = 0
        
        # --- Process measured rays (using the provided indices) ---
        idxs = np.array(idxs)
        measured_count = len(idxs)
        measured_idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        
        # Gather measured ray properties
        measured_start_batch = self.measured_start_points[measured_idxs_tensor]  # shape: (measured_count, 3)
        measured_end_batch   = self.measured_end_points[measured_idxs_tensor]    # shape: (measured_count, 3)
        measured_targets_batch = self.measured_targets[measured_idxs_tensor]     # shape: (measured_count,)
        
        # Sample points along measured rays. 
        # sample_points accepts batched start/end points and returns points of shape (num_pts, N, 3).
        pts_measured, step_size_measured = self.sample_points(measured_start_batch, measured_end_batch, num_pts)
        
        # Optionally add noise.
        if self.noisy:
            noise = (torch.rand(pts_measured.shape) - 0.5) * 0.5
            pts_measured = pts_measured + noise * step_size_measured.unsqueeze(1)
            pts_measured = torch.clamp(pts_measured, -1.0, 1.0)
        
        # For measured rays, valid_mask is 1.
        valid_measured = torch.ones(measured_count, dtype=torch.float)
        
        # For each measured sample, sample one corresponding real ray.
        real_rays_measured = []
        real_starts_measured = []
        real_ends_measured = []
        for _ in range(measured_count):
            if self.measured_loaded_rays_into_ram:
                # Sample one random real ray.
                idx_r = np.random.choice(self.measured_real_rays.shape[0], 1)[0]
                rr = self.measured_real_rays[idx_r]
                rsp = self.measured_real_start_points[idx_r]
                rep = self.measured_real_end_points[idx_r]
            else:
                with h5py.File(self.measured_ray_path, 'r') as f:
                    idx_r = np.random.choice(f["ray"].shape[0], 1)[0]
                    rr = torch.from_numpy(f["ray"][idx_r]).float()
                    rsp = torch.from_numpy(f["start_point"][idx_r]).float()
                    rep = torch.from_numpy(f["end_point"][idx_r]).float()
            real_rays_measured.append(rr)
            real_starts_measured.append(rsp)
            real_ends_measured.append(rep)
        real_rays_measured = torch.stack(real_rays_measured, dim=0)  # shape: (measured_count, L)  [L: ray length]
        real_starts_measured = torch.stack(real_starts_measured, dim=0)  # shape: (measured_count, 3)
        real_ends_measured = torch.stack(real_ends_measured, dim=0)        # shape: (measured_count, 3)

        # --- Process extra rays (sample extra_count rays at random) ---
        if self.args['general']['extra_positions_path'] is not None:
            extra_idxs = np.random.choice(self.num_extra, extra_count, replace=False)
            extra_idxs_tensor = torch.tensor(extra_idxs, dtype=torch.long)
            
            extra_start_batch = self.extra_start_points[extra_idxs_tensor]  # shape: (extra_count, 3)
            extra_end_batch   = self.extra_end_points[extra_idxs_tensor]    # shape: (extra_count, 3)
            extra_targets_batch = self.extra_targets[extra_idxs_tensor]       # shape: (extra_count,)
            
            pts_extra, step_size_extra = self.sample_points(extra_start_batch, extra_end_batch, num_pts)
            if self.noisy:
                noise = (torch.rand(pts_extra.shape) - 0.5) * 0.5
                pts_extra = pts_extra + noise * step_size_extra.unsqueeze(1)
                pts_extra = torch.clamp(pts_extra, -1.0, 1.0)
            
            valid_extra = torch.zeros(extra_count, dtype=torch.float)
            
            real_rays_extra = []
            real_starts_extra = []
            real_ends_extra = []
            for _ in range(extra_count):
                if self.extra_loaded_rays_into_ram:
                    idx_r = np.random.choice(self.extra_real_rays.shape[0], 1)[0]
                    rr = self.extra_real_rays[idx_r]
                    rsp = self.extra_real_start_points[idx_r]
                    rep = self.extra_real_end_points[idx_r]
                else:
                    with h5py.File(self.extra_ray_path, 'r') as f:
                        idx_r = np.random.choice(f["ray"].shape[0], 1)[0]
                        rr = torch.from_numpy(f["ray"][idx_r]).float()
                        rsp = torch.from_numpy(f["start_point"][idx_r]).float()
                        rep = torch.from_numpy(f["end_point"][idx_r]).float()
                real_rays_extra.append(rr)
                real_starts_extra.append(rsp)
                real_ends_extra.append(rep)
            real_rays_extra = torch.stack(real_rays_extra, dim=0)
            real_starts_extra = torch.stack(real_starts_extra, dim=0)
            real_ends_extra = torch.stack(real_ends_extra, dim=0)
        
        # --- Combine measured and extra samples ---
        if self.args['general']['extra_positions_path'] is not None:
            # Concatenate along the batch dimension.
            points_all = torch.cat([pts_measured, pts_extra], dim=0)           # shape: (measured_count + extra_count, num_pts, 3)
            targets_all = torch.cat([measured_targets_batch, extra_targets_batch], dim=0)  # shape: (B_total,)
            start_all = torch.cat([measured_start_batch, extra_start_batch], dim=0)          # shape: (B_total, 3)
            end_all   = torch.cat([measured_end_batch, extra_end_batch], dim=0)              # shape: (B_total, 3)
            valid_all = torch.cat([valid_measured, valid_extra], dim=0)                      # shape: (B_total,)
            
            real_rays_all   = torch.cat([real_rays_measured, real_rays_extra], dim=0)        # shape: (B_total, L)
            real_starts_all = torch.cat([real_starts_measured, real_starts_extra], dim=0)    # shape: (B_total, 3)
            real_ends_all   = torch.cat([real_ends_measured, real_ends_extra], dim=0)        # shape: (B_total, 3)
        else:
            points_all = pts_measured
            targets_all = measured_targets_batch
            start_all = measured_start_batch
            end_all   = measured_end_batch
            valid_all = valid_measured
            
            real_rays_all   = real_rays_measured
            real_starts_all = real_starts_measured
            real_ends_all   = real_ends_measured

        points_all = points_all.contiguous().to(dtype=torch.float)
        targets_all = targets_all.contiguous().to(dtype=torch.float)
        start_all = start_all.contiguous().to(dtype=torch.float)
        end_all   = end_all.contiguous().to(dtype=torch.float)
        valid_all = valid_all.to(dtype=torch.bool)
        real_rays_all = real_rays_all.contiguous().to(dtype=torch.float)
        real_starts_all = real_starts_all.contiguous().to(dtype=torch.float)
        real_ends_all = real_ends_all.contiguous().to(dtype=torch.float)
        
        
        return points_all, targets_all, start_all, end_all, real_rays_all, real_starts_all, real_ends_all, valid_all


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
            persistent_workers=True,
            prefetch_factor=20,
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
