# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import pytorch_lightning as pl
import dask.array as da
import zarr

from tqdm import tqdm
from glob import glob
from tifffile import tifffile
from torch.utils.data import DataLoader

from src import _PATH_DATA


def collate_fn(batch):
    points = batch[0]
    targets = batch[1]
    return points,targets

class Geometry(torch.nn.Module):
    def __init__(self,source_pos,detector_pos,detector_size,detector_pixel_size,object_shape,beam_type="cone"):
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
        self.source_pos[0] /= self.object_shape[0]/2
        self.source_pos[1] /= self.object_shape[1]/2
        self.source_pos[2] /= self.object_shape[2]/2
        
        self.detector_pos[0] /= self.object_shape[0]/2
        self.detector_pos[1] /= self.object_shape[1]/2
        self.detector_pos[2] /= self.object_shape[2]/2
        
        self.u_vec[0] /= self.object_shape[0]/2
        self.u_vec[1] /= self.object_shape[1]/2
        self.u_vec[2] /= self.object_shape[2]/2

        self.v_vec[0] /= self.object_shape[0]/2
        self.v_vec[1] /= self.object_shape[1]/2
        self.v_vec[2] /= self.object_shape[2]/2
        

        self.detector_pixel_coordinates = self.create_grid(self.detector_pos, self.u_vec, self.v_vec, self.detector_size[0],self.detector_size[1])
        
        self.ray_directions()

        self.start_points, self.end_points, self.valid_indices = self.ray_sphere_intersections(self.source_pos.repeat(self.detector_size[0]*self.detector_size[1]).view(-1,3),self.rays.view(-1,3),torch.zeros(3),torch.sqrt(torch.tensor(3.)))

        self.start_points = self.start_points[self.valid_indices]
        self.end_points = self.end_points[self.valid_indices]
        # self.start_points[~self.valid_indices] = self.source_pos
        # self.end_points[~self.valid_indices] = self.detector_pixel_coordinates.view(-1,3)[~self.valid_indices]
    
    def ray_directions(self):
        self.rays = (self.detector_pixel_coordinates - self.source_pos)
        lengths = torch.linalg.norm(self.rays,dim=2)
        self.rays[:,:,0] = self.rays[:,:,0] / lengths
        self.rays[:,:,1] = self.rays[:,:,1] / lengths
        self.rays[:,:,2] = self.rays[:,:,2] / lengths

    def ray_sphere_intersections(self,ray_origins, ray_directions, sphere_center, sphere_radius):
        """
        Calculates the intersection points between rays and a sphere using PyTorch tensors.
        
        Args:
            ray_origins (torch.Tensor): Tensor of 3D coordinates of ray origins (shape: (n, 3)).
            ray_directions (torch.Tensor): Tensor of 3D vectors representing ray directions (shape: (n, 3)).
            sphere_center (torch.Tensor): 3D coordinates of the sphere center.
            sphere_radius (float): Radius of the sphere.
        
        Returns:
            torch.Tensor: Intersection points for each ray (shape: (n, 3)).
        """
        # Compute vector from ray origins to sphere center
        oc = ray_origins - sphere_center
        
        # Compute discriminant
        a = torch.sum(ray_directions**2, dim=1)
        b = 2.0 * torch.sum(oc * ray_directions, dim=1)
        c = torch.sum(oc**2, dim=1) - sphere_radius**2
        discriminant = b**2 - 4*a*c
        
        # Initialize intersection points tensor
        start_points = torch.zeros_like(ray_origins)
        end_points = torch.zeros_like(ray_origins)
        
        # Calculate intersection points for rays with non-negative discriminant
        valid_indices = discriminant >= 0
        t1 = (-b[valid_indices] - torch.sqrt(discriminant[valid_indices])) / (2*a[valid_indices])
        t2 = (-b[valid_indices] + torch.sqrt(discriminant[valid_indices])) / (2*a[valid_indices])
        start_points[valid_indices] = ray_origins[valid_indices] + t1[:, None] * ray_directions[valid_indices]
        end_points[valid_indices] = ray_origins[valid_indices] + t2[:, None] * ray_directions[valid_indices]
        return start_points, end_points, valid_indices

    def create_grid(self,detector_pos, u_vec, v_vec, u_size,v_size):
        # Initialize the grid
        detector_pixels = torch.zeros((u_size, v_size, 3), dtype=torch.double)
    
        # Calculate the starting point of the grid
        start_pos = detector_pos - (u_size//2)*u_vec - (v_size//2)*v_vec + u_vec/2 + v_vec/2
    
        # Create ranges for u and v
        u_range = torch.arange(u_size).view(-1, 1, 1).double()
        v_range = torch.arange(v_size).view(1, -1, 1).double()
    
        # Fill the grid using broadcasting and vectorized operations
        detector_pixels = start_pos + u_range * u_vec + v_range * v_vec
    
        return detector_pixels
    
    def sample_points(self, num_points):
        """
        Parameters
        ----------
        num_points : int
            Number of points sampled per ray 
        """
        # Compute the step size for each ray by dividing the total distance by the number of points
        step_size = (self.end_points - self.start_points) / num_points
    
        # Create a tensor 'steps' of shape (num_points+1, 1, 1)
        # This tensor represents the step indices for each point along the ray
        steps = torch.arange(num_points + 1).unsqueeze(-1).unsqueeze(-1)
    
        # Compute the coordinates of each point along the ray by adding the start point to the product of the step size and the step indices
        # This uses broadcasting to perform the computation for all points and all rays at once
        points = self.start_points + step_size * steps
    
        # Permute the dimensions of the points tensor to match the expected output shape
        return points.permute(1,0,2), step_size

        

class CTpoints(torch.utils.data.Dataset):
    def __init__(self, args_dict, noisy_points=False):

        self.args = args_dict
        data_path = self.args['general']['data_path'] 
        
        positions = np.load(f"{data_path}_positions.npy")
        self.projections = np.load(f"{data_path}_projections.npy")
        
        if self.args['training']['noisy_data']:
            self.projections += np.random.normal(loc=0,scale=self.args['training']['noise_std'],size=self.projections.shape)
            self.projections = np.clip(self.projections,0,1)
        
        img = torch.tensor(tifffile.imread(f"{args_dict['general']['data_path']}.tif"))
        img -= img.min()
        img = img/img.max()
        self.img = img.permute(2,1,0)
        if self.args['training']['imagefit_mode']:
            mgrid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img.shape[0]),
                                               torch.linspace(-1, 1, self.img.shape[1]),
                                               torch.linspace(-1, 1, self.img.shape[2]),
                                               indexing='ij'),
                                dim=-1)
            self.mgrid = mgrid.view(-1,self.img.shape[2],3)
        
        
        self.detector_size = self.projections[0,:,:].shape
        detector_pos = torch.tensor(positions[:,3:6])
        detector_pixel_size = torch.tensor(positions[:,6:])
        
        source_pos = torch.tensor(positions[:,:3])
        object_shape = self.img.shape
        
        # self.geometries = [None]*positions.shape[0]

        # self.points = torch.zeros((self.projections.shape[0],self.projections.shape[1]*self.projections.shape[2],self.args['training']['num_points']+1,3))
        # self.step_sizes = torch.zeros((self.projections.shape[0],self.projections.shape[1]*self.projections.shape[2],3))
        self.points = [None]*positions.shape[0]
        self.step_sizes = [None]*positions.shape[0]
        self.valid_indices = [None]*positions.shape[0]
        
        for i in tqdm(range(positions.shape[0]),desc='Generating points from rays'):
            geometry = Geometry(source_pos[i],detector_pos[i],self.detector_size,detector_pixel_size[i],object_shape)
            self.points[i], self.step_sizes[i] = geometry.sample_points(self.args['training']['num_points'])
            self.valid_indices[i] = geometry.valid_indices

        self.points = torch.cat(self.points).view(-1,self.args['training']['num_points']+1,3)
        self.step_sizes = torch.cat(self.step_sizes).view(-1,3)
        self.valid_indices = torch.cat(self.valid_indices)

        self.noisy = noisy_points
    
    def __len__(self):
        if self.args['training']['imagefit_mode']:
            return self.img.shape[2]
        else:
            return self.points.shape[0]


    def __getitem__(self, idx):
        if self.args['training']['imagefit_mode']:
            points = self.mgrid[:,idx,:]
            targets = self.img[:,:,idx]
        else:
            points = self.points[idx].view(-1,3)
            targets = torch.tensor(self.projections.flatten()[self.valid_indices][idx])
            step_size = self.step_sizes[idx]

            if self.noisy:
                noise = torch.zeros_like(step_size)
                noise = (torch.rand(noise.shape)-0.5)*0.5
                points = points+(noise*step_size)[None,:]
                
            points = points.to(dtype=torch.float16)
            targets = targets.to(dtype=torch.float16)
        return points,targets

    def __getitems__(self,idx):
        if self.args['training']['imagefit_mode']:
            points = self.mgrid[:,idx,:]
            targets = self.img[:,:,idx]
        else:
            points = self.points[idx]
            targets = torch.tensor(self.projections.flatten()[self.valid_indices][idx])
            step_size = self.step_sizes[idx]
            
            if self.noisy:
                noise = (torch.rand(points.shape)-0.5)*0.5
                points = points+(noise*step_size[:,None,:])
                
            points = points.to(dtype=torch.float16)
            targets = targets.to(dtype=torch.float16)
        return points,targets


class CTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args_dict,
        num_poses = 16,
    ):
        """
        Initializes the BugNISTDataModule.

        Parameters
        ----------
        args_dict : Dict
            Dictionary with arguments
        num_poses : int
            Number of poses
        """
        super().__init__()
        self.args = args_dict
        self.num_poses = num_poses
        self.batch_size = self.args["training"]["batch_size"]

    def setup(self, stage=None):
        """
        Setup datasets for different stages (e.g., 'fit', 'test').

        Args:
        - stage (str, optional): The stage for which data needs to be set up. Defaults to None.
        """
        if stage == "fit" or stage is None:
            if self.args['training']['noisy_points']:
                self.train_dataset = CTpoints(self.args, noisy_points=True)
                self.validation_dataset = CTpoints(self.args, noisy_points=False)
            else:
                self.train_dataset = CTpoints(self.args, noisy_points=False)
                self.validation_dataset = CTpoints(self.args, noisy_points=False)
        elif stage == "test":
            self.test_dataset = CTpoints(self.args, noisy_points=False)

    def train_dataloader(self,shuffle=True):
        """
        Returns the training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=3,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.
        """
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=3,
            collate_fn=collate_fn,
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
