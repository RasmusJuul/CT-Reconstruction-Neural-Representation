# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from tifffile import tifffile
from torch.utils.data import DataLoader

from src import _PATH_DATA


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

        self.detector_coordinates()
        self.ray_directions()
        self.min_depth = -source_pos[2]-object_shape[2]/2
        self.max_depth = self.min_depth + object_shape[2]
        self.start_points = (self.source_pos/(self.object_shape[2]/2) + self.rays * self.min_depth/(self.object_shape[2]/2)) 
        self.end_points =  self.source_pos/(self.object_shape[2]/2) + self.rays * self.max_depth/(self.object_shape[2]/2)

    def detector_coordinates(self):
        x_component = self.u_vec*(self.detector_size[0]/2)
        y_component = self.v_vec*(self.detector_size[1]/2)
        
        bottom_left_corner = self.detector_pos - x_component - y_component
        bottom_left_corner_pixel_pos = bottom_left_corner + self.u_vec/6 + self.v_vec/6 #Something weird here I would expect it to be self.u_vec/2 + self.v_vec/2

        detector_pixel_coordinates = torch.zeros((self.detector_size[0],self.detector_size[1],3))
        if self.detector_size[0] == self.detector_size[1]:
            for i in range(self.detector_size[0]):
                detector_pixel_coordinates[:,i,0] = torch.arange(0,self.detector_size[0])
                detector_pixel_coordinates[i,:,1] = torch.arange(0,self.detector_size[1])
        else:
            for i in range(self.detector_size[1]):
                detector_pixel_coordinates[:,i,0] = torch.arange(0,self.detector_size[0])
            for i in range(self.detector_size[0]):
                detector_pixel_coordinates[i,:,1] = torch.arange(0,self.detector_size[1])
        
        self.detector_pixel_coordinates = (detector_pixel_coordinates * self.u_vec + detector_pixel_coordinates * self.v_vec)+bottom_left_corner_pixel_pos
    
    def ray_directions(self):
        self.rays = (self.detector_pixel_coordinates - self.source_pos)
        lengths = torch.linalg.norm(self.rays,dim=2)
        self.rays[:,:,0] = self.rays[:,:,0] / lengths
        self.rays[:,:,1] = self.rays[:,:,1] / lengths
        self.rays[:,:,2] = self.rays[:,:,2] / lengths
    
    def sample_points(self,num_points,noise=False):
        """
        Parameters
        ----------
        num_points : int
            Number of points sampled per ray 
        """
        self.step_size = (self.end_points-self.start_points)/num_points
        points = torch.zeros(self.detector_size[0],self.detector_size[1],num_points+1,3)
        for i in range(num_points+1):
            points[:,:,i,:] = self.start_points + self.step_size*i


        self.points = points.view(-1,num_points+1,3)
        

class CTpoints(torch.utils.data.Dataset):
    def __init__(self, args_dict):

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

        self.points = torch.zeros((self.projections.shape[0],self.projections.shape[1]*self.projections.shape[2],self.args['training']['num_points']+1,3))
        self.step_sizes = torch.zeros((*self.projections.shape,3))
        
        for i in tqdm(range(positions.shape[0]),desc='Generating points from rays'):
            geometries = Geometry(source_pos[i],detector_pos[i],self.detector_size,detector_pixel_size[i],object_shape)
            geometries.sample_points(self.args['training']['num_points'])
            self.points[i] = geometries.points
            self.step_sizes[i] = geometries.step_size
            
        self.points = self.points.view(-1,*self.points.shape[-2:])
        self.step_sizes = self.step_sizes.view(-1,3)
        
    
    def __len__(self):
        if self.args['training']['imagefit_mode']:
            return self.img.shape[2]
        else:
            return self.detector_size[0]*self.detector_size[1]*self.projections.shape[0]


    def __getitem__(self, idx):
        if self.args['training']['imagefit_mode']:
            x = self.mgrid[:,idx,:]
            y = self.img[:,:,idx]
            step_size = None
        else:
            x = self.points[idx].view(-1,3)
            y = self.projections.flatten()[idx]
            step_size = self.step_sizes[idx]
            
        return x,y,step_size



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
        self.dataset = CTpoints(self.args)
        # if stage == "test" or stage is None:
        # if stage == "fit" or stage is None:
            

    def train_dataloader(self,shuffle=True):
        """
        Returns the training data loader.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=3,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=3,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.args["training"]["num_workers"],
            pin_memory=True,
        )