# -*- coding: utf-8 -*-
import os
import torch
import h5py
import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.typing import NDArray
from collections.abc import Sequence

from src import _PATH_DATA
from src.dataloaders import Geometry

def sample_points(start_points, end_points, num_points):
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

def interpolate_points(i,position,start_points,end_points,vol):
    grid = (np.linspace(0, vol.shape[0]-1, vol.shape[0]),np.linspace(0, vol.shape[1]-1, vol.shape[1]),np.linspace(0, vol.shape[2]-1, vol.shape[2]))
    
    points,_ = sample_points(start_points,
                             end_points,
                             256)
    attenuation_values = scipy.interpolate.interpn(grid,vol,(points*(vol.shape[2]//2)+(vol.shape[2]//2-0.5)),bounds_error=False,fill_value=0)

    
    data = {
        "position": [position] * attenuation_values.shape[0],
        "start_point": [sp.numpy() for sp in start_points],
        "end_point": [ep.numpy() for ep in end_points],
        "ray": [ray for ray in attenuation_values]
    }
    
    df = pd.DataFrame(data)
    return df
    

def define_geometry(src_vu_pix: NDArray, vol_z0_pix: float, vol_shape_yxz: Sequence[int] | NDArray, angles: NDArray = np.array([0]), super_sampling: int = 2, detector_z: float = 0.0, pixel_size_x: float = 1.0,pixel_size_y: float = 1.0, angle: bool = False, direction=1,
    ):
    if angle:
        num_imgs = angles.shape[-1]
        if direction == 1:
            src_vu2xyz = np.swapaxes(np.stack([np.zeros(num_imgs),-vol_z0_pix*np.sin(angles),-vol_z0_pix*np.cos(angles)]),0,1)
            det_xyz = np.swapaxes(np.stack([np.zeros(num_imgs),detector_z*np.sin(angles),detector_z*np.cos(angles)]),0,1)
    
            dir_y = np.zeros_like(src_vu2xyz)
            dir_y[:, 0] = pixel_size_y
            
            dir_x = np.zeros_like(src_vu2xyz)
            
            # Compute cross product for all images at once
            cp = np.cross(dir_y/np.linalg.norm(dir_y), det_xyz/np.linalg.norm(det_xyz))
            
            # Compute norms and reshape to match the shape of cp
            norms = np.linalg.norm(cp, axis=1).reshape(-1, 1)
            
            # Normalize cp and multiply by pixel_size_x
            dir_x = (cp / norms) * pixel_size_x
            
            
        elif direction == 2:
            src_vu2xyz = np.swapaxes(np.stack([-vol_z0_pix*np.sin(angles),np.zeros(num_imgs),-vol_z0_pix*np.cos(angles)]),0,1)
            det_xyz = np.swapaxes(np.stack([detector_z*np.sin(angles),np.zeros(num_imgs),detector_z*np.cos(angles)]),0,1)
            
            dir_y = np.zeros_like(src_vu2xyz)
            dir_y[:, 1] = pixel_size_y
            
            dir_x = np.zeros_like(src_vu2xyz)
            
            # Compute cross product for all images at once
            cp = np.cross(dir_y/np.linalg.norm(dir_y), det_xyz/np.linalg.norm(det_xyz))
            
            # Compute norms and reshape to match the shape of cp
            norms = np.linalg.norm(cp, axis=1).reshape(-1, 1)
            
            # Normalize cp and multiply by pixel_size_x
            dir_x = (cp / norms) * pixel_size_x
    
            
        elif direction == 3:
            src_vu2xyz = np.swapaxes(np.stack([-vol_z0_pix*np.sin(angles),-vol_z0_pix*np.cos(angles),np.zeros(num_imgs)]),0,1)
            det_xyz = np.swapaxes(np.stack([detector_z*np.sin(angles),detector_z*np.cos(angles),np.zeros(num_imgs)]),0,1)
            
            dir_y = np.zeros_like(src_vu2xyz)
            dir_y[:, 2] = pixel_size_y
            
            dir_x = np.zeros_like(src_vu2xyz)
            
            # Compute cross product for all images at once
            cp = np.cross(dir_y/np.linalg.norm(dir_y), det_xyz/np.linalg.norm(det_xyz))
            
            # Compute norms and reshape to match the shape of cp
            norms = np.linalg.norm(cp, axis=1).reshape(-1, 1)
            
            # Normalize cp and multiply by pixel_size_x
            dir_x = (cp / norms) * pixel_size_x
        
            
    else:
        num_imgs = src_vu_pix.shape[-1]
        src_vu2xyz = np.empty((num_imgs, 3))
        src_vu2xyz[:, 2] = src_vu_pix[0]
        src_vu2xyz[:, 1] = src_vu_pix[1]
        src_vu2xyz[:, 0] = -vol_z0_pix
        
        det_xyz = np.zeros_like(src_vu2xyz)
        det_xyz[:, 2] = -src_vu_pix[0]
        det_xyz[:, 1] = -src_vu_pix[1]
        det_xyz[:, 0] = detector_z
        
        dir_x = np.zeros_like(src_vu2xyz)
        dir_x[:, 2] = pixel_size_x
        dir_y = np.zeros_like(src_vu2xyz)
        dir_y[:, 1] = pixel_size_y 
        
    det_geometry = np.hstack([src_vu2xyz, det_xyz, dir_x, dir_y])
    return det_geometry
    
if __name__=="__main__":
    # sourec points in multiple of pixel unit:
    sources_v = np.linspace(-5, 5, 4) * 2e1
    sources_u = np.linspace(-5, 5, 4) * 2e1
    src_vu_pix = np.meshgrid(sources_v, sources_u, indexing="ij")
    src_vu_pix = np.stack([c.flatten() for c in src_vu_pix], axis=0)
    
    angles = np.linspace(0,np.pi,16)
    
    #here we are assuming the phantom volume is 1 um voxel size.
    detector_distance = 100e4 #deafult 1, if voxel size = 1um, then 1m = 1e6.
    sample_source_distance = 3e4 #default 1500, if voxel size = 1um, 3cm = 3e4
    pixel_size_x = 55.0 #default 1, 55um pixel size
    pixel_size_y = 55.0 #default 1, 55um pixel size
    
    vol_shape_yxz = np.array([256,256,256])
    df_list = []
    for k in tqdm(range(20,40),unit="volume", desc="interpolating points"):
        with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
            vol = f["volumes"][k,:,:,:].transpose(2,1,0)
            
        for option in [True,False]:
            projector_data = dict(src_vu_pix=np.array(src_vu_pix),
                                  vol_z0_pix=sample_source_distance,
                                  vol_shape_yxz=vol_shape_yxz,
                                  angles=angles,
                                  super_sampling=1,
                                  detector_z=detector_distance,
                                  pixel_size_x=pixel_size_x,
                                  pixel_size_y=pixel_size_y,
                                  angle=option,
                                  direction=1)
            
            positions = define_geometry(**projector_data)
        
            detector_size = vol_shape_yxz[:2]
            detector_pos = torch.tensor(positions[:, 3:6])
            detector_pixel_size = torch.tensor(positions[:, 6:])
            
            source_pos = torch.tensor(positions[:, :3])
            object_shape = vol_shape_yxz
            
            end_points = [None] * positions.shape[0]
            start_points = [None] * positions.shape[0]
            valid_rays = [None] * positions.shape[0]
            
            for i in tqdm(range(positions.shape[0]), desc="Generating points from rays"):
                geometry = Geometry(
                    source_pos[i],
                    detector_pos[i],
                    detector_size,
                    detector_pixel_size[i],
                    object_shape,
                )
                end_points[i] = geometry.end_points
                start_points[i] = geometry.start_points
                valid_rays[i] = geometry.valid_rays
            
            end_points = torch.cat(end_points).view(-1, 3)
            start_points = torch.cat(start_points).view(-1, 3)
            valid_rays = torch.cat(valid_rays)
        
            projection_shape = (16,256,256)
            valid_indices = valid_rays.view(projection_shape)
            idxs = [0]
            for i in range(16):
                if i == 15:
                    idxs.append(idxs[i] + torch.sum(valid_indices[i]).item()-1)
                else:
                    idxs.append(idxs[i] + torch.sum(valid_indices[i]).item())
        
            
            
            df_list.append(
                Parallel(n_jobs=16)(
                    delayed(interpolate_points)(i,positions[i], start_points[idxs[i]:idxs[i+1]], end_points[idxs[i]:idxs[i+1]], vol)
                    for i in range(16)
                    )
                )
            
    # Concatenate all dataframes
    df_list = [df for list_ in df_list for df in list_]
    combined_df = pd.concat(df_list, ignore_index=True)
    # Save the combined dataframe to a new CSV file
    start_points = np.array(list(combined_df.start_point))
    end_points = np.array(list(combined_df.end_point))
    positions = np.array(list(combined_df.position))
    rays = np.array(list(combined_df.ray))
    
    hdf5_path = f"{_PATH_DATA}/FiberDataset/combined_interpolated_points.hdf5"
    # Create HDF5 file
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        # Create a dataset in the file
        dataset = hdf5_file.create_dataset('position', (positions.shape[0], 12), dtype='float64')
        dataset[:] = positions
        dataset = hdf5_file.create_dataset('start_point', (start_points.shape[0], 3), dtype='float64')
        dataset[:] = start_points
        dataset = hdf5_file.create_dataset('end_point', (end_points.shape[0], 3), dtype='float64')
        dataset[:] = end_points
        dataset = hdf5_file.create_dataset('ray', (rays.shape[0], rays.shape[1]), dtype='float64')
        dataset[:] = rays