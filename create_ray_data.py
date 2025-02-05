# -*- coding: utf-8 -*-
import os
import torch
import h5py
import scipy
import numpy as np
import pandas as pd
from glob import glob
import qim3d

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

def interpolate_points(i,start_points,end_points,vol,num_points):
    grid = (
    np.linspace(0, vol.shape[0] - 1, vol.shape[0]),
    np.linspace(0, vol.shape[1] - 1, vol.shape[1]),
    np.linspace(0, vol.shape[2] - 1, vol.shape[2]),
    )
    points,_ = sample_points(start_points,
                             end_points,
                             num_points)
    attenuation_values = scipy.interpolate.interpn(grid,
                                                   vol,
                                                   (points * (np.array(vol.shape) / 2) + (np.array(vol.shape) / 2 - 0.5)),
                                                   bounds_error=False,
                                                   fill_value=0)

    
    data = {
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
    
    # angles = np.linspace(0,np.pi/1.5,4) #bugs
    # angles = np.linspace(0,np.pi/2,2) #two angles 0 and 90
    # angles = np.linspace(0,np.pi,5) #five angles 0, 45, 90, 135, 180
    # angles = np.linspace(0,np.pi,16)
    angles = np.linspace(0,np.pi,50)
    
    #here we are assuming the phantom volume is 1 um voxel size.
    # detector_distance = 150e4 #deafult 1, if voxel size = 1um, then 1m = 1e6. synthetic fibers
    # detector_distance = 160e4 #soldier larva
    detector_distance = 150e4 #Pancreas
    sample_source_distance = 3e4 #default 1500, if voxel size = 1um, 3cm = 3e4
    pixel_size_x = 55.0 #default 1, 55um pixel size
    pixel_size_y = 55.0 #default 1, 55um pixel size

    

    # Define volume shape and rotation direction
    # vol_shape_yxz = np.array([256,256,256]) #Synthetic fibers
    # direction = 1
    # num_points = 256
    
    # vol_shape_yxz = np.array([160,160,192]) #SL
    # direction = 3
    # num_points = 256
    
    vol_shape_yxz = np.array([1,512,512]) #Pancreas
    direction = 2
    num_points = 512

    num_volumes = 100
    ray_count = vol_shape_yxz[0]*vol_shape_yxz[1]*len(angles)*num_volumes

    data_upload_count = 0
    
    hdf5_path = f"{_PATH_DATA}/Task07_Pancreas/combined_interpolated_points_50.hdf5"
    # Create HDF5 file
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        # Create a dataset in the file
        dataset_sp = hdf5_file.create_dataset('start_point', (ray_count, 3), dtype='float16')
        dataset_ep = hdf5_file.create_dataset('end_point', (ray_count, 3), dtype='float16')
        dataset_ray = hdf5_file.create_dataset('ray', (ray_count, num_points), dtype='float16')

        # files = pd.read_csv(f"{_PATH_DATA}/bugnist_256/SL_ray_files.csv").file_name.to_list()[:num_volumes]
        # df_list = []
        # for file in tqdm(files,unit="volume", desc="interpolating points"):
        #     vol = qim3d.io.load(file)
        #     vol -= vol.min()
        #     vol = vol/vol.max()
        #     new_vol = np.zeros((192,160,160))
        #     new_vol[:,:,16:144] = vol
        #     vol = new_vol
    
        # for k in tqdm(range(num_volumes), desc="interpolating points"):
        #     # Synthetic fibers
        #     with h5py.File(f"{_PATH_DATA}/FiberDataset/filaments_volumes.hdf5", 'r') as f:
        #         vol = f["volumes"][k,:,:,:].transpose(2,1,0)
        # for k in tqdm(range(num_volumes), desc="interpolating points"):
        #     # Synthetic fibers
        #     with h5py.File(f"{_PATH_DATA}/synthetic_fibers/train.hdf5", 'r') as f:
        #         vol = f["volumes"][k,:,:,:].transpose(2,1,0)
        for k in tqdm(range(num_volumes), desc="interpolating points"):
            with h5py.File(f"{_PATH_DATA}/Task07_Pancreas/train.hdf5", 'r') as f:
                dataset = f["volumes"]
                vol = dataset[k,:,:,:].transpose(1,2,0)[:,49:50,:]
            df_list = []
                
            
            projector_data = dict(src_vu_pix=np.array(src_vu_pix),
                                  vol_z0_pix=sample_source_distance,
                                  vol_shape_yxz=vol_shape_yxz,
                                  angles=angles,
                                  super_sampling=1,
                                  detector_z=detector_distance,
                                  pixel_size_x=pixel_size_x,
                                  pixel_size_y=pixel_size_y,
                                  angle=True,
                                  direction=direction)
            
            positions = define_geometry(**projector_data)
        
            detector_size = vol.shape[:2]
            detector_pos = torch.tensor(positions[:, 3:6])
            detector_pixel_size = torch.tensor(positions[:, 6:])
            
            source_pos = torch.tensor(positions[:, :3])
            object_shape = torch.tensor(vol.shape)
            
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
                    beam_type="cone",
                )
                end_points[i] = geometry.end_points
                start_points[i] = geometry.start_points
                valid_rays[i] = geometry.valid_rays
            
            end_points = torch.cat(end_points).view(-1, 3)
            start_points = torch.cat(start_points).view(-1, 3)
            valid_rays = torch.cat(valid_rays)
        
            projection_shape = (len(angles),*vol_shape_yxz[:2])
            valid_indices = valid_rays.view(projection_shape)
            idxs = [0]
            for i in range(len(angles)):
                if i == len(angles)-1:
                    idxs.append(idxs[i] + torch.sum(valid_indices[i]).item()-1)
                else:
                    idxs.append(idxs[i] + torch.sum(valid_indices[i]).item())
        
            
            
            df_list.append(
                Parallel(n_jobs=len(angles))(
                    delayed(interpolate_points)(i, start_points[idxs[i]:idxs[i+1]], end_points[idxs[i]:idxs[i+1]], vol, num_points)
                    for i in range(len(angles))
                    )
                )

            df_list = [df for list_ in df_list for df in list_]
            df_data = pd.concat(df_list, ignore_index=True)
            dataset_sp[data_upload_count:data_upload_count+len(df_data)] = np.array(list(df_data.start_point))
            dataset_ep[data_upload_count:data_upload_count+len(df_data)] = np.array(list(df_data.end_point))
            dataset_ray[data_upload_count:data_upload_count+len(df_data)] = np.array(list(df_data.ray))
            
            data_upload_count += len(df_data)
            
    # # Concatenate all dataframes
    # df_list = [df for list_ in df_list for df in list_]
    # combined_df = pd.concat(df_list, ignore_index=True)
    # # Save the combined dataframe to a new CSV file
    # start_points = np.array(list(combined_df.start_point))
    # end_points = np.array(list(combined_df.end_point))
    # rays = np.array(list(combined_df.ray))
    
    # # hdf5_path = f"{_PATH_DATA}/FiberDataset/combined_interpolated_points_16.hdf5"
    
    # # hdf5_path = f"{_PATH_DATA}/synthetic_fibers/combined_interpolated_points_16.hdf5"

    # # hdf5_path = f"{_PATH_DATA}/bugnist_256/SL_combined_interpolated_points_4.hdf5"

    # hdf5_path = f"{_PATH_DATA}/Task07_Pancreas/combined_interpolated_points_50.hdf5"
    # # Create HDF5 file
    # with h5py.File(hdf5_path, 'w') as hdf5_file:
    #     # Create a dataset in the file
    #     dataset = hdf5_file.create_dataset('start_point', (start_points.shape[0], 3), dtype='float16')
    #     dataset[:] = start_points
    #     dataset = hdf5_file.create_dataset('end_point', (end_points.shape[0], 3), dtype='float16')
    #     dataset[:] = end_points
    #     dataset = hdf5_file.create_dataset('ray', (rays.shape[0], rays.shape[1]), dtype='float16')
    #     dataset[:] = rays
