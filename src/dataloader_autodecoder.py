# -*- coding: utf-8 -*-
import os
import math
import torch
import h5py
import psutil
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from tifffile import tifffile
from torch.utils.data import DataLoader

from src import _PATH_DATA

def collate_fn_imagefit(batch):
    points = batch[0]
    targets = batch[1]
    img_idxs = batch[2]
    return points, targets, img_idxs

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