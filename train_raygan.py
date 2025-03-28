import os
import math
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import h5py

import torch
import torch._dynamo
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler
from pytorch_lightning.loggers import WandbLogger
import wandb

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.dataloaders import CTRayDataModule
from src.networks.nfraygan import RayGAN
from src import get_device

torch._dynamo.config.suppress_errors = True


def main(args_dict):
    # if torch.cuda.get_device_properties(0).total_memory / 2**30 > 60:
    #     args_dict["training"]["batch_size"] *= 2
        
    seed_everything(args_dict["general"]["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")
    
    datamodule = CTRayDataModule(args_dict)
    
    projection_shape = np.load(
                f"{_PATH_DATA}/{args_dict['general']['data_path']}_projections.npy"
            ).shape


    model = RayGAN(
        args_dict,
        projection_shape=projection_shape,
    )

    if (
        args_dict["general"]["weights_only"]
        and args_dict["general"]["checkpoint_path"] != None
    ):
        model.load_state_dict(
            torch.load(
                f"{_PATH_MODELS}/{args_dict['general']['checkpoint_path']}",
                map_location=None,
            )["state_dict"],
            strict=False,
        )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    
    wandb_logger = WandbLogger(
        project="Renner",
        name=f"{args_dict['general']['experiment_name']}_projections_{projection_shape[0]}",
    )
    wandb_logger.watch(model, log="all", log_graph=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{_PATH_MODELS}/{args_dict['general']['experiment_name']}_projections_{projection_shape[0]}-{time}",
        filename="{epoch}",
        monitor="val/loss_reconstruction",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
    )

    trainer = Trainer(
        max_epochs=args_dict["training"]["num_epochs"],
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision="16-mixed", #32-true",
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        logger=wandb_logger,
        #strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )

    if (
        not args_dict["general"]["weights_only"]
        and args_dict["general"]["checkpoint_path"] != None
    ):
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=f"{_PATH_MODELS}/{args_dict['general']['checkpoint_path']}",
        )
    else:
        trainer.fit(
            model,
            datamodule=datamodule,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT reconstruction")

    parser_general = parser.add_argument_group("General")
    parser_general.add_argument(
        "--experiment-name", type=str, default="test", help="Name of the experiment"
    )
    parser_general.add_argument(
        "--data-path",
        type=str,
        default="walnut_small_angle/walnut_small",
        help="Path to data",
    )
    parser_general.add_argument(
        "--ray-data-path",
        type=str,
        default="FiberDataset/combined_interpolated_points.hdf5",
        help="Path to ray data",
    )
    parser_general.add_argument(
        "--extra-positions-path",
        type=str,
        default=None,
        help="Path to extra positions for generating rays with no corresponding detector values",
    )
    parser_general.add_argument(
        "--extra-ray-data-path",
        type=str,
        default=None,
        help="Path to extra ray data",
    )
    
    parser_general.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint to continue training from",
    )
    parser_general.add_argument("--seed", type=int, default=42, help="set seed")
    parser_general.add_argument(
        "--weights-only", action="store_true", help="only loads weights from checkpoint"
    )
    parser_general.add_argument(
        "--beam-type",
        type=str,
        default="cone",
        choices=["cone","parallel"],
        help="Path to ray data",
    )

    parser_training = parser.add_argument_group("Training")
    parser_training.add_argument(
        "--num-epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser_training.add_argument(
        "--batch-size", type=int, default=1000, help="Number of rays in each mini-batch"
    )
    parser_training.add_argument(
        "--extra-batch-size", type=int, default=1000, help="Number of extra rays with no detector values in each mini-batch"
    )
    parser_training.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers used in the dataloader",
    )
    parser_training.add_argument(
        "--model-lr",
        type=float,
        default=1e-5,
        help="Learning rate of model parameters for the optimizer",
    )
    parser_training.add_argument(
        "--d-lr",
        type=float,
        default=1e-4,
        help="Learning rate of the discriminator model parameters for the optimizer",
    )
    parser_training.add_argument(
        "--num-points", type=int, default=256, help="Number of points sampled per ray"
    )
    parser_training.add_argument(
        "--noisy-points",
        action="store_true",
        help="Whether or not to add noise to the point",
    )
    parser_training.add_argument(
        "--smoothness-weight",
        type=float,
        default=1e-1,
        help="weight used to scale the smoothness loss, the difference between adjacent points on ray",
    )
    parser_training.add_argument(
        "--consistency-weight",
        type=float,
        default=5e-2,
        help="weight used to scale the consistency loss, the difference between outputs of the network, with very slighty changed points",
    )
    
    parser_training.add_argument(
        "--fml-weight",
        type=float,
        default=1e-3,
        help="weight for feature matching loss, a loss that takes the features from the discriminator and compares features from the real samples and generated samples",
    )
    parser_training.add_argument(
        "--adversarial-weight",
        type=float,
        default=1e-1,
        help="weight used to scale the adversarial loss",
    )
    parser_training.add_argument(
        "--curvature-weight",
        type=float,
        default=1e-5,
        help="weight used to scale the curvature loss (second derivative)",
    )
    
    parser_training.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="constant which will be multiplied by gaussian noise with 0 mean and std of the mean value of the projections",
    )
    parser_training.add_argument(
        "--midpoint",
        type=float,
        default=0.3,
        help="Midpoint (sigmoid) for adversarial transistion weight",
    )
    parser_training.add_argument(
        "--sharpness",
        type=float,
        default=0.1,
        help="Sharpness of sigmoid for adversarial transistion weight",
    )
    

    parser_model = parser.add_argument_group("Model")


    # Arguments for MLP model
    parser_model.add_argument(
        "--num-hidden-layers",
        type=int,
        default=4,
        help="Number of layers in the MLP model",
    )
    parser_model.add_argument(
        "--num-hidden-features",
        type=int,
        default=256,
        help="Number of hidden units in the MLP model",
    )
    parser_model.add_argument(
        "--encoder",
        type=str,
        default=None,
        choices=["hashgrid", "frequency", "spherical", "blob"],
        help="Encoder used in the MLP model",
    )
    parser_model.add_argument(
        "--activation-function",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "tanh", "sigmoid", "elu", "none", "sine"],
        help="Activation function in the MLP model",
    )
    parser_model.add_argument(
        "--multiscale",
        action="store_true",
        help="Whether or not to use multiscale in discriminator",
    )
    parser_model.add_argument(
        "--dilated",
        action="store_true",
        help="Whether or not to use dilation in discriminator",
    )
    parser_model.add_argument(
        "--fft-branch",
        action="store_true",
        help="Whether or not to add a fast fourier transform branch to the discriminator",
    )
    parser_model.add_argument(
        "--segment-aggregation",
        action="store_true",
        help="Whether or not to add a segmented ray aggregation branch to the discriminator",
    )
    parser_model.add_argument(
        "--transformer",
        action="store_true",
        help="Whether or not to use a transformer after convolutions in the discriminator",
    )
    parser_model.add_argument(
        "--transformer-nhead",
        type=int,
        default=4,
        help="Number of layers in the MLP model",
    )
    parser_model.add_argument(
        "--transformer-num-layers",
        type=int,
        default=1,
        help="Number of hidden units in the MLP model",
    )
    
    

    args = parser.parse_args()


    # Args dict used to organise the arguments
    args_dict = {
        "general": {
            "experiment_name": args.experiment_name,
            "data_path": args.data_path,
            'extra_positions_path': args.extra_positions_path,
            "ray_data_path": args.ray_data_path,
            'extra_ray_data_path': args.extra_ray_data_path,
            "seed": args.seed,
            "checkpoint_path": args.checkpoint_path,
            "weights_only": args.weights_only,
            "beam_type": args.beam_type,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "model_lr": args.model_lr,
            "discriminator_lr": args.d_lr,
            "device": get_device().type,
            "num_workers": args.num_workers,
            "num_points": args.num_points,
            "noisy_points": args.noisy_points,
            "smoothness_weight": args.smoothness_weight,
            "consistency_weight":args.consistency_weight,
            "adversarial_weight": args.adversarial_weight,
            "curvature_weight": args.curvature_weight,
            "fml_weight": args.fml_weight,
            "midpoint": args.midpoint,
            "sharpness": args.sharpness,
            "noise_level": args.noise_level,
            "extra_batch_size":args.extra_batch_size,

        },
        "model": {
            "num_hidden_layers": args.num_hidden_layers,
            "num_hidden_features": args.num_hidden_features,
            "encoder": args.encoder,
            "activation_function": args.activation_function,
            "multiscale": args.multiscale,
            "dilated": args.dilated,
            "fft_branch": args.fft_branch,
            "segment_aggregation": args.segment_aggregation,
            "transformer": args.transformer,
            "transformer_nhead": args.transformer_nhead,
            "transformer_num_layers": args.transformer_num_layers,
        },
    }
    main(args_dict)
