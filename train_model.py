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
from src.dataloaders import CTDataModule, ImagefitDataModule
from src.networks.mlp import NeuralField
from src.networks.nfraygan import RayGAN
from src.networks.autodecoder import AutoDecoder, AutoDecoder_adversarial
from src import get_device

torch._dynamo.config.suppress_errors = True


def main(args_dict):
    if torch.cuda.get_device_properties(0).total_memory / 2**30 > 60:
        args_dict["training"]["batch_size"] *= 2
        
    seed_everything(args_dict["general"]["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    
    projection_shape = np.load(
        f"{_PATH_DATA}/{args_dict['general']['data_path']}_projections.npy"
    ).shape
    datamodule = CTDataModule(args_dict)

    model = NeuralField(
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
            strict=True,
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
        monitor="train/loss",
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
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        logger=wandb_logger,
        # strategy="ddp",
        num_sanity_val_steps=-1,
        check_val_every_n_epoch=1,
        # profiler=profiler,
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
        "--num-points", type=int, default=256, help="Number of points sampled per ray"
    )
    parser_training.add_argument(
        "--imagefit-mode",
        action="store_true",
        help="Trains imagefit instead of detectorfit",
    )
    parser_training.add_argument(
        "--noisy-points",
        action="store_true",
        help="Whether or not to add noise to the point",
    )
    parser_training.add_argument(
        "--regularization-weight",
        type=float,
        default=1e-1,
        help="weight used to scale the L1 loss of diffenrence between adjacent points on ray",
    )
    parser_training.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="constant which will be multiplied by gaussian noise with 0 mean and std of the mean value of the projections",
    )
    parser_model = parser.add_argument_group("Model")

    # Shared arguments for all models
    parser_model.add_argument(
        "--model-type",
        type=str,
        default="neuralfield",
        choices=["neuralfield", "neuralfieldsingle"],
        help="Type of model to use",
    )
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
        choices=["hashgrid", "frequency", "spherical"],
        help="Encoder used in the MLP model",
    )
    parser_model.add_argument(
        "--num-freq-bands",
        type=int,
        default=6,
        help="Number of frequency bands in the MLP model if frequency encoder is choosen",
    )
    parser_model.add_argument(
        "--activation-function",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "tanh", "sigmoid", "elu", "none", "sine"],
        help="Activation function in the MLP model",
    )

    args = parser.parse_args()

    # Args dict used to organise the arguments
    args_dict = {
        "general": {
            "experiment_name": args.experiment_name,
            "data_path": args.data_path,
            "seed": args.seed,
            "checkpoint_path": args.checkpoint_path,
            "weights_only": args.weights_only,
            "beam_type": args.beam_type,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "model_lr": args.model_lr,
            "device": get_device().type,
            "num_workers": args.num_workers,
            "num_points": args.num_points,
            "imagefit_mode": args.imagefit_mode,
            "noisy_points": args.noisy_points,
            "regularization_weight": args.regularization_weight,
            "noise_level": args.noise_level,
        },
        "model": {
            "model_type": args.model_type,
            "num_hidden_layers": args.num_hidden_layers,
            "num_hidden_features": args.num_hidden_features,
            "encoder": args.encoder,
            "num_freq_bands": args.num_freq_bands,
            "activation_function": args.activation_function,
        },
    }
    main(args_dict)
