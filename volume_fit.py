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
from src import get_device

torch._dynamo.config.suppress_errors = True

def main(args_dict):
    seed_everything(args_dict["general"]["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    datamodule = ImagefitDataModule(args_dict)
    projection_shape = None
    model = NeuralField(
        args_dict,
        projection_shape=projection_shape,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
        project="Renner",
        name=f"volume_fit_{args_dict['general']['experiment_name']}_{args_dict['model']['encoder']}_{args_dict['model']['activation_function']}",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{_PATH_MODELS}/volume_fit_{args_dict['general']['experiment_name']}_{args_dict['model']['encoder']}_{args_dict['model']['activation_function']}-{time}",
        filename="MLP-{epoch}",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=True,
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
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1,
        # profiler=profiler,
            )
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
        "--num-epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser_training.add_argument(
        "--batch-size", type=int, default=10, help="Number of rays in each mini-batch"
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
        "--imagefit-mode",
        action="store_true",
        help="Trains imagefit instead of detectorfit",
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
    parser_model.add_argument(
        "--latent-size", type=int, default=256, help="Size of the latent vector"
    )
    parser_model.add_argument(
        "--volume-sidelength",
        type=int,
        nargs="+",
        default=(300, 300, 300),
        help="Side lengths of the volume, to be trained on. Most be 3 values seperated by a space e.g. 256 256 256",
    )

    args = parser.parse_args()

    if len(args.volume_sidelength) != 3:
        raise ValueError("volume sidelength must be of length 3")

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
            "imagefit_mode": args.imagefit_mode,
        },
        "model": {
            "model_type": args.model_type,
            "num_hidden_layers": args.num_hidden_layers,
            "num_hidden_features": args.num_hidden_features,
            "encoder": args.encoder,
            "num_freq_bands": args.num_freq_bands,
            "activation_function": args.activation_function,
            "latent_size": args.latent_size,
            "volume_sidelength": tuple(args.volume_sidelength),
        },
    }
    main(args_dict)