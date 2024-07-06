import logging
import datetime
import argparse
import numpy as np

import torch
import torch._dynamo
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler
# from lightning.pytorch.tuner import Tuner
from pytorch_lightning.loggers import WandbLogger
import wandb

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.dataloaders import CTDataModule
from src.models.mlp import MLP
from src import get_device

torch._dynamo.config.suppress_errors = True
    
def main(args_dict):
    seed_everything(args_dict['general']['seed'], workers=True)

    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    torch.set_float32_matmul_precision("medium")

    projection_shape = np.load(f"{args_dict['general']['data_path']}_projections.npy").shape
    
    datamodule = CTDataModule(args_dict,num_poses=projection_shape[0])
    
    
    model = MLP(args_dict, 
                projection_shape=projection_shape,
               )

    if args_dict['general']['checkpoint_path'] != None:
        model.load_state_dict(torch.load(f"{_PATH_MODELS}/{args_dict['general']['checkpoint_path']}", map_location=None)['state_dict'], strict=True)
    
    # if args_dict['training']['compiled']:
    #     model = torch.compile(model,dynamic=True)
            

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{_PATH_MODELS}/{args_dict['general']['experiment_name']}-{time}",
        filename="MLP-{epoch}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=True,
    )

    wandb_logger = WandbLogger(project="Renner", name=args_dict['general']['experiment_name'], log_model="all")
    wandb_logger.watch(model, log="all")
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stopping_callback = EarlyStopping(
        monitor="val/loss_total",
        patience=30,
        verbose=True,
        mode="min",
        strict=False,
        check_on_train_epoch_end=False,
        check_finite = True,
    )

    profiler = PyTorchProfiler(dirpath=".",filename="profile",sort_by_key="cpu_time_total")
    
    trainer = Trainer(
        max_epochs=args_dict['training']['num_epochs'],
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback,lr_monitor],
        log_every_n_steps=25,
        logger=wandb_logger,
        strategy='ddp',
        num_sanity_val_steps=-1,
        # gradient_clip_val=0.5,
        check_val_every_n_epoch=5,
        # profiler=profiler,
        
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='CT reconstruction')

    parser_general = parser.add_argument_group('General')
    parser_general.add_argument('--experiment-name', type=str, default='test', help='Name of the experiment')
    parser_general.add_argument('--data-path', type=str, default='data/synthetic_fibers/train/000/fiber_00000', help='Path to data')
    parser_general.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint to continue training from')
    parser_general.add_argument('--seed', type=int, default=42, help='set seed')
    
    parser_training = parser.add_argument_group('Training')
    parser_training.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train')
    parser_training.add_argument('--batch-size', type=int, default=1000, help='Number of rays in each mini-batch')
    parser_training.add_argument('--num-workers', type=int, default=0, help='Number of workers used in the dataloader')
    parser_training.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser_training.add_argument('--num-points', type=int, default=256, help='Number of points sampled per ray')
    parser_training.add_argument('--imagefit-mode', action='store_true', help='Enable training of imagefit in addition to detector fitting')
    parser_training.add_argument('--compiled', action='store_true', help='Whether or not to torch compile the model')
    parser_training.add_argument('--noisy-points', action='store_true', help='Whether or not to add noise to the point')
    parser_training.add_argument('--regularization-weight', type=float, default=1e-1, help='weight used to scale the L1 loss of diffenrence between adjacent points on ray')
    parser_training.add_argument('--noisy-data', action='store_true', help='Whether or not to add noise to the projections')
    parser_training.add_argument('--noise-std', type=float, default=1e-2, help='standard deviation of the gaussian noise added to the projections if noisy-data is on')
    
    
    parser_model = parser.add_argument_group('Model')
    
    # Shared arguments for all models
    parser_model.add_argument('--model-type', type=str, default='mlp', choices=['mlp']
                                    ,help='Type of model to use')
    # Arguments for MLP model
    parser_model.add_argument('--num-hidden-layers', type=int, default=4, help='Number of layers in the MLP model')
    parser_model.add_argument('--num-hidden-features', type=int, default=256, help='Number of hidden units in the MLP model')
    parser_model.add_argument('--num-freq-bands', type=int, default=6, help='Number of frequency bands in the MLP model')
    parser_model.add_argument('--activation-function', type=str, default='relu', choices=['relu', 'leaky_relu','tanh', 'sigmoid', 'elu','none','sine'], help='Activation function in the MLP model')
    
    args = parser.parse_args()
    
    # Args dict used to organise the arguments
    args_dict = {
        "general": {
            "experiment_name": args.experiment_name,
            "data_path":args.data_path,
            "seed":args.seed,
            "checkpoint_path":args.checkpoint_path,
            
        },
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate":args.learning_rate,
            "device":get_device().type,
            "num_workers":args.num_workers,
            "num_points":args.num_points,
            "imagefit_mode":args.imagefit_mode,
            "compiled":args.compiled,
            "noisy_points":args.noisy_points,
            "regularization_weight":args.regularization_weight,
            "noisy_data":args.noisy_data,
            "noise_std":args.noise_std,
        },
        "model": { 
            "model_type": args.model_type,
            "num_hidden_layers": args.num_hidden_layers,
            "num_hidden_features": args.num_hidden_features,
            "num_freq_bands": args.num_freq_bands,
            "activation_function": args.activation_function,
        },
    }
    main(args_dict)