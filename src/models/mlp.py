import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from pytorch_lightning import LightningModule
import torchmetrics as tm
import tifffile
from tqdm import tqdm

import tinycudann as tcnn

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT


def get_activation_function(activation_function, args_dict, **kwargs):
    if activation_function == "relu":
        return torch.nn.ReLU(**kwargs)
    elif activation_function == "leaky_relu":
        return torch.nn.LeakyReLU(**kwargs)
    elif activation_function == "sigmoid":
        return torch.nn.Sigmoid(**kwargs)
    elif activation_function == "tanh":
        return torch.nn.Tanh(**kwargs)
    elif activation_function == "elu":
        return torch.nn.ELU(**kwargs)
    elif activation_function == "none":
        return torch.nn.Identity(**kwargs)
    elif activation_function == "sine":
        return torch.jit.script(Sine(**kwargs)).to(
            device=args_dict["training"]["device"]
        )
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")


class Sine(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # See Siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # In siren paper see supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


@torch.jit.script
def compute_projection_values(
    num_points: int,
    attenuation_values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    I0 = 1
    # Compute the spacing between ray points
    dx = lengths / (num_points)

    # Compute the sum of mu * dx along each ray
    attenuation_sum = torch.sum(attenuation_values * dx[:, None], dim=1)

    return attenuation_sum


class NeuralField(LightningModule):

    def __init__(self, args_dict, projection_shape=(256, 256)):
        super(NeuralField, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.latent_lr = args_dict["training"]["latent_lr"]
        self.full_mode = args_dict["training"]["full_mode"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.l1_regularization_weight = args_dict["training"]["regularization_weight"]

        self.num_freq_bands = args_dict["model"]["num_freq_bands"]
        self.num_hidden_layers = args_dict["model"]["num_hidden_layers"]
        self.num_hidden_features = args_dict["model"]["num_hidden_features"]
        self.activation_function = args_dict["model"]["activation_function"]

        config = {
            "encoding_hashgrid": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5
            },
            "encoding_spherical": {
                "otype": "SphericalHarmonics",
            	"degree": 4
            },
            "encoding_frequency": {
            	"otype": "Frequency",
            	"n_frequencies": 12              
            },
            "encoding_blob": {
                "otype": "OneBlob", 
                "n_bins": 16,
            },
        }
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=config[f"encoding_{args_dict['model']['encoder']}"])
            num_input_features = self.encoder.n_output_dims
        else:
            self.encoder = None
            num_input_features = 3  # x,y,z coordinate

        layers_first_half = []
        layers_second_half = []
        for i in range(self.num_hidden_layers):
            if i % 2 == 0:
                layers_first_half.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            self.num_hidden_features, self.num_hidden_features
                        ),
                        get_activation_function(self.activation_function, args_dict),
                    )
                )
            else:
                layers_second_half.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            self.num_hidden_features, self.num_hidden_features
                        ),
                        get_activation_function(self.activation_function, args_dict),
                    )
                )

        self.mlp_first_half = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.num_hidden_features),
            get_activation_function(self.activation_function, args_dict),
            *layers_first_half,
        )
        self.mlp_second_half = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_hidden_features + num_input_features, self.num_hidden_features
            ),
            get_activation_function(self.activation_function, args_dict),
            *layers_second_half,
            torch.nn.Linear(self.num_hidden_features, 1),
            torch.nn.Sigmoid(),
        )

        if self.activation_function == "sine":
            self.mlp_first_half.apply(sine_init)
            self.mlp_second_half.apply(sine_init)
            self.mlp_first_half[0].apply(first_layer_sine_init)

        self.params = torch.nn.ModuleDict(
            {
                "encoder":torch.nn.ModuleList([self.encoder]),
                "model": torch.nn.ModuleList(
                    [self.mlp_first_half, self.mlp_second_half]
                ),
            }
        )

        self.loss_fn = torch.nn.MSELoss()
        self.volumefit_loss = torch.nn.L1Loss()
        self.l1_regularization = torch.nn.L1Loss()
        self.validation_step_outputs = []
        self.validation_step_gt = []

        self.smallest_train_loss = torch.inf
        self.train_epoch_loss = 0

    def forward(self, pts):
        pts_shape = pts.shape

        if len(pts.shape) > 2:
            pts = pts.view(-1, 3)

        if self.encoder != None:
            enc = self.encoder(pts).to(dtype=pts.dtype)
        else:
            enc = pts
        
        out = self.mlp_first_half(enc)

        inputs2 = torch.cat([enc, out], dim=1)

        out = self.mlp_second_half(inputs2)

        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1], -1)

        return out


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        points, target, idxs = batch
        lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
        attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
        detector_value_hat = compute_projection_values(
            points.shape[1], attenuation_values, lengths
        )

        loss = self.loss_fn(detector_value_hat, target)

        smoothness_loss = self.l1_regularization(
            attenuation_values[:, 1:], attenuation_values[:, :-1]
        )  # punish model for big changes between adjacent points (to make it smooth)
        loss += self.l1_regularization_weight * smoothness_loss

        self.log_dict(
            {
                "train/loss": loss,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.train_epoch_loss += loss
        return loss

    def validation_step(self, batch, batch_idx):
        points, target, idxs = batch
        
        lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
        attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
        detector_value_hat = compute_projection_values(
            points.shape[1], attenuation_values, lengths
        )

        loss = self.loss_fn(detector_value_hat, target)

        smoothness_loss = self.l1_regularization(
            attenuation_values[:, 1:], attenuation_values[:, :-1]
        )  # punish model for big changes between adjacent points (to make it smooth)
        loss += self.l1_regularization_weight * smoothness_loss

        self.validation_step_outputs.append(detector_value_hat.detach().cpu())
        self.validation_step_gt.append(target.detach().cpu())

        self.log_dict(
            {
                "val/loss": loss,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_gt = torch.cat(self.validation_step_gt)
        vol = self.trainer.val_dataloaders.dataset.vol.to(device=self.device)
        vol_shape = vol.shape

        valid_rays = self.trainer.val_dataloaders.dataset.valid_rays.view(
            self.projection_shape
        )
        preds = torch.zeros(self.projection_shape, dtype=all_preds.dtype)
        preds[valid_rays] = all_preds
        gt = torch.zeros(self.projection_shape, dtype=all_gt.dtype)
        gt[valid_rays] = all_gt

        for i in np.random.randint(0, self.projection_shape[0], 5):
            self.logger.log_image(
                key="val/projection",
                images=[preds[i], gt[i], (gt[i] - preds[i])],
                caption=[f"pred_{i}", f"gt_{i}", f"residual_{i}"],
            )  # log projection images

        mgrid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, vol_shape[0]),
                torch.linspace(-1, 1, vol_shape[1]),
                torch.linspace(-1, 1, vol_shape[1]),
                indexing="ij",
            ),
            dim=-1,
        )
        
        outputs = torch.zeros_like(vol)
        for i in range(mgrid.shape[0]):
            with torch.no_grad():
                outputs[i] = self.forward(mgrid[i].view(-1, 3).to(device=self.device)).view(
                    outputs[i].shape
                )

        self.log(
            "val/loss_reconstruction",
            self.loss_fn(outputs, vol),
            batch_size=self.batch_size,
        )
        self.logger.log_image(
            key="val/reconstruction",
            images=[
                outputs[vol_shape[2] // 2, :, :],
                vol[vol_shape[2] // 2, :, :],
                (vol[vol_shape[2] // 2, :, :] - outputs[vol_shape[2] // 2, :, :]),
                outputs[:, vol_shape[2] // 2, :],
                vol[:, vol_shape[2] // 2, :],
                (vol[:, vol_shape[2] // 2, :] - outputs[:, vol_shape[2] // 2, :]),
                outputs[:, :, vol_shape[2] // 2],
                vol[:, :, vol_shape[2] // 2],
                (vol[:, :, vol_shape[2] // 2] - outputs[:, :, vol_shape[2] // 2]),
            ],
            caption=[
                "pred_xy",
                "gt_xy",
                "residual_xy",
                "pred_yz",
                "gt_yz",
                "residual_yz",
                "pred_xz",
                "gt_xz",
                "residual_xz",
            ],
        )

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_gt.clear()  # free memory

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):

        lr_lambda = lambda epoch: 0.99 ** max(0, (epoch - 100))
        optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.params.encoder.parameters(),
                        "lr": 1e-2,
                        "eps": 1e-15,
                        "weight_decay":1e-6,
                        "betas":(0.9,0.99)
                    },
                    {
                        "params": self.params.model.parameters(),
                        "lr": self.model_lr,
                    },
                ],
                amsgrad=True,
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
