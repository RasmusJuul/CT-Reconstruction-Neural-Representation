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

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.encoder import get_encoder
from src.networks import get_activation_function, Sine, sine_init, first_layer_sine_init, compute_projection_values, SaturatingGaussianMixtureLoss

# class NeuralField(LightningModule):

#     def __init__(self, args_dict, projection_shape=(16, 256, 256)):
#         super(NeuralField, self).__init__()
#         self.save_hyperparameters()

#         self.projection_shape = projection_shape
#         self.volume_init = args_dict['training']['imagefit_mode']
#         if not self.volume_init:
#             self.l1_regularization_weight = args_dict["training"]["regularization_weight"]
            
#         self.model_lr = args_dict["training"]["model_lr"]
#         self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
#         self.batch_size = args_dict["training"]["batch_size"]

#         self.num_freq_bands = args_dict["model"]["num_freq_bands"]
#         self.num_hidden_layers = args_dict["model"]["num_hidden_layers"]
#         self.num_hidden_features = args_dict["model"]["num_hidden_features"]
#         self.activation_function = args_dict["model"]["activation_function"]
        
#         # Initialising encoder
#         if args_dict['model']['encoder'] != None:
#             self.encoder = get_encoder(args_dict['model']['encoder'])
#             num_input_features = self.encoder.output_dim
#         else:
#             self.encoder = None
#             num_input_features = 3  # x,y,z coordinate

#         layers_first_half = []
#         layers_second_half = []
#         for i in range(self.num_hidden_layers):
#             if i % 2 == 0:
#                 layers_first_half.append(
#                     torch.nn.Sequential(
#                         torch.nn.Linear(
#                             self.num_hidden_features, self.num_hidden_features
#                         ),
#                         get_activation_function(self.activation_function, args_dict),
#                     )
#                 )
#             else:
#                 layers_second_half.append(
#                     torch.nn.Sequential(
#                         torch.nn.Linear(
#                             self.num_hidden_features, self.num_hidden_features
#                         ),
#                         get_activation_function(self.activation_function, args_dict),
#                     )
#                 )

#         self.mlp_first_half = torch.nn.Sequential(
#             torch.nn.Linear(num_input_features, self.num_hidden_features),
#             get_activation_function(self.activation_function, args_dict),
#             *layers_first_half,
#         )
#         self.mlp_second_half = torch.nn.Sequential(
#             torch.nn.Linear(
#                 self.num_hidden_features + num_input_features, self.num_hidden_features
#             ),
#             get_activation_function(self.activation_function, args_dict),
#             *layers_second_half,
#             torch.nn.Linear(self.num_hidden_features, 1),
#             torch.nn.Sigmoid(),
#         )

#         if self.activation_function == "sine":
#             self.mlp_first_half.apply(sine_init)
#             self.mlp_second_half.apply(sine_init)
#             self.mlp_first_half[0].apply(first_layer_sine_init)

#         self.params = torch.nn.ModuleDict(
#             {
#                 "encoder":torch.nn.ModuleList([self.encoder]),
#                 "model": torch.nn.ModuleList(
#                     [self.mlp_first_half, self.mlp_second_half]
#                 ),
#             }
#         )

#         self.loss_fn = torch.nn.MSELoss()
#         self.volumefit_loss = torch.nn.L1Loss()
#         self.reg_loss = torch.nn.L1Loss()
#         self.GaussianLoss = SaturatingGaussianMixtureLoss([[0.0],[1.0]],[0.1,0.1])
#         self.validation_step_outputs = []
#         self.validation_step_gt = []

#         self.smallest_train_loss = torch.inf
#         self.train_epoch_loss = 0

#     def forward(self, pts):
#         pts_shape = pts.shape

#         if len(pts.shape) > 2:
#             pts = pts.view(-1, 3)

#         if self.encoder != None:
#             enc = self.encoder(pts).to(dtype=pts.dtype)
#         else:
#             enc = pts
        
#         out = self.mlp_first_half(enc)

#         inputs2 = torch.cat([enc, out], dim=1)

#         out = self.mlp_second_half(inputs2)

#         if len(pts_shape) > 2:
#             out = out.view(*pts_shape[:-1], -1)

#         return out


#     def training_step(self, batch, batch_idx):
#         if self.volume_init:
#             points, target = batch
#             attenuation_values = self.forward(points).view(target.shape)
#             loss = self.volumefit_loss(attenuation_values, target)

#             self.log_dict(
#             {
#                 "train/loss": loss,
#             },
#             on_step=True,
#             on_epoch=True,
#             sync_dist=True,
#             batch_size=self.batch_size,
#             )

#             return loss
            
            
#         points, target = batch
#         lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
#         attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
#         detector_value_hat = compute_projection_values(
#             points.shape[1], attenuation_values, lengths
#         )
        
#         loss = self.loss_fn(detector_value_hat, target)

#         smoothness_loss = self.l1_regularization_weight *self.reg_loss(
#             attenuation_values[:, 1:], attenuation_values[:, :-1]
#         )  # punish model for big changes between adjacent points (to make it smooth)
        

#         # g_loss = self.GaussianLoss(detector_value_hat)*1e-3

#         loss += smoothness_loss #+ g_loss
        
#         self.log_dict(
#             {
#                 "train/loss": loss,
#                 "train/Smoothness Loss": smoothness_loss,
#                 #"train/Gaussian Loss": g_loss,
#             },
#             on_step=True,
#             on_epoch=True,
#             sync_dist=True,
#             batch_size=self.batch_size,
#         )
#         self.train_epoch_loss += loss
#         return loss

#     def validation_step(self, batch, batch_idx):
#         if not self.volume_init:
#             points, target = batch
            
#             lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
#             attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
#             detector_value_hat = compute_projection_values(
#                 points.shape[1], attenuation_values, lengths
#             )
    
#             loss = self.loss_fn(detector_value_hat, target)
    
#             self.validation_step_outputs.append(detector_value_hat.detach().cpu())
#             self.validation_step_gt.append(target.detach().cpu())
    
#             self.log_dict(
#                 {
#                     "val/loss": loss,
#                 },
#                 on_step=False,
#                 on_epoch=True,
#                 sync_dist=True,
#                 batch_size=self.batch_size,
#             )

#     def on_validation_epoch_end(self):
#         vol = self.trainer.val_dataloaders.dataset.vol.to(device=self.device)
#         vol_shape = vol.shape
#         if not self.volume_init:
#             all_preds = torch.cat(self.validation_step_outputs)
#             all_gt = torch.cat(self.validation_step_gt)
#             vol = self.trainer.val_dataloaders.dataset.vol.to(device=self.device)
#             vol_shape = vol.shape
    
#             valid_rays = self.trainer.val_dataloaders.dataset.valid_rays.view(
#                 self.projection_shape
#             )
#             preds = torch.zeros(self.projection_shape, dtype=all_preds.dtype)
#             preds[valid_rays] = all_preds
#             gt = torch.zeros(self.projection_shape, dtype=all_gt.dtype)
#             gt[valid_rays] = all_gt
    
#             self.logger.log_image(
#                 key="val/projection",
#                 images=[preds[:,:,preds.shape[2]//2], gt[:,:,gt.shape[2]//2], (gt[:,:,gt.shape[2]//2] - preds[:,:,preds.shape[2]//2])],
#                 # images=[preds[preds.shape[0]//2,:,:], gt[gt.shape[0]//2,:,:], (gt[gt.shape[0]//2,:,:] - preds[preds.shape[0]//2,:,:])],
#                 caption=[f"pred", f"gt", f"residual"],
#             )  # log projection images

#         mgrid = torch.stack(
#             torch.meshgrid(
#                 torch.linspace((0 if vol_shape[0] == 1 else -1), (0 if vol_shape[0] == 1 else 1), vol_shape[0]),
#                 torch.linspace((0 if vol_shape[1] == 1 else -1), (0 if vol_shape[1] == 1 else 1), vol_shape[1]),
#                 torch.linspace((0 if vol_shape[2] == 1 else -1), (0 if vol_shape[2] == 1 else 1), vol_shape[2]),
#                 indexing="ij",
#             ),
#             dim=-1,
#         )
        
#         outputs = torch.zeros_like(vol)
#         for i in range(mgrid.shape[0]):
#             with torch.no_grad():
#                 outputs[i] = self.forward(mgrid[i].view(-1, 3).to(device=self.device)).view(
#                     outputs[i].shape
#                 )

#         self.log(
#             "val/loss_reconstruction",
#             self.loss_fn(outputs, vol),
#             batch_size=self.batch_size,
#         )
#         self.logger.log_image(
#             key="val/reconstruction",
#             images=[
#                 outputs[:, vol_shape[1] // 2, :],
#                 vol[:, vol_shape[1] // 2, :],
#                 (vol[:, vol_shape[1] // 2, :] - outputs[:, vol_shape[1] // 2, :]),
                
#                 outputs[vol_shape[0] // 2, :, :],
#                 vol[vol_shape[0] // 2, :, :],
#                 (vol[vol_shape[0] // 2, :, :] - outputs[vol_shape[0] // 2, :, :]),
                
#                 outputs[:, :, vol_shape[2] // 2],
#                 vol[:, :, vol_shape[2] // 2],
#                 (vol[:, :, vol_shape[2] // 2] - outputs[:, :, vol_shape[2] // 2]),
#             ],
#             caption=[
#                 "pred_xz",
#                 "gt_xz",
#                 "residual_xz",
#                 "pred_yz",
#                 "gt_yz",
#                 "residual_yz",
#                 "pred_xy",
#                 "gt_xy",
#                 "residual_xy",
#             ],
#         )

#         self.validation_step_outputs.clear()  # free memory
#         self.validation_step_gt.clear()  # free memory

#     def test_step(self, batch, batch_idx):
#         return None

#     def configure_optimizers(self):

#         # lr_lambda = lambda epoch: 0.99 ** max(0, (epoch - 100))
#         optimizer = torch.optim.AdamW(
#                 [
#                     {
#                         "params": self.params.encoder.parameters(),
#                         "lr": 1e-2,
#                         "eps": 1e-15,
#                         "weight_decay":1e-6,
#                         "betas":(0.9,0.99)
#                     },
#                     {
#                         "params": self.params.model.parameters(),
#                         "lr": self.model_lr,
#                     },
#                 ],
#                 amsgrad=True,
#             )

#         # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#         # lr_scheduler_config = {
#         #     "scheduler": lr_scheduler,
#         #     "interval": "epoch",
#         # }
#         # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
#         return optimizer


class NeuralField(LightningModule):

    def __init__(self, args_dict, projection_shape=(16, 256, 256)):
        super(NeuralField, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.volume_init = args_dict['training']['imagefit_mode']

        self.model_lr = args_dict["training"]["model_lr"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.num_freq_bands = args_dict["model"]["num_freq_bands"]
        self.num_hidden_layers = args_dict["model"]["num_hidden_layers"]
        self.num_hidden_features = args_dict["model"]["num_hidden_features"]
        self.activation_function = args_dict["model"]["activation_function"]

        # Loss config
        self.loss_cfg = args_dict.get("losses", {})

        # Track steps for annealing
        self.total_steps_estimate = None
        self.current_step = 0

        # Encoder
        if args_dict['model']['encoder'] is not None:
            self.encoder = get_encoder(args_dict['model']['encoder'])
            num_input_features = self.encoder.output_dim
        else:
            self.encoder = None
            num_input_features = 3

        layers_first_half = []
        layers_second_half = []
        for i in range(self.num_hidden_layers):
            if i % 2 == 0:
                layers_first_half.append(
                    nn.Sequential(
                        nn.Linear(self.num_hidden_features, self.num_hidden_features),
                        get_activation_function(self.activation_function, args_dict),
                    )
                )
            else:
                layers_second_half.append(
                    nn.Sequential(
                        nn.Linear(self.num_hidden_features, self.num_hidden_features),
                        get_activation_function(self.activation_function, args_dict),
                    )
                )

        self.mlp_first_half = nn.Sequential(
            nn.Linear(num_input_features, self.num_hidden_features),
            get_activation_function(self.activation_function, args_dict),
            *layers_first_half,
        )
        self.mlp_second_half = nn.Sequential(
            nn.Linear(self.num_hidden_features + num_input_features, self.num_hidden_features),
            get_activation_function(self.activation_function, args_dict),
            *layers_second_half,
            nn.Linear(self.num_hidden_features, 1),
            nn.Sigmoid(),
        )

        if self.activation_function == "sine":
            self.mlp_first_half.apply(sine_init)
            self.mlp_second_half.apply(sine_init)
            self.mlp_first_half[0].apply(first_layer_sine_init)

        self.params = nn.ModuleDict({
            "encoder": nn.ModuleList([self.encoder]),
            "model": nn.ModuleList([self.mlp_first_half, self.mlp_second_half]),
        })

        self.loss_fn = nn.MSELoss()
        self.volumefit_loss = nn.L1Loss()
        self.GaussianLoss = SaturatingGaussianMixtureLoss([[0.0],[1.0]],[0.1,0.1])
        self.validation_step_outputs = []
        self.validation_step_gt = []

        self.smallest_train_loss = torch.inf
        self.train_epoch_loss = 0

    def forward(self, pts):
        pts_shape = pts.shape
        if len(pts.shape) > 2:
            pts = pts.view(-1, 3)
        if self.encoder is not None:
            enc = self.encoder(pts).to(dtype=pts.dtype)
        else:
            enc = pts
        out = self.mlp_first_half(enc)
        inputs2 = torch.cat([enc, out], dim=1)
        out = self.mlp_second_half(inputs2)
        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1], -1)
        return out

    # === REGULARIZER HELPERS ===
    def loss_tv_ray(self, densities):
        return (densities[:,1:] - densities[:,:-1]).abs().mean()

    def loss_lap_ray(self, densities):
        lap = densities[:,2:] - 2*densities[:,1:-1] + densities[:,:-2]
        return (lap**2).mean()

    def loss_grad_3d(self, pts, num_points):
        idx = torch.randperm(pts.shape[0]*pts.shape[1], device=pts.device)[:num_points]
        pts_sub = pts.view(-1,3)[idx].detach().requires_grad_(True)
        dens_sub = self.forward(pts_sub)
        grad = torch.autograd.grad(dens_sub.sum(), pts_sub, create_graph=True)[0]
        return (grad.pow(2).sum(dim=-1)).mean()

    def loss_grad_dir(self, pts, ray_dirs, num_points):
        idx = torch.randperm(pts.shape[0]*pts.shape[1], device=pts.device)[:num_points]
        pts_sub = pts.view(-1,3)[idx].detach().requires_grad_(True)
        dens_sub = self.forward(pts_sub)
        grad = torch.autograd.grad(dens_sub.sum(), pts_sub, create_graph=True)[0]
        dirs_sub = ray_dirs.view(-1,3)[idx]
        dir_dot = (grad * dirs_sub).sum(dim=-1)
        return (dir_dot**2).mean()

    def loss_spec_ray(self, densities, cutoff):
        spec = torch.fft.rfft(densities, dim=-1)
        freq_bins = spec.shape[-1]
        k_cut = int(cutoff * freq_bins)
        hf_energy = (spec.abs()[:, k_cut:]**2).mean()
        return hf_energy

    def get_loss_weight(self, cfg):
        """Returns possibly annealed weight."""
        w = cfg["weight"]
        if cfg.get("anneal", False):
            frac = cfg.get("anneal_frac", 0.5)
            steps_full = int(frac * self.total_steps_estimate)
            decay_factor = max(0.0, 1.0 - min(self.current_step, steps_full) / steps_full)
            return w * decay_factor
        return w

    # === TRAINING STEP ===
    def training_step(self, batch, batch_idx):
        if self.total_steps_estimate is None:
            self.total_steps_estimate = self.trainer.train_dataloader.dataset.max_steps
        
        self.current_step += 1

        if self.volume_init:
            points, target = batch
            attenuation_values = self.forward(points).view(target.shape)
            loss = self.volumefit_loss(attenuation_values, target)
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
            return loss

        points, target, ray_dirs = batch
        lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
        attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
        detector_value_hat = compute_projection_values(points.shape[1], attenuation_values, lengths)

        loss = self.loss_fn(detector_value_hat, target)

        # Optional regularizers
        if self.loss_cfg.get("tv_ray", {}).get("enabled", False):
            loss += self.get_loss_weight(self.loss_cfg["tv_ray"]) * self.loss_tv_ray(attenuation_values)
        if self.loss_cfg.get("lap_ray", {}).get("enabled", False):
            loss += self.get_loss_weight(self.loss_cfg["lap_ray"]) * self.loss_lap_ray(attenuation_values)
        if self.loss_cfg.get("grad_3d", {}).get("enabled", False):
            loss += self.get_loss_weight(self.loss_cfg["grad_3d"]) * self.loss_grad_3d(points, self.loss_cfg["grad_3d"]["num_points"])
        if self.loss_cfg.get("grad_dir", {}).get("enabled", False):
            loss += self.get_loss_weight(self.loss_cfg["grad_dir"]) * self.loss_grad_dir(points, ray_dirs, self.loss_cfg["grad_dir"]["num_points"])
        if self.loss_cfg.get("spec_ray", {}).get("enabled", False):
            loss += self.get_loss_weight(self.loss_cfg["spec_ray"]) * self.loss_spec_ray(attenuation_values, self.loss_cfg["spec_ray"]["cutoff"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.train_epoch_loss += loss
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.volume_init:
            points, target, _ = batch
            
            lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
            attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
            detector_value_hat = compute_projection_values(
                points.shape[1], attenuation_values, lengths
            )
    
            loss = self.loss_fn(detector_value_hat, target)
    
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
        vol = self.trainer.val_dataloaders.dataset.vol.to(device=self.device)
        vol_shape = vol.shape
        if not self.volume_init:
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
    
            self.logger.log_image(
                key="val/projection",
                images=[preds[:,:,preds.shape[2]//2], gt[:,:,gt.shape[2]//2], (gt[:,:,gt.shape[2]//2] - preds[:,:,preds.shape[2]//2])],
                # images=[preds[preds.shape[0]//2,:,:], gt[gt.shape[0]//2,:,:], (gt[gt.shape[0]//2,:,:] - preds[preds.shape[0]//2,:,:])],
                caption=[f"pred", f"gt", f"residual"],
            )  # log projection images

        mgrid = torch.stack(
            torch.meshgrid(
                torch.linspace((0 if vol_shape[0] == 1 else -1), (0 if vol_shape[0] == 1 else 1), vol_shape[0]),
                torch.linspace((0 if vol_shape[1] == 1 else -1), (0 if vol_shape[1] == 1 else 1), vol_shape[1]),
                torch.linspace((0 if vol_shape[2] == 1 else -1), (0 if vol_shape[2] == 1 else 1), vol_shape[2]),
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
                outputs[:, vol_shape[1] // 2, :],
                vol[:, vol_shape[1] // 2, :],
                (vol[:, vol_shape[1] // 2, :] - outputs[:, vol_shape[1] // 2, :]),
                
                outputs[vol_shape[0] // 2, :, :],
                vol[vol_shape[0] // 2, :, :],
                (vol[vol_shape[0] // 2, :, :] - outputs[vol_shape[0] // 2, :, :]),
                
                outputs[:, :, vol_shape[2] // 2],
                vol[:, :, vol_shape[2] // 2],
                (vol[:, :, vol_shape[2] // 2] - outputs[:, :, vol_shape[2] // 2]),
            ],
            caption=[
                "pred_xz",
                "gt_xz",
                "residual_xz",
                "pred_yz",
                "gt_yz",
                "residual_yz",
                "pred_xy",
                "gt_xy",
                "residual_xy",
            ],
        )

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_gt.clear()  # free memory

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):

        # lr_lambda = lambda epoch: 0.99 ** max(0, (epoch - 100))
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

        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # lr_scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "epoch",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer

