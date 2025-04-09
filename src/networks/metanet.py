import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import torchmetrics as tm

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.encoder import get_encoder
from src.networks.discriminator import Discriminator

from src.networks import get_activation_function, Sine, sine_init, first_layer_sine_init, compute_projection_values

class RayGAN(LightningModule):

    def __init__(self, args_dict, projection_shape=(16, 256, 256)):
        super(RayGAN, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.d_lr = args_dict["training"]["discriminator_lr"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.smoothness_loss_weight = args_dict["training"]["smoothness_weight"]
        self.adversarial_loss_weight = args_dict["training"]["adversarial_weight"]
        self.consistency_loss_weight = args_dict["training"]["consistency_weight"]
        self.feature_matching_loss_weight = args_dict["training"]["fml_weight"]
        self.curvature_loss_weight  = args_dict["training"]["curvature_weight"]

        self.num_hidden_layers = args_dict["model"]["num_hidden_layers"]
        self.num_hidden_features = args_dict["model"]["num_hidden_features"]
        self.activation_function = args_dict["model"]["activation_function"]
        self.args = args_dict
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = get_encoder(args_dict['model']['encoder'])
            num_input_features = self.encoder.output_dim
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

        self.layers = nn.ModuleList(
            [nn.Linear(num_input_features, self.num_hidden_features)] + [nn.Linear(self.num_hidden_features, self.num_hidden_features) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

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

        self.Discriminator = Discriminator(args_dict["training"]["num_points"],
                                           use_multiscale=args_dict["model"]["multiscale"],
                                           use_dilated=args_dict["model"]["dilated"],
                                           use_fft_branch=args_dict["model"]["fft_branch"],
                                           use_segment_aggregation=args_dict["model"]["segment_aggregation"],
                                           use_transformer=args_dict["model"]["transformer"],
                                           transformer_nhead=args_dict["model"]["transformer_nhead"],
                                           transformer_num_layers=args_dict["model"]["transformer_num_layers"])


        self.params = torch.nn.ModuleDict(
            {
                "encoder": torch.nn.ModuleList([self.encoder]),
                "model": torch.nn.ModuleList(
                    [self.mlp_first_half, self.mlp_second_half]
                ),
                "discriminator": torch.nn.ModuleList([self.Discriminator]), 
            }
        )

        self.automatic_optimization = False
        
        self.loss_fn = torch.nn.MSELoss()
        
        self.reg_loss = torch.nn.L1Loss()

        self.acc = tm.classification.BinaryAccuracy()

        self.validation_step_outputs = []
        self.validation_step_gt = []

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def curvature_loss(self, rays):
        # rays: (B, L)
        # Compute second differences along dimension 1.
        second_diff = rays[:, :-2] - 2 * rays[:, 1:-1] + rays[:, 2:]
        return second_diff.abs().mean()

    def compute_adversarial_weight(self):
        """
        Compute a smooth transition weight from 0 to 1 over the training epochs.
        - Sigmoid: `sigmoid((epoch - midpoint) / sharpness)`
        """
        total_epochs = self.trainer.max_epochs
        epoch = self.current_epoch
    
        # Sigmoid ramp-up (midpoint at 30% of training, sharpness controls transition speed)
        midpoint = torch.tensor(self.args['training']['midpoint'] * total_epochs)
        sharpness = torch.tensor(self.args['training']['sharpness'])
        lambda_adv = 1 / (1 + torch.exp(-sharpness * (epoch - midpoint)))
    
        return lambda_adv.item()


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

    def outer_loop(self):
        

    def training_step(self, batch, batch_idx):
        # Unpack batch -- note that valid_mask is True for measured rays and False for extra rays.
        points, target, start_points, end_points, real_ray, real_start_points, real_end_points, valid_mask = batch
    
        optimizer_g, optimizer_d = self.optimizers()
        lambda_adv = self.compute_adversarial_weight()
    
        self.toggle_optimizer(optimizer_g)
    
        # ------------------------------
        # 1. Reconstruction Loss (only for measured rays)
        # ------------------------------
        measured_points = points[valid_mask]      # measured rays with detector values
        measured_target = target[valid_mask]
        measured_start  = start_points[valid_mask]
        measured_end    = end_points[valid_mask]
    
        # Compute lengths along each measured ray.
        lengths = torch.linalg.norm(measured_points[:, -1, :] - measured_points[:, 0, :], dim=1)
    
        # Forward pass for measured rays (only once)
        attenuation_measured = self.forward(measured_points).view(measured_points.shape[0], measured_points.shape[1])
        detector_value_hat = compute_projection_values(measured_points.shape[1], attenuation_measured, lengths)
        recon_loss = self.loss_fn(detector_value_hat, measured_target)
    
        # ------------------------------
        # 2. Adversarial Loss and Regularization (for all rays)
        # ------------------------------
        # Process extra rays separately.
        extra_mask = ~valid_mask
        if extra_mask.sum() > 0:
            extra_points = points[extra_mask]
            attenuation_extra = self.forward(extra_points).view(extra_points.shape[0], extra_points.shape[1])
        else:
            attenuation_extra = None
    
        # Combine measured and extra forward outputs into one tensor.
        # (We assume that the ordering in 'points' is preserved in valid_mask.)
        attenuation_all = torch.empty(points.shape[0], points.shape[1], device=points.device, dtype=attenuation_measured.dtype)
        attenuation_all[valid_mask] = attenuation_measured
        if extra_mask.sum() > 0:
            attenuation_all[extra_mask] = attenuation_extra
    
        # Total Variation (Smoothness) Loss
        if self.smoothness_loss_weight != 0:
            smoothness_loss = self.smoothness_loss_weight * self.reg_loss(
                attenuation_all[:, 1:], attenuation_all[:, :-1]
            )
        else:
            smoothness_loss = 0
    
        # For consistency loss, we need the forward pass on perturbed rays.
        if self.consistency_loss_weight != 0:
            perturbed_points = torch.clamp(points + torch.randn_like(points) * 0.001, -1, 1)
            perturbed_attenuation = self.forward(perturbed_points).view(points.shape[0], points.shape[1])
            consistency_loss = self.consistency_loss_weight * F.mse_loss(attenuation_all, perturbed_attenuation)
        else:
            consistency_loss = 0
    
        # Curvature Loss computed
        if self.curvature_loss_weight != 0:
            curv_loss = self.curvature_loss_weight * self.curvature_loss(attenuation_all)
        else:
            curv_loss = 0
            
        # Adversarial Loss (applied on all rays)
        valid_smooth = torch.empty(attenuation_all.size(0), 1).uniform_(0.8, 1.0).type_as(attenuation_all)
        g_loss = self.adversarial_loss_weight * self.adversarial_loss(
            self.Discriminator(attenuation_all, start_points, end_points), valid_smooth
        )
    
        # Feature Matching Loss
        if self.feature_matching_loss_weight != 0:
            real_features = self.Discriminator.extract_base_feature(real_ray)
            fake_features = self.Discriminator.extract_base_feature(attenuation_all)
            feature_matching_loss = self.feature_matching_loss_weight * F.mse_loss(real_features, fake_features)
        else:
            feature_matching_loss = 0
    
        # Total generator loss: reconstruction loss (on measured rays) plus regularizations/adversarial terms.
        total_g_loss = (recon_loss + smoothness_loss + consistency_loss + curv_loss +
                        lambda_adv * (g_loss + feature_matching_loss))
    
        self.manual_backward(total_g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
    
        # ------------------------------
        # 3. Discriminator Training
        # ------------------------------
        if lambda_adv > 0:
            self.toggle_optimizer(optimizer_d)
            valid_smooth_d = torch.empty(real_ray.size(0), 1).uniform_(0.9, 1.0).type_as(real_ray)
            fake_smooth = torch.empty(attenuation_all.size(0), 1).uniform_(0.0, 0.1).type_as(attenuation_all)
    
            pred_real = self.Discriminator(real_ray, real_start_points, real_end_points)
            pred_fake = self.Discriminator(attenuation_all.detach(), start_points, end_points)
    
            real_loss = self.adversarial_loss(pred_real, valid_smooth_d)
            fake_loss = self.adversarial_loss(pred_fake, fake_smooth)
            d_loss = (real_loss + fake_loss) / 2
    
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
    
            pred_real_labels = (torch.sigmoid(pred_real) > 0.5).float()
            pred_fake_labels = (torch.sigmoid(pred_fake) > 0.5).float()
            acc_real = self.acc(pred_real_labels, torch.ones_like(pred_real_labels))
            acc_fake = self.acc(pred_fake_labels, torch.zeros_like(pred_fake_labels))
            acc_total = (acc_real + acc_fake) / 2
        else:
            d_loss = 0
            acc_total = 0
    
        # ------------------------------
        # 4. Logging and LR Scheduling
        # ------------------------------
        self.log_dict({
            "train/reconstruction_loss": recon_loss,
            "train/smoothness_loss": smoothness_loss,
            "train/consistency_loss": consistency_loss,
            "train/curvature_loss": curv_loss,
            "train/generator_loss": g_loss if lambda_adv > 0 else 0,
            "train/feature_matching_loss": feature_matching_loss if lambda_adv > 0 else 0,
            "train/total_g_loss": total_g_loss if lambda_adv > 0 else 0,
            "train/discriminator_loss": d_loss if lambda_adv > 0 else 0,
            "train/adversarial_weight": lambda_adv,
            "train/discriminator_accuracy": acc_total if lambda_adv > 0 else 0,
        }, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
    
        if self.trainer.is_last_batch:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()


    def validation_step(self, batch, batch_idx):
        points, target, start_points, end_points, real_ray, real_start_points, real_end_points, valid_mask = batch
        
        measured_points = points[valid_mask]      # measured rays with detector values
        measured_target = target[valid_mask]
        measured_start  = start_points[valid_mask]
        measured_end    = end_points[valid_mask]
    
        # Compute lengths along each measured ray.
        lengths = torch.linalg.norm(measured_points[:, -1, :] - measured_points[:, 0, :], dim=1)
    
        # Forward pass for measured rays (only once)
        attenuation_measured = self.forward(measured_points).view(measured_points.shape[0], measured_points.shape[1])
        detector_value_hat = compute_projection_values(measured_points.shape[1], attenuation_measured, lengths)

        loss = self.loss_fn(detector_value_hat, measured_target)

        self.validation_step_outputs.append(detector_value_hat.detach().cpu())
        self.validation_step_gt.append(measured_target.detach().cpu())

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

        valid_rays = self.trainer.val_dataloaders.dataset.measured_valid_mask.view(
            self.projection_shape
        )
        preds = torch.zeros(self.projection_shape, dtype=all_preds.dtype)
        preds[valid_rays] = all_preds
        gt = torch.zeros(self.projection_shape, dtype=all_gt.dtype)
        gt[valid_rays] = all_gt
        
        self.logger.log_image(
                key="val/projection",
                images=[preds[:,:,preds.shape[2]//2], gt[:,:,gt.shape[2]//2], (gt[:,:,gt.shape[2]//2] - preds[:,:,preds.shape[2]//2])],
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
        optimizer_g = torch.optim.AdamW(
            [
                {"params": self.params.encoder.parameters(), "lr": 1e-2, "eps": 1e-15, "weight_decay": 1e-6, "betas": (0.9, 0.99)},
                {"params": self.params.model.parameters(), "lr": self.model_lr},
            ],
            amsgrad=True,
        )
    
        optimizer_d = torch.optim.AdamW(self.Discriminator.parameters(), lr=self.d_lr, amsgrad=True)
    
        # Learning rate schedules
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_g, lambda epoch: 0.995 ** max(0, (epoch - 5000))),
            'interval': 'epoch',
            'frequency': 1
        }
    
        scheduler_d = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_d, lambda epoch: 0.99 ** max(0, (epoch - 2500))),
            'interval': 'epoch',
            'frequency': 1
        }
    
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
