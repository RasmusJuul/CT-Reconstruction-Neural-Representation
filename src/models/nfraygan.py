import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision
import math
import numpy as np
from pytorch_lightning import LightningModule
import torchmetrics as tm
import tifffile
from tqdm import tqdm

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.encoder import get_encoder

from src.models import get_activation_function, Sine, sine_init, first_layer_sine_init, compute_projection_values

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

        self.Discriminator = Discriminator(args_dict["training"]["num_points"])

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

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

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
        
    def training_step(self, batch, batch_idx):
        points, target, start_points, end_points, real_ray, real_start_points, real_end_points = batch
    
        optimizer_g, optimizer_d = self.optimizers()
        
        # Compute the lambda for smooth transition from reconstruction to adversarial loss
        lambda_adv = self.compute_adversarial_weight()

        self.toggle_optimizer(optimizer_g)
        # Compute reconstruction loss
        lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
        attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
        detector_value_hat = compute_projection_values(points.shape[1], attenuation_values, lengths)
        recon_loss = self.loss_fn(detector_value_hat, target)
        
        # Smoothness Loss
        smoothness_loss = self.smoothness_loss_weight * self.reg_loss(
            attenuation_values[:, 1:], attenuation_values[:, :-1]
        )

        # Self-Consistency Loss
        perturbed_points = torch.clamp(points + torch.randn_like(points) * 0.001, -1, 1)
        perturbed_attenuation = self.forward(perturbed_points).view(points.shape[0], points.shape[1])
        consistency_loss = self.consistency_loss_weight * F.mse_loss(attenuation_values, perturbed_attenuation)
        
        # Combined reconstruction loss
        total_recon_loss = recon_loss + smoothness_loss + consistency_loss
        
        # Adversarial loss (gradually introduced)
        valid_smooth = torch.empty(attenuation_values.size(0), 1).uniform_(0.8, 1.0).type_as(attenuation_values)
        g_loss = self.adversarial_loss_weight * self.adversarial_loss(
            self.Discriminator(attenuation_values, start_points, end_points), valid_smooth
        )
        
        # Feature matching loss (optional, to stabilize GAN training)
        real_features = self.Discriminator.extract_features(real_ray)
        fake_features = self.Discriminator.extract_features(attenuation_values)
        feature_matching_loss = self.feature_matching_loss_weight * F.mse_loss(real_features, fake_features)
        
        # **Weighted Generator Loss**
        total_g_loss = total_recon_loss + lambda_adv * (g_loss + feature_matching_loss)

        self.manual_backward(total_g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
    
        # Train the discriminator (only if lambda_adv > 0 to avoid early training)
        if lambda_adv > 0:
            self.toggle_optimizer(optimizer_d)
            valid_smooth = torch.empty(real_ray.size(0), 1).uniform_(0.9, 1.0).type_as(real_ray)
            fake_smooth = torch.empty(attenuation_values.size(0), 1).uniform_(0.0, 0.1).type_as(attenuation_values)
    
            pred_real = self.Discriminator(real_ray, real_start_points, real_end_points)
            pred_fake = self.Discriminator(attenuation_values.detach(), start_points, end_points)
    
            real_loss = self.adversarial_loss(pred_real, valid_smooth)
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
    
        # **Log values**
        self.log_dict({
            "train/reconstruction_loss": recon_loss,
            "train/smoothness_loss": smoothness_loss,
            "train/consistency_loss": consistency_loss,
            "train/total_recon_loss": total_recon_loss,
            "train/generator_loss": g_loss if lambda_adv > 0 else 0,
            "train/feature_matching_loss": feature_matching_loss if lambda_adv > 0 else 0,
            "train/total_g_loss": total_g_loss if lambda_adv > 0 else 0,
            "train/discriminator_loss": d_loss if lambda_adv > 0 else 0,
            "train/adversarial_weight": lambda_adv,
            "train/discriminator_accuracy": acc_total if lambda_adv > 0 else 0,
        }, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # Learning Rate Scheduler Step
        if self.trainer.is_last_batch:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()

    def validation_step(self, batch, batch_idx):
        # points, target, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points = batch
        points, target, start_points, end_points, real_ray, real_start_points, real_end_points = batch
        
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



# New in attempt to get it to work on pancreas
class SelfAttention1d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        self.query_conv = spectral_norm(nn.Conv1d(in_dim, in_dim // 8, kernel_size=1))
        self.key_conv   = spectral_norm(nn.Conv1d(in_dim, in_dim // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv1d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, L = x.size()
        proj_query = self.query_conv(x).view(B, -1, L)  
        proj_key   = self.key_conv(x).view(B, -1, L)    
        proj_value = self.value_conv(x).view(B, C, L)    
        attention = torch.bmm(proj_query.transpose(1, 2), proj_key)  
        attention = F.softmax(attention, dim=-1)                     
        out = torch.bmm(proj_value, attention.transpose(1, 2))     
        out = self.gamma * out + x
        return out

class Discriminator(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        # First convolution block with spectral normalization and dropout
        self.conv1 = spectral_norm(nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3))
        self.conv2 = spectral_norm(nn.Conv1d(16, 32, kernel_size=3, padding=1))
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.skip1 = spectral_norm(nn.Conv1d(16, 32, kernel_size=1))
        
        # Second convolution block
        self.conv3 = spectral_norm(nn.Conv1d(32, 64, kernel_size=3, padding=1, dilation=2))
        self.conv4 = spectral_norm(nn.Conv1d(64, 32, kernel_size=3, padding=1))
        self.skip2 = spectral_norm(nn.Conv1d(64, 32, kernel_size=1))
        
        # Further convolutions
        self.conv5 = spectral_norm(nn.Conv1d(32, 16, kernel_size=3, padding=1))
        self.conv6 = spectral_norm(nn.Conv1d(16, 8, kernel_size=3, padding=1))
        
        # Self-Attention block
        self.attention = SelfAttention1d(8)
        
        # Final convolution to produce 1 channel output
        self.conv7 = spectral_norm(nn.Conv1d(8, 1, kernel_size=3, padding=1))
        
        # MLP for final classification (input: global pooled feature + start and end points)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 3 + 3, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )
        
    def forward(self, ray, start_point, end_point):
        features = self.extract_features(ray)
        mlp_input = torch.cat([features, start_point, end_point], dim=1)
        validity = self.mlp(mlp_input)
        return validity
        
    def extract_features(self, ray):
        """
        Process the input ray through the conv and attention layers to obtain a feature representation.
        Returns a tensor of shape (B, 1) after global average pooling.
        """
        x = ray.unsqueeze(1)  # shape: (B, 1, L)
        x1 = self.leaky_relu(self.conv1(x))
        x1 = self.dropout(x1)
        x2 = self.leaky_relu(self.conv2(x1))
        x2 = self.dropout(x2)
        x2 = x2 + self.skip1(x1)
        x3 = self.leaky_relu(self.conv3(x2))
        x3 = self.dropout(x3)
        x4 = self.leaky_relu(self.conv4(x3))
        x4 = self.dropout(x4)
        x4 = x4 + self.skip2(x3)
        x5 = self.leaky_relu(self.conv5(x4))
        x5 = self.dropout(x5)
        x6 = self.leaky_relu(self.conv6(x5))
        x6 = self.dropout(x6)
        x_attended = self.attention(x6)
        x_final = self.conv7(x_attended)  # shape: (B, 1, L')
        features = x_final.squeeze(1).mean(dim=1, keepdim=True)  # shape: (B, 1)
        return features




# Original simple discriminator used for most early test
# class Discriminator(torch.nn.Module):
#     def __init__(self, num_points):
#         super().__init__()
        
#         self.conv_ray = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool1d(2,2),
#             nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=3),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool1d(2,2),
#             nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear((num_points//4)+3+3, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )

#     def forward(self, ray, start_point,end_point):
#         conv_out = self.conv_ray(ray.unsqueeze(dim=1))
#         mlp_input = torch.cat([conv_out.squeeze(),start_point,end_point],dim=1)
#         validity = self.mlp(mlp_input)

#         return validity
