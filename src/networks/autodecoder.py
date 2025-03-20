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
from src.networks import get_activation_function, Sine, sine_init, first_layer_sine_init, compute_projection_values
from src.networks.discriminator import Discriminator

class AutoDecoder(LightningModule):
    def __init__(
        self, args_dict, projection_shape=(300, 300), num_volumes=1000, latent=None
    ):
        super(AutoDecoder, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.latent_lr = args_dict["training"]["latent_lr"]
        self.imagefit_mode = args_dict["training"]["imagefit_mode"]
        self.full_mode = args_dict["training"]["full_mode"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.smoothness_loss_weight = args_dict["training"]["smoothness_weight"]
        self.adversarial_loss_weight = args_dict["training"]["adversarial_weight"]
        self.consistency_loss_weight = args_dict["training"]["consistency_weight"]
        self.feature_matching_loss_weight = args_dict["training"]["fml_weight"]
        self.curvature_loss_weight  = args_dict["training"]["curvature_weight"]

        self.num_freq_bands = args_dict["model"]["num_freq_bands"]
        self.num_hidden_layers = args_dict["model"]["num_hidden_layers"]
        self.num_hidden_features = args_dict["model"]["num_hidden_features"]
        self.activation_function = args_dict["model"]["activation_function"]
        self.latent_size = args_dict["model"]["latent_size"]
        self.num_volumes = num_volumes
        self.latent = latent

        # Initialising latent vectors
        self.lat_vecs = torch.nn.Embedding(num_volumes, self.latent_size)
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            1 / math.sqrt(self.latent_size),
        )
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = get_encoder(args_dict['model']['encoder'])
            num_input_features = self.encoder.output_dim + self.latent_size
        else:
            self.encoder = None
            num_input_features = 3  + self.latent_size # x,y,z coordinate

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
                "latent_vectors": torch.nn.ModuleList([self.lat_vecs]),
                "encoder":torch.nn.ModuleList([self.encoder]),
                "model": torch.nn.ModuleList(
                    [self.mlp_first_half, self.mlp_second_half]
                ),
            }
        )

        self.loss_fn = torch.nn.MSELoss()
        self.volumefit_loss = torch.nn.L1Loss()
        self.reg_loss = torch.nn.L1Loss()
        self.ms_ssim = tm.image.MultiScaleStructuralSimilarityIndexMeasure(
                        data_range=1.0,
                        gaussian_kernel=True,
                        kernel_size=11,
                        sigma=1.5,
                        reduction='elementwise_mean',
                        k1=0.01, 
                        k2=0.03,
                        betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                        normalize='relu')
        self.alpha = 0.2
        self.validation_step_outputs = []
        self.validation_step_gt = []

        self.random_sample_idx = np.random.randint(self.num_volumes)
        self.prediction_img = []
        self.gt_img = []

        self.smallest_train_loss = torch.inf
        self.train_epoch_loss = 0

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
    
        # Sigmoid ramp-up (midpoint controls what % into training adversarial weight should be 0.5, sharpness controls transition speed)
        midpoint = torch.tensor(self.args['training']['midpoint'] * total_epochs)
        sharpness = torch.tensor(self.args['training']['sharpness'])
        lambda_adv = 1 / (1 + torch.exp(-sharpness * (epoch - midpoint)))
    
        return lambda_adv.item()

    def batch_latent_vector(self,points_shape):
        return self.latent.repeat(points_shape[0], points_shape[1], 1, 1).permute(2, 0, 1, 3).contiguous().view(-1, self.latent_size)

    def forward(self, pts, vecs):
        pts_shape = pts.shape

        if len(pts.shape) > 2:
            pts = pts.view(-1, 3)

        if self.encoder != None:
            enc = self.encoder(pts)
        else:
            enc = pts

        enc = torch.cat([vecs, enc], dim=1)

        out = self.mlp_first_half(enc)

        inputs2 = torch.cat([enc, out], dim=1)

        out = self.mlp_second_half(inputs2)

        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1], -1)

        return out
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        if self.imagefit_mode:
            points, target, idxs = batch
            batch_vecs = (
                self.lat_vecs(idxs)
                .repeat(points.shape[1], points.shape[2], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            attenuation_values = self.forward(points, batch_vecs)
            attenuation_values = attenuation_values.view(target.shape)
            
            msssim = (1 - self.ms_ssim(attenuation_values.unsqueeze(dim=1),
                                       target.unsqueeze(dim=1)))
            l1 = self.volumefit_loss(attenuation_values, target)
            loss = self.alpha * msssim + (1 - self.alpha) * l1

            l2_size_loss = torch.sum(torch.norm(self.lat_vecs(idxs), dim=1))
            reg_loss = (1e-4 * min(1, self.current_epoch / 100) * l2_size_loss) / len(
                idxs
            )

            loss += reg_loss.cuda()


            self.log_dict(
                {
                    "train/loss": loss,
                },
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            if len(self.prediction_img) == 0:
                if self.random_sample_idx in idxs:
                    for i in (idxs == self.random_sample_idx).nonzero(as_tuple=True)[0]:
                        self.prediction_img.append(attenuation_values[i])
                        self.gt_img.append(target[i])

            return loss
        else:
            points, target, start_points, end_points, real_ray, real_start_points, real_end_points, valid_mask = batch
            optimizer_g, optimizer_d = self.optimizers()
            lambda_adv = self.compute_adversarial_weight()
            batch_vecs = batch_latent_vector(points.shape)
        
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
            attenuation_measured = self.forward(measured_points, batch_vecs[valid_mask]).view(measured_points.shape[0], measured_points.shape[1])
            detector_value_hat = compute_projection_values(measured_points.shape[1], attenuation_measured, lengths)
            recon_loss = self.loss_fn(detector_value_hat, measured_target)
            
            latent_size_loss = 1e-4 * torch.mean(self.latent.pow(2))
            
            # ------------------------------
            # 2. Adversarial Loss and Regularization (for all rays)
            # ------------------------------
            # Process extra rays separately.
            
            
            extra_mask = ~valid_mask
            if extra_mask.sum() > 0:
                extra_points = points[extra_mask]
                attenuation_extra = self.forward(extra_points,batch_vecs[extra_mask]).view(extra_points.shape[0], extra_points.shape[1])
            else:
                attenuation_extra = None

            
            # Combine measured and extra forward outputs into one tensor.
            # (We assume that the ordering in 'points' is preserved in valid_mask.)
            attenuation_all = torch.empty(points.shape[0], points.shape[1], device=points.device, dtype=attenuation_measured.dtype)
            attenuation_all[valid_mask] = attenuation_measured
            if extra_mask.sum() > 0:
                attenuation_all[extra_mask] = attenuation_extra
        
            # Total Variation (Smoothness) Loss
            smoothness_loss = self.smoothness_loss_weight * self.reg_loss(
                attenuation_all[:, 1:], attenuation_all[:, :-1]
            )
        
            # For consistency loss, we need the forward pass on perturbed rays.
            perturbed_points = torch.clamp(points + torch.randn_like(points) * 0.001, -1, 1)
            perturbed_attenuation = self.forward(perturbed_points, batch_vecs).view(points.shape[0], points.shape[1])
            consistency_loss = self.consistency_loss_weight * F.mse_loss(attenuation_all, perturbed_attenuation)
        
            # Curvature Loss computed
            curv_loss = self.curvature_loss_weight * self.curvature_loss(attenuation_all)
        
            # Adversarial Loss (applied on all rays)
            valid_smooth = torch.empty(attenuation_all.size(0), 1).uniform_(0.8, 1.0).type_as(attenuation_all)
            g_loss = self.adversarial_loss_weight * self.adversarial_loss(
                self.Discriminator(attenuation_all, start_points, end_points), valid_smooth
            )
        
            # Feature Matching Loss
            real_features = self.Discriminator.extract_base_feature(real_ray)
            fake_features = self.Discriminator.extract_base_feature(attenuation_all)
            feature_matching_loss = self.feature_matching_loss_weight * F.mse_loss(real_features, fake_features)
        
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

            self.train_epoch_loss += loss
            
            if self.trainer.is_last_batch:
                sch1, sch2 = self.lr_schedulers()
                sch1.step()
                sch2.step()

    def on_train_epoch_end(self):
        if self.imagefit_mode:
            # log prediction images
            self.logger.log_image(
                key="train/image",
                images=[
                    self.prediction_img[0],
                    self.gt_img[0],
                    (self.gt_img[0] - self.prediction_img[0]),
                ],
                caption=[
                    f"pred_{self.random_sample_idx}",
                    f"gt_{self.random_sample_idx}",
                    f"residual_{self.random_sample_idx}",
                ],
            )
            # free memory
            self.prediction_img.clear()
            self.gt_img.clear()
            # Update index
            self.random_sample_idx = np.random.randint(self.num_volumes)
        else:
            self.train_epoch_loss /= len(self.trainer.train_dataloader)
            if self.train_epoch_loss < self.smallest_train_loss:
                self.smallest_train_loss = self.train_epoch_loss
                torch.save(
                    self.latent, f"{self.data_path}/latent_vector-{self.latent_size}.pt"
                )
            self.train_epoch_loss = 0

    def validation_step(self, batch, batch_idx):
        if self.imagefit_mode:
            return None
        else:
            points, target, start_points, end_points, real_ray, real_start_points, real_end_points, valid_mask = batch
            
        
            measured_points = points[valid_mask]      # measured rays with detector values
            measured_target = target[valid_mask]
            measured_start  = start_points[valid_mask]
            measured_end    = end_points[valid_mask]

            batch_vecs = batch_latent_vector(measured_points.shape)
        
            # Compute lengths along each measured ray.
            lengths = torch.linalg.norm(measured_points[:, -1, :] - measured_points[:, 0, :], dim=1)
        
            # Forward pass for measured rays (only once)
            attenuation_measured = self.forward(measured_points,batch_vecs).view(measured_points.shape[0], measured_points.shape[1])
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
        if self.imagefit_mode:
            return None
        else:
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
            outputs = torch.zeros_like(vol, dtype=torch.float)
            for i in range(mgrid.shape[0]):
                with torch.no_grad():
                    vec = (
                        self.latent.repeat(mgrid[i].shape[0], mgrid[i].shape[1], 1, 1)
                        .permute(2, 0, 1, 3)
                        .contiguous()
                        .view(-1, self.latent_size)
                    )
                    outputs[i] = self.forward(mgrid[i].view(-1, 3).cuda(), vec).view(
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

        if self.imagefit_mode:
            lr_lambda = lambda epoch: 0.97 ** max(0, (epoch - 10000))
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
                    {
                        "params": self.params.latent_vectors.parameters(),
                        "lr": self.latent_lr,
                    },
                ],
                amsgrad=True,
            )

        else:
            lr_lambda = lambda epoch: 0.97 ** max(0, (epoch - 30))
            if self.latent != None:
                if self.full_mode:
                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": self.params.model.parameters(),
                                "lr": self.model_lr,
                            },
                            {
                                "params": [self.latent],
                                "lr": self.latent_lr,
                            },
                        ],
                        amsgrad=True,
                    )
                else:
                    optimizer = torch.optim.AdamW(
                        [self.latent], lr=self.latent_lr, amsgrad=True
                    )
            else:
                optimizer = torch.optim.AdamW(
                    self.params.model.parameters(), lr=self.model_lr, amsgrad=True
                )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}



