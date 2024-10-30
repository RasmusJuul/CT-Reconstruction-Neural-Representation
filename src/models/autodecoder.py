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

        self.l1_regularization_weight = args_dict["training"]["regularization_weight"]

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

        config = {
            "encoding_hashgrid": {
                "otype": "HashGrid",
                "n_levels": 32,
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
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128,
                "n_hidden_layers": 6
            },
        }
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=config[f"encoding_{args_dict['model']['encoder']}"])
            num_input_features = self.encoder.n_output_dims + self.latent_size
        else:
            self.encoder = None
            num_input_features = 3 + self.latent_size  # x,y,z coordinate

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
                        torch.nn.Dropout(p=0.1),
                    )
                )
            else:
                layers_second_half.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            self.num_hidden_features, self.num_hidden_features
                        ),
                        get_activation_function(self.activation_function, args_dict),
                        torch.nn.Dropout(p=0.1),
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
                "latent_vectors": torch.nn.ModuleList([self.lat_vecs]),
                "encoder":torch.nn.ModuleList([self.encoder]),
                "model": torch.nn.ModuleList(
                    [self.mlp_first_half, self.mlp_second_half]
                ),
            }
        )

        self.loss_fn = torch.nn.MSELoss()
        self.volumefit_loss = torch.nn.L1Loss()
        self.l1_regularization = torch.nn.L1Loss()
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
        points, target, idxs = batch
        if self.imagefit_mode:
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
            batch_vecs = (
                self.latent.repeat(points.shape[0], points.shape[1], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
            attenuation_values = self.forward(points, batch_vecs).view(
                points.shape[0], points.shape[1]
            )
            detector_value_hat = compute_projection_values(
                points.shape[1], attenuation_values, lengths
            )

            loss = self.loss_fn(detector_value_hat, target)

            smoothness_loss = self.l1_regularization(
                attenuation_values[:, 1:], attenuation_values[:, :-1]
            )  # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight * smoothness_loss

            loss += 1e-4 * torch.mean(self.latent.pow(2))

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
                self.train_epoch_loss = 0
                torch.save(
                    self.latent, f"{self.data_path}/latent_vector-{self.latent_size}.pt"
                )

    def validation_step(self, batch, batch_idx):
        points, target, idxs = batch
        if self.imagefit_mode:
            return None
        else:
            batch_vecs = (
                self.latent.repeat(points.shape[0], points.shape[1], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
            attenuation_values = self.forward(points, batch_vecs).view(
                points.shape[0], points.shape[1]
            )
            detector_value_hat = compute_projection_values(
                points.shape[1], attenuation_values, lengths
            )

            loss = self.loss_fn(detector_value_hat, target)

            smoothness_loss = self.l1_regularization(
                attenuation_values[:, 1:], attenuation_values[:, :-1]
            )  # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight * smoothness_loss

            loss += 1e-4 * torch.mean(self.latent.pow(2))

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


class AutoDecoder_adversarial(LightningModule):

    def __init__(
        self, args_dict, projection_shape=(300, 300), num_volumes=25, latent=None
    ):
        super(AutoDecoder_adversarial, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.latent_lr = args_dict["training"]["latent_lr"]
        self.imagefit_mode = args_dict["training"]["imagefit_mode"]
        self.full_mode = args_dict["training"]["full_mode"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.l1_regularization_weight = args_dict["training"]["regularization_weight"]

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

        config = {
            "encoding_hashgrid": {
                "otype": "HashGrid",
                "n_levels": 32,
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
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128,
                "n_hidden_layers": 6
            },
        }
        
        # Initialising encoder
        if args_dict["model"]["encoder"] != None:
            self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])
            num_input_features = self.encoder.n_output_dims + self.latent_size
            
        else:
            self.encoder = None
            num_input_features = 3 + self.latent_size  # x,y,z coordinate


        # network = tcnn.Network(n_input_dims=encoding.n_output_dims+128, n_output_dims=1, network_config=config["network"])
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
                        torch.nn.Dropout(p=0.1),
                    )
                )
            else:
                layers_second_half.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            self.num_hidden_features, self.num_hidden_features
                        ),
                        get_activation_function(self.activation_function, args_dict),
                        torch.nn.Dropout(p=0.1),
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

        self.vol_size = args_dict["model"]["volume_sidelength"]
        self.Discriminator = Discriminator((self.vol_size[0],self.vol_size[1]))
        
        self.params = torch.nn.ModuleDict(
            {
                "latent_vectors": torch.nn.ModuleList([self.lat_vecs]),
                "encoder":torch.nn.ModuleList([self.encoder]),
                "model": torch.nn.ModuleList(
                    [self.mlp_first_half, self.mlp_second_half]
                ),
                "Discriminator": torch.nn.ModuleList([self.Discriminator]),
            }
        )
        
        
        
        self.automatic_optimization = False
        
        self.loss_fn = torch.nn.MSELoss()
        self.volumefit_loss = torch.nn.L1Loss()
        self.l1_regularization = torch.nn.L1Loss()
        self.ms_ssim = tm.image.MultiScaleStructuralSimilarityIndexMeasure(
                        data_range=1.0,
                        gaussian_kernel=True,
                        kernel_size=3,
                        sigma=1.5,
                        reduction='elementwise_mean',
                        k1=0.01, 
                        k2=0.03,
                        betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                        normalize='relu')

        self.acc = tm.classification.BinaryAccuracy()
        self.alpha = 0.2
        self.validation_step_outputs = []
        self.validation_step_gt = []

        self.random_sample_idx = np.random.randint(self.num_volumes)
        self.prediction_img = []
        self.gt_img = []
        self.generated_img = []

        self.smallest_train_loss = torch.inf
        self.train_epoch_loss = 0

        self.current_step_ = 0

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

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
        points, target, idxs = batch
        if self.imagefit_mode:
            optimizer_nf,optimizer_g, optimizer_d = self.optimizers()
            # train generator
            # generate images
            self.toggle_optimizer(optimizer_nf)
            batch_vecs = (
                self.lat_vecs(idxs)
                .repeat(points.shape[1], points.shape[2], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            attenuation_values = self.forward(points, batch_vecs)
            attenuation_values = attenuation_values.view(target.shape)

            l1 = self.volumefit_loss(attenuation_values, target)
            # loss = l1
            
            msssim = (1 - self.ms_ssim(attenuation_values.unsqueeze(dim=1),
                                       target.unsqueeze(dim=1)))
            loss = self.alpha * msssim + (1 - self.alpha) * l1

            l2_size_loss = torch.sum(torch.norm(self.lat_vecs(idxs), dim=1))
            reg_loss = (1e-4 * min(1, self.current_epoch / 100) * l2_size_loss) / len(
                idxs
            )

            loss += reg_loss.cuda()
            
            if len(self.prediction_img) == 0:
                if self.random_sample_idx in idxs:
                    for i in (idxs == self.random_sample_idx).nonzero(as_tuple=True)[0]:
                        self.prediction_img.append(attenuation_values[i])
                        self.gt_img.append(target[i])

            if torch.any(torch.isnan(loss)):
                print("nan in loss")
                raise ValueError('Nan in output')

            self.manual_backward(loss)
            optimizer_nf.step()
            optimizer_nf.zero_grad()
            self.untoggle_optimizer(optimizer_nf)

            self.toggle_optimizer(optimizer_g)
            vecs = self.lat_vecs(idxs)
            random_vectors = self.lat_vecs(torch.tensor(
                                        np.random.choice(
                                            np.arange(self.lat_vecs.num_embeddings),
                                            idxs.shape[0]),
                                        device="cuda")
                                    ).detach()
            vec_inter = torch.lerp(vecs,random_vectors,torch.rand(1,device="cuda"))
            vectors = torch.normal(mean=vec_inter,std=self.lat_vecs.weight.std(dim=0).detach()*0.5)
            
            
            batch_vecs = (
            vectors
            .repeat(points.shape[1], points.shape[2], 1, 1)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, self.latent_size)
            )
            generated_slices = self.forward(points, batch_vecs)
            generated_slices = generated_slices.view(target.shape).unsqueeze(dim=1)
            
            if len(self.generated_img) == 0:
                self.generated_img.append(generated_slices.detach().cpu())
            
            valid = torch.ones(target.size(0), 1)
            valid = valid.type_as(target)

            if self.current_epoch > 10:
                # adversarial loss is binary cross-entropy
                g_loss = self.adversarial_loss(self.Discriminator(generated_slices), valid)
                self.manual_backward(g_loss)
                optimizer_g.step()
                optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
    
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            self.toggle_optimizer(optimizer_d)
    
            # how well can it label as real?
            pred_target = self.Discriminator(target.unsqueeze(dim=1))
            real_loss = self.adversarial_loss(pred_target, valid)
            acc_real = self.acc(pred_target, valid)
    
            # how well can it label as fake?
            fake = torch.zeros(target.size(0), 1)
            fake = fake.type_as(target)
            pred_generated = self.Discriminator(generated_slices.detach())
            fake_loss = self.adversarial_loss(pred_generated, fake)
            acc_fake =  self.acc(pred_generated,fake)
            
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            if torch.any(torch.isnan(d_loss)):
                print("nan in d_loss")
                raise ValueError('Nan in output')
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            if self.current_epoch > 10:
                self.log_dict(
                    {
                        "train/loss": loss,
                        "train/g_loss": g_loss,
                        "train/d_loss": d_loss,
                        "train/acc_fake":acc_fake,
                        "train/acc_real":acc_real,
                    },
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
            else:
                self.log_dict(
                    {
                        "train/loss": loss,
                        "train/d_loss": d_loss,
                        "train/acc_fake":acc_fake,
                        "train/acc_real":acc_real,
                    },
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
            
            
            return loss
        
        else:
            batch_vecs = (
                self.latent.repeat(points.shape[0], points.shape[1], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
            attenuation_values = self.forward(points, batch_vecs).view(
                points.shape[0], points.shape[1]
            )
            detector_value_hat = compute_projection_values(
                points.shape[1], attenuation_values, lengths
            )

            loss = self.loss_fn(detector_value_hat, target)

            smoothness_loss = self.l1_regularization(
                attenuation_values[:, 1:], attenuation_values[:, :-1]
            )  # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight * smoothness_loss

            loss += 1e-4 * torch.mean(self.latent.pow(2))

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
            
            grid = torchvision.utils.make_grid(self.generated_img[0])
            self.logger.log_image(key="train/generated_images",
                                  images=[grid],
                                  caption=[self.current_epoch])
            
            # free memory
            self.prediction_img.clear()
            self.gt_img.clear()
            self.generated_img.clear()
            # Update index
            self.random_sample_idx = np.random.randint(self.num_volumes)
        else:
            self.train_epoch_loss /= len(self.trainer.train_dataloader)
            if self.train_epoch_loss < self.smallest_train_loss:
                self.smallest_train_loss = self.train_epoch_loss
                self.train_epoch_loss = 0
                torch.save(
                    self.latent, f"{self.data_path}/latent_vector-{self.latent_size}.pt"
                )

    def validation_step(self, batch, batch_idx):
        points, target, idxs = batch
        if self.imagefit_mode:
            return None
        else:
            batch_vecs = (
                self.latent.repeat(points.shape[0], points.shape[1], 1, 1)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1, self.latent_size)
            )
            lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
            attenuation_values = self.forward(points, batch_vecs).view(
                points.shape[0], points.shape[1]
            )
            detector_value_hat = compute_projection_values(
                points.shape[1], attenuation_values, lengths
            )

            loss = self.loss_fn(detector_value_hat, target)

            smoothness_loss = self.l1_regularization(
                attenuation_values[:, 1:], attenuation_values[:, :-1]
            )  # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight * smoothness_loss

            loss += 1e-4 * torch.mean(self.latent.pow(2))

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
            optimizer_g = torch.optim.AdamW(
                [
                    {
                        "params": self.params.model.parameters(),
                        "lr": 3e-6,
                    },
                ],
                amsgrad=True,
            )
            
            optimizer_d = torch.optim.AdamW(self.Discriminator.parameters(),lr=1e-5,amsgrad=True)

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
        if self.imagefit_mode:
            return [optimizer,optimizer_g,optimizer_d],[]
        else:
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

# 256x256
class Discriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1),
        )

    def forward(self, img):
        validity = self.model(img).squeeze().unsqueeze(dim=1)
        

        return validity

# 100x100
# class Discriminator(torch.nn.Module):
#     def __init__(self, img_shape):
#         super().__init__()
        
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(5,5),
#             nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(5,5),
#             nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(4,4),
#             nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1),
#         )

#     def forward(self, img):
#         validity = self.model(img).squeeze().unsqueeze(dim=1)
        

#         return validity
