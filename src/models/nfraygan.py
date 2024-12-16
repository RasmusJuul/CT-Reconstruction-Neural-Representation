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

@torch.jit.script
def gaussian(x: torch.Tensor,
             mean: torch.Tensor,
             std: torch.Tensor
            ) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(torch.tensor(2 * torch.pi)))

class GaussianLoss(nn.Module):
    def __init__(self, means, stds, weights):
        super(GaussianLoss, self).__init__()
        self.means = means
        self.stds = stds
        self.weights = weights

    def forward(self, x):
        loss = 0
        for mean, std, weight in zip(self.means, self.stds, self.weights):
            loss += weight * gaussian(x, mean, std)
        return -torch.log(loss+1e-8) + 0.6904994249343872 #adding 0.69.. so the result is zero for the peaks instead of negative, depends on the weight and std of the gaussian

class RayGAN(LightningModule):

    def __init__(self, args_dict, projection_shape=(16, 256, 256)):
        super(RayGAN, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.d_lr = args_dict["training"]["discriminator_lr"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.l1_regularization_weight = args_dict["training"]["regularization_weight"]

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
        self.l1_regularization = torch.nn.L1Loss()

        self.acc = tm.classification.BinaryAccuracy()

        self.validation_step_outputs = []
        self.validation_step_gt = []

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

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
        points, target, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points = batch
        
        
        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_g)
        
        lengths = torch.linalg.norm((points[:, -1, :] - points[:, 0, :]), dim=1)
        attenuation_values = self.forward(points).view(points.shape[0], points.shape[1])
        detector_value_hat = compute_projection_values(
            points.shape[1], attenuation_values, lengths
        )

        loss = self.loss_fn(detector_value_hat, target)

        smoothness_loss = self.l1_regularization(
            attenuation_values[:, 1:], attenuation_values[:, :-1]
        )  # punish model for big changes between adjacent points (to make it smooth)
        total_loss = loss + self.l1_regularization_weight * smoothness_loss
        
        if self.current_epoch > 20:
            valid = torch.ones(attenuation_values.size(0), 1)
            valid = valid.type_as(attenuation_values)
            g_loss = 1e-1 * self.adversarial_loss(self.Discriminator(attenuation_values,position,start_points,end_points), valid)
            total_loss += g_loss


        self.manual_backward(total_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        if self.current_epoch > 20:
            if self.current_epoch%2 == 0:
                # train discriminator
                # Measure discriminator's ability to classify real from generated samples
                self.toggle_optimizer(optimizer_d)
        
                # how well can it label as real?
                valid = torch.ones(real_ray.size(0), 1)
                valid = valid.type_as(real_ray)
                pred_target = self.Discriminator(real_ray,real_position,real_start_points,real_end_points)
                real_loss = self.adversarial_loss(pred_target, valid)
                self.acc(pred_target, valid)
        
                # how well can it label as fake?
                fake = torch.zeros(attenuation_values.size(0), 1)
                fake = fake.type_as(attenuation_values)
                pred_generated = self.Discriminator(attenuation_values.detach(),position,start_points,end_points)
                fake_loss = self.adversarial_loss(pred_generated, fake)
                self.acc(pred_generated,fake)
                
                # discriminator loss is the average of these
                d_loss = (real_loss + fake_loss) / 2
                if torch.any(torch.isnan(d_loss)):
                    print("nan in d_loss")
                    raise ValueError('Nan in output')
                self.manual_backward(d_loss)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)
    
    
                self.log_dict(
                        {
                            "train/loss": loss,
                            "train/generator_loss": g_loss,
                            "train/total_loss": total_loss,
                            "train/discriminator_loss": d_loss,
                            "train/acc":self.acc,
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
                            "train/total_loss": total_loss,
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
                        "train/total_loss": total_loss,
                    },
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
        if self.trainer.is_last_batch:
            sch1, sch2 = self.lr_schedulers()
            sch1.step()
            sch2.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        points, target, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points = batch
        
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

        for i in np.random.randint(0, self.projection_shape[0], 2):
            self.logger.log_image(
                key="val/projection",
                images=[preds[i], gt[i], (gt[i] - preds[i])],
                caption=[f"pred_{i}", f"gt_{i}", f"residual_{i}"],
            )  # log projection images

        mgrid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, vol_shape[0]),
                torch.linspace(-1, 1, vol_shape[1]),
                torch.linspace(-1, 1, vol_shape[2]),
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
                outputs[vol_shape[0] // 2, :, :],
                vol[vol_shape[0] // 2, :, :],
                (vol[vol_shape[0] // 2, :, :] - outputs[vol_shape[0] // 2, :, :]),
                outputs[:, vol_shape[1] // 2, :],
                vol[:, vol_shape[1] // 2, :],
                (vol[:, vol_shape[1] // 2, :] - outputs[:, vol_shape[1] // 2, :]),
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
        optimizer_g = torch.optim.AdamW(
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
            
            
        optimizer_d = torch.optim.AdamW(self.Discriminator.parameters(),lr=self.d_lr,amsgrad=True)
        
        lr_lambda_g = lambda epoch: 0.99 ** max(0, (epoch - 50))
        lr_lambda_d = lambda epoch: 0.98 ** max(0, (epoch - 25))
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda_g),
            'interval': 'epoch',
            'frequency': 1
        }
    
        scheduler_d = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda_d),
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer_g,optimizer_d],[scheduler_g,scheduler_d]


# ray discriminator (used for fibers)
class Discriminator(torch.nn.Module):
    def __init__(self, num_points):
        super().__init__()
        
        self.conv_ray = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2,2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2,2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
        )
        self.mlp = nn.Sequential(
            nn.Linear((num_points//4)+3+3+12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, ray, position, start_point,end_point):
        conv_out = self.conv_ray(ray.unsqueeze(dim=1))
        mlp_input = torch.cat([conv_out.squeeze(),position,start_point,end_point],dim=1)
        validity = self.mlp(mlp_input)

        return validity


class NeuralGAN(LightningModule):

    def __init__(self, args_dict, projection_shape=(16, 256, 256),vol_shape=(256,256,256)):
        super(NeuralGAN, self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict["training"]["model_lr"]
        self.d_lr = args_dict["training"]["discriminator_lr"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict["training"]["batch_size"]

        self.l1_regularization_weight = args_dict["training"]["regularization_weight"]

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

        self.Discriminator = Discriminator(args_dict["training"]["num_points"]).cuda()
        
        self.SliceDiscriminator = SliceDiscriminator().cuda()
        
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
        self.l1_regularization = torch.nn.L1Loss()

        means = torch.tensor([0.,0.1, 1.0],device="cuda")
        stds = torch.tensor([0.02, 0.02, 0.02],device="cuda")
        weights = torch.tensor([0.08, 0.099999, 0.1],device="cuda")
        self.gaussian_loss = GaussianLoss(means, stds, weights)

        self.acc_ray = tm.classification.BinaryAccuracy()
        self.acc_slice = tm.classification.BinaryAccuracy()
        

        self.validation_step_outputs = []
        self.validation_step_gt = []

        self.mgrid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, vol_shape[0]),
                torch.linspace(-1, 1, vol_shape[1]),
                torch.linspace(-1, 1, vol_shape[2]),
                indexing="ij",
            ),
            dim=-1,
        ).to(dtype=torch.float, device=self.device)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

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
        points, target, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points, real_slices, embedding = batch
        
        
        optimizer_g, optimizer_d_ray,optimizer_d_slice = self.optimizers()
        self.toggle_optimizer(optimizer_g)
        
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

        # loss += 1e-2 * self.gaussian_loss(attenuation_values).mean()
        
        valid = torch.ones(attenuation_values.size(0), 1, device=self.device)
        valid = valid.type_as(attenuation_values)
        ray_loss = 1e-2 * self.adversarial_loss(self.Discriminator(attenuation_values,position,start_points,end_points), valid)
        
        slice_xy = self.forward(self.mgrid[int(256*embedding[0,0]),:,:].reshape(-1, 3).to(device=self.device)).view(256,256)
        slice_xz = self.forward(self.mgrid[:,int(256*embedding[1,1]),:].reshape(-1, 3).to(device=self.device)).view(256,256)
        slice_yz = self.forward(self.mgrid[:,:,int(256*embedding[2,2])].reshape(-1, 3).to(device=self.device)).view(256,256)
        slices = torch.stack((slice_xy,slice_xz,slice_yz)).unsqueeze(dim=1).to(device=self.device)

        slice_loss = 1e-1 * self.adversarial_loss(self.SliceDiscriminator(slices,embedding), torch.ones(3,1, device=self.device).type_as(slices))

        total_loss = loss + ray_loss + slice_loss
            
        
        self.manual_backward(total_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        if self.current_epoch%2 == 0:
            # train ray discriminator
            # Measure discriminator's ability to classify real from generated samples
            self.toggle_optimizer(optimizer_d_ray)
    
            # how well can it label as real?
            valid = torch.ones(real_ray.size(0), 1, device=self.device).type_as(real_ray)
            pred_target = self.Discriminator(real_ray,real_position,real_start_points,real_end_points)
            real_loss = self.adversarial_loss(pred_target, valid)
            self.acc_ray(pred_target, valid)
    
            # how well can it label as fake?
            fake = torch.zeros(attenuation_values.size(0), 1, device=self.device).type_as(attenuation_values)
            pred_generated = self.Discriminator(attenuation_values.detach(),position,start_points,end_points)
            fake_loss = self.adversarial_loss(pred_generated, fake)
            self.acc_ray(pred_generated,fake)
            
            # discriminator loss is the average of these
            d_ray_loss = (real_loss + fake_loss) / 2
            if torch.any(torch.isnan(d_ray_loss)):
                print("nan in d_loss")
                raise ValueError('Nan in output')
            self.manual_backward(d_ray_loss)
            optimizer_d_ray.step()
            optimizer_d_ray.zero_grad()
            self.untoggle_optimizer(optimizer_d_ray)
    
            # train slice discriminator
            # Measure discriminator's ability to classify real from generated samples
            self.toggle_optimizer(optimizer_d_slice)
    
            # how well can it label as real?
            valid = torch.ones(3, 1, device=self.device).type_as(real_ray)
            pred_target = self.SliceDiscriminator(real_slices,embedding)
            real_loss = self.adversarial_loss(pred_target, valid)
            self.acc_slice(pred_target, valid)
    
            # how well can it label as fake?
            fake = torch.zeros(3, 1, device=self.device).type_as(slices)
            pred_generated = self.SliceDiscriminator(slices.detach(), embedding)
            fake_loss = self.adversarial_loss(pred_generated, fake)
            self.acc_slice(pred_generated,fake)
            
            # discriminator loss is the average of these
            d_slice_loss = (real_loss + fake_loss) / 2
            if torch.any(torch.isnan(d_slice_loss)):
                print("nan in d_loss")
                raise ValueError('Nan in output')
            self.manual_backward(d_slice_loss)
            optimizer_d_slice.step()
            optimizer_d_slice.zero_grad()
            self.untoggle_optimizer(optimizer_d_slice)
    
    
            self.log_dict(
                    {
                        "train/loss": loss,
                        "train/ray_loss": ray_loss,
                        "train/slice_loss": slice_loss,
                        "train/total_loss": total_loss,
                        "train/discriminator_ray_loss": d_ray_loss,
                        "train/discriminator_slice_loss": d_slice_loss,
                        "train/acc_ray":self.acc_ray,
                        "train/acc_slice":self.acc_slice,
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
                        "train/ray_loss": ray_loss,
                        "train/slice_loss": slice_loss,
                        "train/total_loss": total_loss,
                    },
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.batch_size,
                )
        
        return loss

    def validation_step(self, batch, batch_idx):
        points, target, position, start_points, end_points, real_ray, real_position, real_start_points, real_end_points, real_slices, embedding = batch
        
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

        for i in np.random.randint(0, self.projection_shape[0], 2):
            self.logger.log_image(
                key="val/projection",
                images=[preds[i], gt[i], (gt[i] - preds[i])],
                caption=[f"pred_{i}", f"gt_{i}", f"residual_{i}"],
            )  # log projection images

        
        
        outputs = torch.zeros_like(vol)
        for i in range(self.mgrid.shape[0]):
            with torch.no_grad():
                outputs[i] = self.forward(self.mgrid[i].view(-1, 3).to(device=self.device)).view(
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
                outputs[vol_shape[0] // 2, :, :],
                vol[vol_shape[0] // 2, :, :],
                (vol[vol_shape[0] // 2, :, :] - outputs[vol_shape[0] // 2, :, :]),
                outputs[:, vol_shape[1] // 2, :],
                vol[:, vol_shape[1] // 2, :],
                (vol[:, vol_shape[1] // 2, :] - outputs[:, vol_shape[1] // 2, :]),
                outputs[:, :, vol_shape[2] // 2],
                vol[:, :, vol_shape[2] // 2],
                (vol[:, :, vol_shape[2] // 2] - outputs[:, :, vol_shape[2] // 2]),
            ],
            caption=[
                "pred_yz",
                "gt_yz",
                "residual_yz",
                "pred_xz",
                "gt_xz",
                "residual_xz",
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
            
            
        optimizer_d_ray = torch.optim.AdamW(self.Discriminator.parameters(),lr=self.d_lr,amsgrad=True)
        optimizer_d_slice = torch.optim.AdamW(self.SliceDiscriminator.parameters(),lr=self.d_lr,amsgrad=True)

        
        return [optimizer_g,optimizer_d_ray,optimizer_d_slice],[]


# 256x256
class SliceDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4,32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32,1)
        )
    def forward(self, img, embedding):
        conv_out = self.model(img).squeeze().unsqueeze(dim=0)
        x = torch.cat((conv_out,embedding)).T
        validity = self.classifier(x)
        
        return validity
