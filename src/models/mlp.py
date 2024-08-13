import torch
import math
import numpy as np
from pytorch_lightning import LightningModule
import torchmetrics as tm
import tifffile
from tqdm import tqdm
# import tinycudann as tcnn
from src.encoder import get_encoder
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT

def get_activation_function(activation_function,args_dict,**kwargs):
    if activation_function == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif activation_function == 'leaky_relu':
        return torch.nn.LeakyReLU(**kwargs)
    elif activation_function == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)
    elif activation_function == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif activation_function == 'elu':
        return torch.nn.ELU(**kwargs)
    elif activation_function == 'none':
        return torch.nn.Identity(**kwargs)
    elif activation_function == 'sine':
        return torch.jit.script(Sine(**kwargs)).to(device=args_dict['training']['device'])
    else:
        raise ValueError(f"Unknown activation function: {activation_function}")
        
class Sine(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input: torch.Tensor
               ) -> torch.Tensor:
        # See Siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # In siren paper see supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

@torch.jit.script
def compute_projection_values(num_points: int,
                              attenuation_values: torch.Tensor,
                              lengths: torch.Tensor,
                             ) -> torch.Tensor:
    I0 = 1
    # Compute the spacing between ray points
    dx = lengths / (num_points)

    # Compute the sum of mu * dx along each ray
    attenuation_sum = torch.sum(attenuation_values * dx[:,None], dim=1)

    return attenuation_sum

class MLP(LightningModule):

    def __init__(self,args_dict,projection_shape=(300,300),num_volumes=1000,latent=None):
        super(MLP,self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.model_lr = args_dict['training']['model_lr']
        self.latent_lr = args_dict['training']['latent_lr']
        self.imagefit_mode = args_dict['training']['imagefit_mode']
        self.full_mode = args_dict["training"]["full_mode"]
        self.data_path = f"{_PATH_DATA}/{args_dict['general']['data_path']}"
        self.batch_size = args_dict['training']['batch_size']

        self.l1_regularization_weight = args_dict['training']['regularization_weight']

        self.num_freq_bands = args_dict['model']['num_freq_bands']
        self.num_hidden_layers = args_dict['model']['num_hidden_layers']
        self.num_hidden_features = args_dict['model']['num_hidden_features']
        self.activation_function = args_dict['model']['activation_function']
        self.latent_size = args_dict['model']['latent_size']
        self.num_volumes = num_volumes
        self.latent = latent
        
        # Initialising latent vectors
        self.lat_vecs = torch.nn.Embedding(num_volumes,self.latent_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data,
                              0.0,
                              1 / math.sqrt(self.latent_size),
                            )
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = get_encoder(encoding=args_dict['model']['encoder'])
            num_input_features = self.encoder.output_dim+self.latent_size
        else:
            self.encoder = None
            num_input_features = 3+self.latent_size # x,y,z coordinate

        

        layers = []
        for i in range(self.num_hidden_layers):
            layers.append(torch.nn.Sequential(torch.nn.Linear(self.num_hidden_features,self.num_hidden_features),
                                         get_activation_function(self.activation_function,args_dict),
                                         ))


        self.mlp = torch.nn.Sequential(torch.nn.Linear(num_input_features,self.num_hidden_features),
                                       get_activation_function(self.activation_function,args_dict),
                                       *layers,
                                        torch.nn.Linear(self.num_hidden_features,1),
                                        torch.nn.Sigmoid(),
                                        )

        if self.activation_function == 'sine':
            self.mlp.apply(sine_init)
            self.mlp[0].apply(first_layer_sine_init)


        self.params = torch.nn.ModuleDict({
            'latent_vectors': torch.nn.ModuleList([self.lat_vecs]),
            'model': torch.nn.ModuleList([self.encoder,self.mlp])})

        self.loss_fn = torch.nn.MSELoss()
        self.volumefit_loss = torch.nn.L1Loss()
        self.l1_regularization = torch.nn.L1Loss()
        self.psnr = tm.image.PeakSignalNoiseRatio()
        self.validation_step_outputs = []
        self.validation_step_gt = []
        self.msssim = tm.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

        self.random_sample_idx = np.random.randint(self.num_volumes)
        self.prediction_img = []
        self.gt_img = []

        self.smallest_train_loss = torch.inf
        self.train_epoch_loss = 0
        
    def forward(self, pts, vecs):
        pts_shape = pts.shape

        if len(pts.shape) > 2:
            pts = pts.view(-1,3)

        if self.encoder != None:
            enc = self.encoder(pts)
        else:
            enc = pts
            
        enc = torch.cat([vecs, enc], dim=1)

        out = self.mlp(enc)

        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1],-1)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        points, target, idxs = batch
        if self.imagefit_mode:
            batch_vecs = self.lat_vecs(idxs).repeat(points.shape[1],points.shape[2],1,1).permute(2,0,1,3).contiguous().view(-1,self.latent_size)
            attenuation_values = self.forward(points,batch_vecs)
            attenuation_values = attenuation_values.view(target.shape)
                
            loss = self.volumefit_loss(attenuation_values,target)
    
            msssim_loss = 0.84*(1-self.msssim(attenuation_values.unsqueeze(dim=1),target.unsqueeze(dim=1)))
    
            contrast_loss = 1e-3*(1-(attenuation_values.max() - attenuation_values.min()))
    
            l2_size_loss = torch.sum(torch.norm(self.lat_vecs(idxs), dim=1))
            reg_loss = (
                1e-4 * min(1, self.current_epoch / 100) * l2_size_loss
            ) / len(idxs)
    
            loss += reg_loss.cuda()
            loss += contrast_loss.cuda()
            loss += msssim_loss.cuda()
            
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
                    for i in ((idxs == self.random_sample_idx).nonzero(as_tuple=True)[0]):
                        self.prediction_img.append(attenuation_values[i])
                        self.gt_img.append(target[i])
                
            return loss
        else:
            batch_vecs = self.latent.repeat(points.shape[0],points.shape[1],1,1).permute(2,0,1,3).contiguous().view(-1,self.latent_size)
            lengths = torch.linalg.norm((points[:,-1,:] - points[:,0,:]),dim=1)
            attenuation_values = self.forward(points,batch_vecs).view(points.shape[0],points.shape[1])
            detector_value_hat = compute_projection_values(points.shape[1],attenuation_values,lengths)

            loss = self.loss_fn(detector_value_hat, target)
            
            smoothness_loss = self.l1_regularization(attenuation_values[:,1:],attenuation_values[:,:-1]) # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight*smoothness_loss
            
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
            self.logger.log_image(key="train/image", images=[self.prediction_img[0], self.gt_img[0], (self.gt_img[0]-self.prediction_img[0])], caption=[f"pred_{self.random_sample_idx}", f"gt_{self.random_sample_idx}",f"residual_{self.random_sample_idx}"]) 
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
                torch.save(self.latent, f"{self.data_path}/latent_vector-{self.latent_size}.pt")

    def validation_step(self, batch, batch_idx):
        points, target, idxs = batch
        if self.imagefit_mode:
            return None
        else:
            batch_vecs = self.latent.repeat(points.shape[0],points.shape[1],1,1).permute(2,0,1,3).contiguous().view(-1,self.latent_size)
            lengths = torch.linalg.norm((points[:,-1,:] - points[:,0,:]),dim=1)
            attenuation_values = self.forward(points,batch_vecs).view(points.shape[0],points.shape[1])
            detector_value_hat = compute_projection_values(points.shape[1],attenuation_values,lengths)

            loss = self.loss_fn(detector_value_hat, target)
            
            smoothness_loss = self.l1_regularization(attenuation_values[:,1:],attenuation_values[:,:-1]) # punish model for big changes between adjacent points (to make it smooth)
            loss += self.l1_regularization_weight*smoothness_loss
            
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
            valid_rays = self.trainer.val_dataloaders.dataset.valid_rays.view(self.projection_shape)
            preds = torch.zeros(self.projection_shape,dtype=all_preds.dtype)
            preds[valid_rays] = all_preds
            gt = torch.zeros(self.projection_shape,dtype=all_gt.dtype)
            gt[valid_rays] = all_gt

            for i in np.random.randint(0,self.projection_shape[0],5):
                self.logger.log_image(key="val/projection", images=[preds[i], gt[i], (gt[i]-preds[i])], caption=[f"pred_{i}", f"gt_{i}",f"residual_{i}"]) # log projection images

            mgrid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, vol_shape[0]), torch.linspace(-1, 1, vol_shape[1]), torch.linspace(-1, 1, vol_shape[1]), indexing='ij'),dim=-1)
            outputs = torch.zeros_like(vol,dtype=torch.float)
            for i in range(mgrid.shape[0]):
                with torch.no_grad():
                    vec = self.latent.repeat(mgrid[i].shape[0],mgrid[i].shape[1],1,1).permute(2,0,1,3).contiguous().view(-1,self.latent_size)
                    outputs[i] = self.forward(mgrid[i].view(-1,3).cuda(), vec).view(outputs[i].shape)

            self.log("val/loss_reconstruction",self.loss_fn(outputs,vol),batch_size=self.batch_size)
            self.logger.log_image(key="val/reconstruction", images=[outputs[vol_shape[2]//2,:,:], vol[vol_shape[2]//2,:,:], (vol[vol_shape[2]//2,:,:]-outputs[vol_shape[2]//2,:,:]),
                                                                    outputs[:,vol_shape[2]//2,:], vol[:,vol_shape[2]//2,:], (vol[:,vol_shape[2]//2,:]-outputs[:,vol_shape[2]//2,:]),
                                                                    outputs[:,:,vol_shape[2]//2], vol[:,:,vol_shape[2]//2], (vol[:,:,vol_shape[2]//2]-outputs[:,:,vol_shape[2]//2])],
                                  caption=["pred_xy", "gt_xy","residual_xy",
                                           "pred_yz", "gt_yz","residual_yz",
                                           "pred_xz", "gt_xz","residual_xz",])
                

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_gt.clear()  # free memory

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        
        if self.imagefit_mode:
            lr_lambda = lambda epoch: 0.97 ** epoch
            optimizer = torch.optim.AdamW([
                    {
                        "params": self.params.model.parameters(),
                        "lr": self.model_lr,
                    },
                    {
                        "params": self.params.latent_vectors.parameters(),
                        "lr": self.latent_lr,
                    },
                ], amsgrad=True)
            
        else:
            lr_lambda = lambda epoch: 0.97 ** max(0,(epoch-30))
            if self.latent != None:
                if self.full_mode:
                    optimizer = torch.optim.AdamW([
                        {
                            "params": self.params.model.parameters(),
                            "lr": self.model_lr,
                        },
                        {
                            "params": [self.latent],
                            "lr": self.latent_lr,
                        },
                    ], amsgrad=True)
                else:
                    optimizer = torch.optim.AdamW([self.latent],lr=self.latent_lr, amsgrad=True)
            else:
                optimizer = torch.optim.AdamW(self.params.model.parameters(),lr=self.model_lr, amsgrad=True)
                
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class MLP_2d(LightningModule):

    def __init__(self,args_dict,projection_shape,num_volumes):
        super(MLP_2d,self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.lr = args_dict['training']['learning_rate']
        self.imagefit_mode = args_dict["training"]["imagefit_mode"]

        self.l1_regularization_weight = args_dict['training']['regularization_weight']

        self.num_freq_bands = args_dict['model']['num_freq_bands']
        self.num_hidden_layers = args_dict['model']['num_hidden_layers']
        self.num_hidden_features = args_dict['model']['num_hidden_features']
        self.activation_function = args_dict['model']['activation_function']
        self.latent_size = args_dict['model']['latent_size']
        self.num_volumes = num_volumes

        
        # Initialising latent vectors
        self.lat_vecs = torch.nn.Embedding(self.num_volumes,self.latent_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data,
                              0.0,
                              1 / math.sqrt(self.latent_size),
                            )
        
        # Initialising encoder
        if args_dict['model']['encoder'] != None:
            self.encoder = get_encoder(encoding=args_dict['model']['encoder'],input_dim=2)
            num_input_features = self.encoder.output_dim+self.latent_size
        else:
            self.encoder = None
            num_input_features = 2+self.latent_size # x,y,z coordinate

        

        layers = []
        for i in range(self.num_hidden_layers):
            layers.append(torch.nn.Sequential(torch.nn.Linear(self.num_hidden_features,self.num_hidden_features),
                                         get_activation_function(self.activation_function,args_dict),
                                         ))


        self.mlp = torch.nn.Sequential(torch.nn.Linear(num_input_features,self.num_hidden_features),
                                       get_activation_function(self.activation_function,args_dict),
                                       *layers,
                                        torch.nn.Linear(self.num_hidden_features,1),
                                        torch.nn.Sigmoid(),
                                        )

        if self.activation_function == 'sine':
            self.mlp.apply(sine_init)
            self.mlp[0].apply(first_layer_sine_init)


        self.params = torch.nn.ModuleDict({
            'latent_vectors': torch.nn.ModuleList([self.lat_vecs]),
            'model': torch.nn.ModuleList([self.encoder,self.mlp])})

        self.loss_fn = torch.nn.L1Loss()
        self.msssim = tm.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

        self.random_sample_idx = np.random.randint(self.num_volumes)
        self.prediction_img = []
        self.gt_img = []
        
    def forward(self, pts, vecs):
        pts_shape = pts.shape

        if len(pts.shape) > 2:
            pts = pts.view(-1,2)

        if self.encoder != None:
            enc = self.encoder(pts)
        else:
            enc = pts
            
        enc = torch.cat([vecs, enc], dim=1)

        out = self.mlp(enc)

        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1],-1)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        points, target, idxs = batch
        batch_vecs = self.lat_vecs(idxs).repeat(points.shape[1],points.shape[2],1,1).permute(2,0,1,3).contiguous().view(-1,self.latent_size)
        attenuation_values = self.forward(points,batch_vecs)
        attenuation_values = attenuation_values.view(target.shape)
            
        loss = self.loss_fn(attenuation_values,target)

        msssim_loss = 0.84*(1-self.msssim(attenuation_values.unsqueeze(dim=1),target.unsqueeze(dim=1)))

        contrast_loss = 1e-3*(1-(attenuation_values.max() - attenuation_values.min()))

        l2_size_loss = torch.sum(torch.norm(self.lat_vecs(idxs), dim=1))
        reg_loss = (
            1e-4 * min(1, self.current_epoch / 100) * l2_size_loss
        ) / len(idxs)

        loss += reg_loss.cuda()
        loss += contrast_loss.cuda()
        loss += msssim_loss.cuda()

        
        self.log_dict(
            {
                "train/loss": loss,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        if self.random_sample_idx in idxs:
            i = ((idxs == self.random_sample_idx).nonzero(as_tuple=True)[0]).item()
            self.prediction_img.append(attenuation_values[i])
            self.gt_img.append(target[i])
            
        
        return loss

    def on_train_epoch_end(self):
        self.logger.log_image(key="train/image", images=[self.prediction_img[0], self.gt_img[0], (self.gt_img[0]-self.prediction_img[0])], caption=[f"pred_{self.random_sample_idx}", f"gt_{self.random_sample_idx}",f"residual_{self.random_sample_idx}"]) # log prediction images
        self.prediction_img.clear()  # free memory
        self.gt_img.clear()
        self.random_sample_idx = np.random.randint(self.num_volumes)

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
                {
                    "params": self.params.model.parameters(),
                    "lr": self.lr,
                },
                {
                    "params": self.params.latent_vectors.parameters(),
                    "lr": 1e-3,
                },
            ], amsgrad=True)
        lr_lambda = lambda epoch: 0.97 ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, T_mult=2)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, T_mult=2, eta_min=1e-6)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        


