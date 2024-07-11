import torch
import numpy as np
from pytorch_lightning import LightningModule
import torchmetrics as tm
import tifffile
from tqdm import tqdm

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

class TrigonometricEncoder(torch.nn.Module):
    
    def __init__(self,num_freq_bands: int):
        super(TrigonometricEncoder,self).__init__()
        
        self.num_freq_bands = num_freq_bands
           
    def forward(self,idxs: torch.Tensor
               ) -> torch.Tensor:
        idxs = idxs.T
        enc = [idxs]
        for i in range(self.num_freq_bands):
            enc.append(torch.sin(2**i *torch.pi*idxs))
            enc.append(torch.cos(2**i *torch.pi*idxs))
            
        enc = torch.cat(enc)
        
        return enc.T
        
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

def lr_lambda(epoch: int):
    if epoch <= 100:
        return 1.0
    else:
        return 0.97 ** (epoch-100)

class MLP(LightningModule):

    def __init__(self,args_dict,projection_shape):
        super(MLP,self).__init__()
        self.save_hyperparameters()

        self.projection_shape = projection_shape
        self.lr = args_dict['training']['learning_rate']
        self.imagefit_mode = args_dict["training"]["imagefit_mode"]

        self.l1_regularization_weight = args_dict['training']['regularization_weight']

        

        self.num_freq_bands = args_dict['model']['num_freq_bands']
        self.num_hidden_layers = args_dict['model']['num_hidden_layers']
        self.num_hidden_features = args_dict['model']['num_hidden_features']
        self.activation_function = args_dict['model']['activation_function']

        if self.num_freq_bands > 0:
            self.trig_encoder = torch.jit.script(TrigonometricEncoder(num_freq_bands=self.num_freq_bands))
            # self.trig_encoder = TrigonometricEncoder(num_freq_bands=self.num_freq_bands)
            self.trig_encoder = self.trig_encoder.to(device=args_dict['training']['device'])
            num_input_features = 3 * (self.num_freq_bands * 2 + 1) # +1 for the constant (original input)
        else:
            num_input_features = 3 # x,y,z coordinate


        layers = []
        for i in range(self.num_hidden_layers):
            layers.append(torch.nn.Sequential(torch.nn.Linear(self.num_hidden_features,self.num_hidden_features),
                                         get_activation_function(self.activation_function,args_dict),
                                         ))


        self.mlp = torch.nn.Sequential(torch.nn.Linear(num_input_features,self.num_hidden_features),
                                       get_activation_function(self.activation_function,args_dict),
                                       *layers,
                                        torch.nn.Linear(self.num_hidden_features,1),
                                        torch.nn.ReLU(),
                                        )

        if self.activation_function == 'sine':
            self.mlp.apply(sine_init)
            self.mlp[0].apply(first_layer_sine_init)

        self.loss_fn = torch.nn.MSELoss()
        self.l1_regularization = torch.nn.L1Loss()
        self.psnr = tm.image.PeakSignalNoiseRatio()
        self.validation_step_outputs = []
        self.validation_step_gt = []

    def forward(self, pts):
        pts_shape = pts.shape

        if len(pts.shape) > 2:
            pts = pts.view(-1,3)

        if self.num_freq_bands > 0:
            enc = self.trig_encoder(pts)
        else:
            enc = pts

        out = self.mlp(enc)

        if len(pts_shape) > 2:
            out = out.view(*pts_shape[:-1],-1)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        points, target = batch

        if self.imagefit_mode:
            attenuation_values = self.forward(points)
            attenuation_values = attenuation_values.view(target.shape)
            loss = self.loss_fn(attenuation_values,target)
            self.log_dict(
                {
                    "train/loss": loss,
                },
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            
            return loss
        else:
            lengths = torch.linalg.norm((points[:,-1,:] - points[:,0,:]),dim=1)
            attenuation_values = self.forward(points).view(points.shape[0],points.shape[1])
            detector_value_hat = compute_projection_values(points.shape[1],attenuation_values,lengths)

            smoothness_loss = self.l1_regularization(attenuation_values[:,1:],attenuation_values[:,:-1]) # punish model for big changes between adjacent points (to make it smooth)
            loss = self.loss_fn(detector_value_hat, target)

            if self.current_epoch > 10:
                clamp_loss = (torch.maximum(2*torch.abs(attenuation_values-.5)-1,torch.tensor(0,device=self.device))).mean() #punishing model hard for having outputs outside of 0-1
            else:
                clamp_loss = 0

            total_loss = loss + self.l1_regularization_weight*smoothness_loss + clamp_loss
        
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/loss_total":total_loss,
                    "train/l1_regularization":smoothness_loss,
                    "train/clamp_loss":clamp_loss
                },
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            
            return total_loss

    def validation_step(self, batch, batch_idx):
        points, target = batch
        if self.imagefit_mode:
            attenuation_values = self.forward(points)
            attenuation_values = attenuation_values.view(target.shape)
            
            loss = self.loss_fn(attenuation_values,target)
            
            self.validation_step_outputs.append(attenuation_values)
            self.validation_step_gt.append(target)
        else:
            lengths = torch.linalg.norm((points[:,-1,:] - points[:,0,:]),dim=1)
            attenuation_values = self.forward(points).view(points.shape[0],points.shape[1])
            detector_value_hat = compute_projection_values(points.shape[1],attenuation_values,lengths)

            smoothness_loss = self.l1_regularization(attenuation_values[:,1:],attenuation_values[:,:-1])
    
            loss = self.loss_fn(detector_value_hat, target)
            
            self.validation_step_outputs.append(detector_value_hat.detach().cpu())
            self.validation_step_gt.append(target.detach().cpu())

            total_loss = loss + self.l1_regularization_weight*smoothness_loss

        self.log_dict(
            {
                "val/loss": loss,
                "val/loss_total":total_loss,
                "val/l1_regularization":smoothness_loss,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        

    def on_validation_epoch_end(self):     
        all_preds = torch.cat(self.validation_step_outputs)
        all_gt = torch.cat(self.validation_step_gt)
        img = self.trainer.val_dataloaders.dataset.img.to(device=self.device)
        self.img_shape = img.shape
        if self.imagefit_mode:
            preds = all_preds.view(self.img_shape)
            gt = all_gt.view(self.img_shape)
            self.logger.log_image(key="val/reconstruction", images=[preds[self.img_shape[2]//2,:,:], gt[self.img_shape[2]//2,:,:]], caption=["pred", "gt"]) # log projection images
            psnr = self.psnr(preds.unsqueeze(dim=0).unsqueeze(dim=0),gt.unsqueeze(dim=0).unsqueeze(dim=0))
            self.log("val/reconstruction",psnr)
        else:
            valid_indices = self.trainer.val_dataloaders.dataset.valid_indices.view(self.projection_shape)
            preds = torch.zeros(self.projection_shape,dtype=all_preds.dtype)
            preds[valid_indices] = all_preds
            gt = torch.zeros(self.projection_shape,dtype=all_gt.dtype)
            gt[valid_indices] = all_gt

            for i in range(self.projection_shape[0]):
                self.logger.log_image(key="val/projection", images=[preds[i], gt[i], (gt[i]-preds[i])], caption=[f"pred_{i}", f"gt_{i}",f"residual_{i}"]) # log projection images

            img = self.trainer.val_dataloaders.dataset.img
            mgrid = torch.stack(torch.meshgrid(torch.linspace(-1-(1/img.shape[0]), 1-(1/img.shape[0]), img.shape[0]), torch.linspace(-1-(1/img.shape[0]), 1-(1/img.shape[0]), img.shape[0]), torch.linspace(-1-(1/img.shape[0]), 1-(1/img.shape[0]), img.shape[0]), indexing='ij'),dim=-1)
            mgrid = mgrid.view(-1,img.shape[2],3)
            outputs = torch.zeros((*mgrid.shape[:2],1))
            with torch.no_grad():
                for i in range(mgrid.shape[1]):
                    output = self.forward(mgrid[:,i,:].to(device=self.device))
                    
                    outputs[:,i,:] = output.cpu()
    
                outputs = outputs.view(self.img_shape)

            self.log("val/loss_reconstruction",self.loss_fn(outputs,img))
            self.logger.log_image(key="val/reconstruction", images=[outputs[self.img_shape[2]//2,:,:], img[self.img_shape[2]//2,:,:], (img[self.img_shape[2]//2,:,:]-outputs[self.img_shape[2]//2,:,:]),
                                                                    outputs[:,self.img_shape[2]//2,:], img[:,self.img_shape[2]//2,:], (img[:,self.img_shape[2]//2,:]-outputs[:,self.img_shape[2]//2,:]),
                                                                    outputs[:,:,self.img_shape[2]//2], img[:,:,self.img_shape[2]//2], (img[:,:,self.img_shape[2]//2]-outputs[:,:,self.img_shape[2]//2])],
                                  caption=["pred_xy", "gt_xy","residual_xy",
                                           "pred_yz", "gt_yz","residual_yz",
                                           "pred_xz", "gt_xz","residual_xz",])
            

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_gt.clear()  # free memory

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True)
        # return optimizer  
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        



