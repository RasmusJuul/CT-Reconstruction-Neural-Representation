import torch
import torch.nn as nn
import math
import numpy as np

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


class SaturatingGaussianMixtureLoss(nn.Module):
    def __init__(self, target_means, variances, max_loss=1.0):
        super().__init__()
        self.register_buffer('means', torch.tensor(target_means, dtype=torch.float32))  # (num_targets, dim)
        self.register_buffer('variances', torch.tensor(variances, dtype=torch.float32))  # (num_targets,)
        self.max_loss = max_loss

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Ensure shape is (batch_size, dim)

        means = self.means.to(x.device)  # (num_targets, dim)
        variances = self.variances.to(x.device)  # (num_targets,)
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        means = means.unsqueeze(0)  # (1, num_targets, dim)
        variances = variances.unsqueeze(0)  # (1, num_targets)

        sq_dist = ((x - means) ** 2).sum(dim=2)  # (batch_size, num_targets)
        min_sq_dist = sq_dist.min(dim=1).values  # (batch_size,)
        loss = self.max_loss * (1 - torch.exp(-min_sq_dist / (2 * variances.min())))
        return loss.mean()










