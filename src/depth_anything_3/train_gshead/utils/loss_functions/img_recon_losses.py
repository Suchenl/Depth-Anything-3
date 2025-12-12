import lpips
from typing import Literal
import torch

class LPIPSLoss:
    def __init__(self, net: Literal['alex', 'vgg']='vgg', device='cpu'):
        self.net = lpips.LPIPS(net=net).to(device)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

    def __call__(self, pred, target):
        # range: [-1, 1]
        return self.net(pred, target).mean()

class PixelSimLoss:
    """
    A class to calculate channel-weighted L1 or L2 (MSE) pixel similarity loss.
    The weights are applied to the error (difference) of the channels, 
    not directly to the prediction and target values.
    """
    def __init__(self,
                 norm: Literal['L1', 'L2'] = 'L2',
                 channel_weight: torch.Tensor = torch.tensor([0.2126, 0.7152, 0.0722]),
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 device='cpu'):
        """
        Args: 
            channel_weight (torch.Tensor): A tensor of shape (C,) containing the weights for each channel.
                The weights are applied to the error (difference) of the channels, not directly to the prediction and target values.
                Sort of weights: [R, G, B] -> [0.2126, 0.7152, 0.0722] for ITU-R BT.709
        """
        if norm not in ['L1', 'L2']:
            raise ValueError(f"Invalid norm: {norm}. Must be 'L1' or 'L2'.")
        self.norm = norm
        self.reduction = reduction
        # Ensure channel_weight is on the correct device and has the shape (1, C, 1, 1) 
        # for broadcasting during multiplication with loss_per_pixel (B, C, H, W).
        self.channel_weight = channel_weight.to(device).view(1, -1, 1, 1)
        # Dim check
        if self.channel_weight.size(1) != 3:
             print(f"Warning: channel_weight size is {self.channel_weight.size(1)}. Ensure it matches your image channel dimension (C).")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted loss.
        """
        # 1. Calculate the difference (error)
        diff = pred - target
        # 2. Calculate the loss per pixel and per channel based on the specified norm
        if self.norm == 'L1':
            loss_per_pixel = torch.abs(diff)
        elif self.norm == 'L2':
            loss_per_pixel = diff ** 2
        else:
            raise ValueError(f"Invalid norm: {self.norm}. Must be 'L1' or 'L2'.")
        # 3. Apply the channel weights to the loss
        # Output shape: (B, C, H, W)
        weighted_loss = loss_per_pixel * self.channel_weight 
        # 4. Apply the specified reduction
        if self.reduction == 'mean':
            # Mean over all elements
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            # Sum over all elements
            return torch.sum(weighted_loss)
        elif self.reduction == 'none':
            # Return the unaggregated loss tensor (B, C, H, W)
            return weighted_loss
        else:
            # Should not happen due to __init__ check
            raise ValueError(f"Invalid reduction value encountered: {self.reduction}")