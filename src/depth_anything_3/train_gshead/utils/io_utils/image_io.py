import torch
from PIL import Image
import numpy as np
from typing import Union

def tensor_to_pil(tensor: torch.Tensor, mode: str = 'RGB') -> Image.Image:
    tensor =  tensor.detach().permute(1, 2, 0)
    if tensor.min() < -0.1: # if tensor.min() < -0.1, treat it as having range [-1, 1]
        tensor = (tensor + 1) / 2
    image_np = np.clip(tensor.cpu().numpy(), 0, 1) * 255
    image_np = image_np.astype(np.uint8).squeeze()
    pil_image = Image.fromarray(image_np, mode=mode)
    
    return pil_image
    # """
    # Convert PyTorch Tensor to PIL Image
    
    # Args:
    #     tensor: Input tensor, supports multiple formats:
    #             - [C, H, W] shape
    #             - [H, W, C] shape  
    #             - [B, C, H, W] shape (takes first image from batch)
    #             - [H, W] shape (grayscale)
    #     mode: Output PIL image mode, e.g., 'RGB', 'L' (grayscale), 'RGBA', etc.
    
    # Returns:
    #     PIL.Image object
    # """
    # # Ensure tensor is on CPU
    # tensor = tensor.cpu().detach()
    
    # # Handle batch data: take first image from batch
    # if tensor.dim() == 4:
    #     tensor = tensor[0]  # Take first image from batch [C, H, W]
    
    # # Handle different tensor formats
    # if tensor.dim() == 3:
    #     # [C, H, W] or [H, W, C]
    #     if tensor.size(0) in [1, 3, 4]:  # Channels in first dimension
    #         tensor = tensor.permute(1, 2, 0)  # Convert to [H, W, C]
    #     # Now should be in [H, W, C] format
    # elif tensor.dim() == 2:
    #     # Grayscale [H, W] → [H, W, 1]
    #     tensor = tensor.unsqueeze(-1)
    # else:
    #     raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Supported: 2D, 3D or 4D (batch) tensors")
    
    # # Convert to numpy array
    # numpy_array = tensor.numpy()
    
    # # Handle data type and value range
    # if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
    #     # Floating point tensor: assume range [0, 1] or [-1, 1]
    #     if numpy_array.min() < 0:
    #         # Range [-1, 1], convert to [0, 1]
    #         numpy_array = (numpy_array + 1) / 2
    #     # Ensure within [0, 1] range
    #     numpy_array = np.clip(numpy_array, 0, 1)
    #     # Convert to 8-bit integer [0, 255]
    #     numpy_array = (numpy_array * 255).astype(np.uint8)
    # elif numpy_array.dtype == np.uint8:
    #     # Already 8-bit integer, use directly
    #     pass
    # else:
    #     # Convert other data types to uint8
    #     numpy_array = numpy_array.astype(np.uint8)
    
    # # Determine mode based on channel count
    # channels = numpy_array.shape[2]
    # if mode is None:
    #     if channels == 1:
    #         mode = 'L'
    #     elif channels == 3:
    #         mode = 'RGB'
    #     elif channels == 4:
    #         mode = 'RGBA'
    #     else:
    #         raise ValueError(f"Unsupported number of channels: {channels}")
    
    # # Create PIL image
    # if mode == 'L':
    #     pil_image = Image.fromarray(numpy_array.squeeze(), mode=mode)
    # else:
    #     pil_image = Image.fromarray(numpy_array, mode=mode)
    
    # return pil_image