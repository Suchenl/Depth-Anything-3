import torch
from PIL import Image
import numpy as np

def mask_to_pil(mask: torch.Tensor, mode: str = 'L', thre=None) -> Image.Image:
    if thre is not None:
        mask = (mask > thre).float()
    image_np = np.clip(mask.cpu().numpy(), 0, 1) * 255
    image_np = image_np.astype(np.uint8)
    pil_image = Image.fromarray(image_np, mode=mode)
    return pil_image
