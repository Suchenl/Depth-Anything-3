import numpy as np
import torch
import cv2
import h5py
from typing import *
from PIL import Image
import re

def check_cycle_consistency(flow_01, flow_10):
    H, W = flow_01.shape[:2]
    new_coords = flow_01 + np.stack(
        np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1
    )
    flow_reprojected = cv2.remap(
        flow_10, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )
    cycle = flow_reprojected + flow_01
    cycle = np.linalg.norm(cycle, axis=-1)
    occlusion_mask = (cycle > 0.1 * min(H, W)).astype(np.float32)  # 1 denote occlusion
    return occlusion_mask

def process_read_flow(flow_read, out_type: Literal["numpy", "tensor"] = "tensor", nan_to: Literal["zero"] = "zero", calc_valid: bool = False):
    """ Process a flow field read from a file."""
    # Process NaN value
    if nan_to == "zero":
        flow = np.nan_to_num(flow_read, nan=0)
    # Processing for output specific format
    if out_type == "numpy":  # output shape: (H, W, 2)
        pass 
    elif out_type == "tensor":  # output shape: (2, H, W)
        if flow.shape[2] == 2: # [H, W, 2]
            flow = np.transpose(flow, [2, 0, 1])
        flow = torch.from_numpy(flow)
    else:
        raise ValueError(f"Invalid output type: {out_type}. Please choose from 'numpy' or 'tensor'.")
    if calc_valid:
        valid = ~np.isnan(flow_read).any(axis=-1)[None, :, :]
        if out_type == "numpy":
            pass 
        elif out_type == "tensor":
            valid = torch.from_numpy(valid).float()
        return flow, valid
    else: 
        return flow

def read_flo5(read_path, out_type: Literal["numpy", "tensor"] = "tensor", nan_to: Literal["zero"] = "zero", calc_valid: bool = False):
    """
    Read a flo5 file and return the flow field as a numpy array or a torch tensor.
    Args:
        read_path (str): Path to the flo5 file.
        out_type (str): Output type. Can be 'numpy' or 'tensor'.
        nan_to (str): How to process NaN values. Can be 'zero'.
        calc_valid (bool): Whether to calculate the valid mask.
    Returns:
        flow (np.ndarray or torch.Tensor): The flow field.
        valid (np.ndarray or torch.Tensor): The valid mask. Only returned if calc_valid is True.
    """
    with h5py.File(read_path, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {read_path} does not have a 'flow' key. Is this a valid flo5 file?")
        flow_read = f["flow"][()] # type: np.ndarray, shape: (H, W, 2) or maybe (2, H, W)   
    return process_read_flow(flow_read, out_type, nan_to, calc_valid)

def write_flo5(flow, save_path):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)

def read_flo(read_path, out_type: Literal["numpy", "tensor"] = "tensor", nan_to: Literal["zero"] = "zero", calc_valid: bool = False):
    """
    Read a flo file and return the flow field as a numpy array or a torch tensor.
    Args:
        read_path (str): Path to the flo file.
        out_type (str): Output type. Can be 'numpy' or 'tensor'.
        nan_to (str): How to process NaN values. Can be 'zero'.
        calc_valid (bool): Whether to calculate the valid mask.
    Returns:
        flow (np.ndarray or torch.Tensor): The flow field.
        valid (np.ndarray or torch.Tensor): The valid mask. Only returned if calc_valid is True.
    """
    with open(read_path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            flow_read = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape flow_read into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            flow_read = np.resize(flow_read, (int(h), int(w), 2))
    return process_read_flow(flow_read, out_type, nan_to, calc_valid)
                
def read_flow(read_path, out_type: Literal["numpy", "tensor"] = "tensor", nan_to: Literal["zero"] = "zero", calc_valid: bool = False):
    """
    Read a flow file and return the flow field as a numpy array or a torch tensor.
    Args:
        read_path (str): Path to the flow file.
        out_type (str): Output type. Can be 'numpy' or 'tensor'.
        nan_to (str): How to process NaN values. Can be 'zero'.
        calc_valid (bool): Whether to calculate the valid mask.
    Returns:
        flow (np.ndarray or torch.Tensor): The flow field.
        valid (np.ndarray or torch.Tensor): The valid mask. Only returned if calc_valid is True.    
    """
    if read_path.endswith(".flo"):
        return read_flo(read_path, out_type, nan_to, calc_valid)
    elif read_path.endswith(".flo5"):
        return read_flo5(read_path, out_type, nan_to, calc_valid)
    else:
        raise ValueError(f"Invalid file type: {read_path}. Please choose from '.flo' or '.flo5'.")
