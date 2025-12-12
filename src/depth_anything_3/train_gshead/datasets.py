from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from depth_anything_3.utils.io.input_processor import InputProcessor


class DL3DVDataSet(Dataset):
    """
    Minimal placeholder dataset for DL3DV-10K style samples.

    Each item in the loaded list should be a dict with keys:
    - image: list[np.ndarray | Image.Image | str]
    - extrinsics: np.ndarray or None
    - intrinsics: np.ndarray or None
    - depth: optional ground-truth depth tensor
    """

    def __init__(self, 
                 dataset_path: str = "DL3DV-10K/main",
                 subsets: List[str] = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K"]):
        self.dataset_path = Path(dataset_path)
        self.samples: List[Dict[str, Any]] = self._load_dataset(self.dataset_path)

    def _load_dataset(self, dataset_path: Path) -> List[Dict[str, Any]]:
        if not dataset_path.exists(): raise FileNotFoundError(dataset_path)

        for subset
        
        
    

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


class TrainDA3Static3DGSDataset(Dataset):
    """
    Light wrapper that reuses the InputProcessor so the model receives tensors
    in the same shape/order as the inference pipeline.
    """

    def __init__(self, 
                 dataset_name: str = "DL3DV-10K", 
                 dataset_path: str = "DL3DV-10K/main",
                 process_res: int = 504, 
                 process_res_method: str = "upper_bound_resize"):
        self.input_processor = InputProcessor()
        self.process_res = process_res
        self.process_res_method = process_res_method

        if dataset_name == "DL3DV-10K":
            self.dataset = DL3DVDataSet(dataset_path=dataset_path)
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.dataset[index]
        imgs, extrinsics, intrinsics = self._preprocess_inputs(
            data["image"], data.get("extrinsics"), data.get("intrinsics")
        )
        sample = {"image": imgs, "extrinsics": extrinsics, "intrinsics": intrinsics}
        if "depth" in data:
            sample["depth"] = torch.as_tensor(data["depth"])
        return sample

    def _preprocess_inputs(
        self,
        image: List[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            self.process_res,
            self.process_res_method,
        )
        return imgs_cpu, extrinsics, intrinsics


def split_val_dataset(dataset: Dataset,
                      split_num: int = None,
                      split_ratio: float = None,
                      generator: torch.Generator = None):
    """
    Split a validation dataset from the given dataset.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(42)
        
    if split_num is not None:
        split_num = min(split_num, len(dataset))
        val_set, train_set = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[split_num, len(dataset) - split_num],
            generator=Generator
            )
        
    elif split_ratio is not None:
        val_set, train_set = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))],
            generator=Generator
            )

    else:
        raise ValueError("Either split_num or split_ratio must be provided.")

    return train_set, val_set