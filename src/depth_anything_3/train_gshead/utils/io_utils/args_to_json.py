# io_utils.py
import json
import argparse
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import os


def _primitiveize(obj, _seen=None):
    """
    Recursively convert arbitrary Python objects into JSON-serializable primitives.
    - argparse.Namespace / SimpleNamespace -> dict
    - Path -> str
    - torch.device -> str
    - torch.dtype -> str
    - torch.Tensor -> dict {__tensor__:{dtype,shape,data}}
    - np.ndarray / np.generic -> list / python scalar
    - tuple/set -> {"__tuple__":[...]} / {"__set__":[...]}
    - fallback: str(obj)
    """    
    # ------------------ 原始错误代码：会引起某些bool, int, float类型变为str ------------------
    # if _seen is None:
    #     _seen = set()
    # oid = id(obj)
    # if oid in _seen:
    #     return str(obj)  # prevent cycles
    # _seen.add(oid)
    
    # if obj is None or isinstance(obj, (bool, int, float, str)):
    #     return obj
    
    # ------------------ Start of modification ------------------
    # (modification principle: put base type checking before circular reference checking)
    # Step 1: Start with the simplest and most common base type. 
    #         These types have no subobjects and do not need or deserve circular reference checking.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Step 2: For all other complex objects (containers, class instances, etc.), 
    #         circular reference checking now begins.
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return str(obj)  # Discover the loop, return a string to indicate an interruption
    _seen.add(oid)
    # ------------------ End of modifications ------------------
    
    if isinstance(obj, (argparse.Namespace, SimpleNamespace)):
        return {k: _primitiveize(v, _seen) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {str(k): _primitiveize(v, _seen) for k, v in obj.items()}

    if isinstance(obj, Path):
        return str(obj)

    # torch objects
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": {
                "dtype": str(obj.dtype).replace("torch.", ""),
                "shape": list(obj.shape),
                "data": obj.detach().cpu().tolist(),
            }
        }

    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, tuple):
        return {"__tuple__": [_primitiveize(v, _seen) for v in obj]}
    if isinstance(obj, set):
        return {"__set__": [_primitiveize(v, _seen) for v in obj]}
    if isinstance(obj, list):
        return [_primitiveize(v, _seen) for v in obj]

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)

    if hasattr(obj, "__dict__"):
        try:
            return {k: _primitiveize(v, _seen) for k, v in vars(obj).items()}
        except Exception:
            return str(obj)

    return str(obj)

def save_namespace_to_json(namespace: argparse.Namespace, save_path: str, indent: int = 4, ensure_ascii: bool = False):
    """Save argparse.Namespace (possibly nested) to JSON."""
    data = _primitiveize(namespace)
    save_path = str(save_path)
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    return save_path


def _restore_special(obj):
    """Restore special markers like __tuple__, __set__, __tensor__."""
    if isinstance(obj, dict):
        if "__tuple__" in obj:
            return tuple(_restore_special(x) for x in obj["__tuple__"])
        if "__set__" in obj:
            return set(_restore_special(x) for x in obj["__set__"])
        if "__tensor__" in obj:
            t = obj["__tensor__"]
            try:
                return torch.tensor(
                    t["data"],
                    dtype=getattr(torch, t["dtype"]),
                ).reshape(t["shape"])
            except Exception:
                return t["data"]
        return {k: _restore_special(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_special(x) for x in obj]
    return obj


def _dict_to_namespace(d):
    """Recursively convert dict -> argparse.Namespace (used in load)."""
    if not isinstance(d, dict):
        return d
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(_restore_special(v)))
    return ns


def load_json_to_namespace(path: str, restore_device: bool = True):
    """Load JSON saved by save_namespace_to_json and convert to argparse.Namespace."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ns = _dict_to_namespace(data)

    # optional: restore torch.device / dtype
    if restore_device:
        def _restore(obj):
            if isinstance(obj, argparse.Namespace):
                for k, v in vars(obj).items():
                    setattr(obj, k, _restore(v))
                return obj
            if isinstance(obj, dict):
                return {kk: _restore(vv) for kk, vv in obj.items()}
            if isinstance(obj, list):
                return [_restore(x) for x in obj]
            if isinstance(obj, str):
                if obj == "cpu" or obj.startswith("cuda"):
                    try:
                        return torch.device(obj)
                    except Exception:
                        return obj
                if obj.startswith("float") or obj.startswith("int") or obj.startswith("torch."):
                    try:
                        return getattr(torch, obj.replace("torch.", ""))
                    except Exception:
                        return obj
                return obj
            return obj
        ns = _restore(ns)

    return ns

if __name__ == "__main__":
    # Create a test namespace with mixed types
    args = argparse.Namespace(
        exp_name="test_experiment",
        seed=42,
        use_cuda=True,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        image_size=(256, 256),
        valid_ids={1, 2, 3},
        tensor=torch.randn(2, 3),
        np_array=np.array([1.5, 2.5, 3.5]),
        nested=argparse.Namespace(inner="hello", value=None)
    )

    save_path = Path("checkpoints/config.json")

    # Save
    save_namespace_to_json(args, save_path, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved args to {save_path}")

    # Load
    loaded_args = load_json_to_namespace(save_path)

    print("\n=== Loaded Namespace ===")
    print(loaded_args)

    # Verify types
    print("\n=== Type Checks ===")
    print("device:", type(loaded_args.device), loaded_args.device)
    print("dtype:", type(loaded_args.dtype), loaded_args.dtype)
    print("image_size:", type(loaded_args.image_size), loaded_args.image_size)
    print("valid_ids:", type(loaded_args.valid_ids), loaded_args.valid_ids)
    print("tensor:", type(loaded_args.tensor), loaded_args.tensor.shape, loaded_args.tensor.dtype)
    print("np_array:", type(loaded_args.np_array), loaded_args.np_array)
    print("nested:", type(loaded_args.nested), vars(loaded_args.nested))
