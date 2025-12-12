import torch
from depth_anything_3.api import DepthAnything3
import os
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./ckpts")
    parser.add_argument("--save_type", type=str, default="hf", choices=["huggingface / hf", "pytorch / torch"])
    args = parser.parse_args()
    return args

def main(args):
    # set up args
    save_dir = args.save_dir
    save_type = args.save_type
    
    # Create save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    # model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
    del model.model.gs_head
    del model.model.gs_adapter
    # print("Model:\n", model)

    if save_type == "huggingface" or save_type == "hf":
        save_path = os.path.join(save_dir, "da3-giant-no3dgs")
        model.save_pretrained(save_path, safe_serialization=True)
        print("Weights saved to:", save_path)

    elif save_type == "pytorch" or save_type == "torch":
        save_path = os.path.join(save_dir, "da3-giant-no3dgs.bin")
        torch.save(model.state_dict(), save_path)
        print("Weights saved to:", save_path)

    else:
        raise ValueError("Invalid save_type.")

if __name__ == "__main__":
    args = set_args()
    main(args)
    
    

