# =====================================================================================
#
# This script provides a robust argument parsing utility for Python projects that
# use a combination of JSON configuration files and command-line arguments.
#
# Core Functionality:
# 1.  Recursive JSON Loading: It starts with a main JSON config file specified
#     via the command line. It recursively scans this file and any nested JSON
#     files for string values that are paths to other `.json` files, loading
#     and merging them into a single configuration structure.
# 2.  Attribute-Style Access: The final configuration is converted into a
#     `types.SimpleNamespace` object. This allows for clean, attribute-style
#     access (e.g., `args.model.name`) instead of dictionary-style access
#     (`args['model']['name']`).
# 3.  Command-Line Overrides: Any argument specified on the command line will
#     override the corresponding value from the JSON files. This is crucial for
#     running experiments without modifying config files.
# 4.  Argument Inclusion: All arguments defined in the `ArgumentParser` will be
#     present in the final `args` object, even if they are not in the JSON files.
# 5.  Cycle Detection: It includes protection against circular references in JSON
#     files (e.g., `a.json` includes `b.json`, and `b.json` includes `a.json`),
#     preventing infinite recursion and program freezes.
# 6.  Relative Path Resolution: Correctly handles relative paths for nested
#     JSON files based on their parent file's location.
#
# =====================================================================================

import json
import argparse
from types import SimpleNamespace
import os

def recursive_json_to_namespace(data, base_path='', visited_paths=None):
    """
    Recursively converts dictionaries, loading JSON file paths into SimpleNamespace objects.
    Includes cycle detection to prevent infinite recursion.

    Args:
        data: The current data to process (dict, list, string, etc.).
        base_path (str): The directory of the current JSON file, used for resolving relative paths.
        visited_paths (set, optional): A set of absolute paths currently in the recursion stack.
                                       Used internally for cycle detection. Defaults to None.
    """
    # Initialize the visited set for the top-level call. This is crucial for cycle detection.
    if visited_paths is None:
        visited_paths = set()

    # If data is a dictionary, recursively convert its values.
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = recursive_json_to_namespace(value, base_path, visited_paths)
        return SimpleNamespace(**new_dict)

    # If data is a list, recursively convert its items.
    elif isinstance(data, list):
        return [recursive_json_to_namespace(item, base_path, visited_paths) for item in data]

    # If data is a string, check if it's a path to a JSON file.
    elif isinstance(data, str):
        potential_path = os.path.join(base_path, data)
        
        if data.endswith('.json') and os.path.exists(potential_path):
            # Get the absolute, normalized path to ensure consistent cycle checks.
            abs_path = os.path.abspath(potential_path)

            # --- CYCLE DETECTION LOGIC ---
            if abs_path in visited_paths:
                print(f"  - Warning: Circular reference detected for '{potential_path}'. "
                      f"Returning path as a string to break the loop.")
                return data  # Return the path string to break the cycle.
            # -----------------------------

            try:
                # Add the current path to the visited set BEFORE the recursive call.
                visited_paths.add(abs_path)

                with open(potential_path, 'r') as f:
                    nested_data = json.load(f)
                
                # The directory of this nested file becomes the new base_path for its contents.
                new_base_path = os.path.dirname(potential_path)
                
                print(f"  - Recursively loaded config from: {potential_path}")
                result = recursive_json_to_namespace(nested_data, new_base_path, visited_paths)

                # Remove the path AFTER the recursive call returns (backtrack).
                visited_paths.remove(abs_path)
                return result

            except Exception as e:
                # Ensure path is removed from the set even if an error occurs.
                if abs_path in visited_paths:
                    visited_paths.remove(abs_path)
                print(f"  - Warning: Could not load or parse JSON file '{potential_path}': {e}")
                return data # Return the original path string on failure.
        else:
            # If not a .json file path or file doesn't exist, return the string itself.
            return data
            
    # For any other data type (int, float, bool, None), return it directly.
    else:
        return data

def parse_args(parser):
    """
    Parses command-line arguments and recursively loads all nested JSON configuration files.
    
    The final configuration prioritizes arguments in the following order:
    1. Command-line arguments.
    2. Values from JSON files.
    3. Default values defined in the ArgumentParser.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser instance with defined arguments.

    Returns:
        types.SimpleNamespace: A nested namespace object containing the final configuration.
    """
    # 1. Parse the initial command-line arguments.
    entry_args = parser.parse_args()
    
    # 2. Load the main configuration file from the path specified by the --config argument.
    if not hasattr(entry_args, 'config'):
        base_path = ''
        config_from_json = {}
        pass
    else:
        print("Loading main config file...")
        main_config_path = entry_args.config
        if not main_config_path or not os.path.exists(main_config_path):
            # raise FileNotFoundError(f"Main config file not found at: {main_config_path}")
            print(f"Main config file not found at: {main_config_path}")
            config_from_json = {}
        else:
            print(f"Loading main config from: {main_config_path}")
            with open(main_config_path, 'r') as f:
                config_from_json = json.load(f)
    
        # The base path for resolving nested paths is the directory of the main config file.
        base_path = os.path.dirname(main_config_path)
    
    # 3. (CORRECTED MERGE LOGIC) Establish the final configuration dictionary.
    # Start with the argparse defaults as the base.
    final_config_dict = vars(entry_args)
    
    # Update the defaults with values from the JSON file.
    final_config_dict.update(config_from_json)
    
    # Finally, override with any non-default command-line arguments provided by the user.
    # We re-iterate through entry_args to ensure command-line values have the highest priority.
    cli_args_dict = vars(entry_args)
    parser_defaults = {action.dest: action.default for action in parser._actions}

    for key, value in cli_args_dict.items():
        # Override if the value is not the parser's default.
        # This correctly handles cases where the user provides a value,
        # including overriding a JSON value with a CLI value.
        if key in parser_defaults and value != parser_defaults[key]:
            final_config_dict[key] = value

    # 4. Now, recursively process the fully merged dictionary.
    # This will load any nested JSON paths and convert the entire structure.
    final_args = recursive_json_to_namespace(final_config_dict, base_path)
    
    return final_args

# Example of how to use this utility
def set_args():
    """
    Defines the argument parser and returns the final parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Example project using advanced arg parser.")
    
    # --- Core Settings ---
    parser.add_argument('--config', type=str, default='configs/train/config1.json',
                        help='Path to the main JSON configuration file.')
    
    # --- Logging and Saving ---
    parser.add_argument('--log_dir', type=str, default=None,
                        help="Directory to save logs. Overrides JSON setting.")
    parser.add_argument('--checkpoints_savedir', type=str, default="outputs/checkpoints",
                        help="Directory to save model checkpoints. Overrides JSON setting.")
    
    # --- Training Hyperparameters (can be overridden) ---
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training. Overrides JSON setting.')
    
    args = parse_args(parser)
    return args

if __name__ == '__main__':
    # This is a demonstration of how the parser would be called in a main script.
    # To run this example, you would need to create a dummy config file like:
    # `configs/train/config1.json`
    #
    # Example `config1.json`:
    # {
    #   "experiment_name": "json_experiment",
    #   "batch_size": 8,
    #   "model_config": "configs/models/model_a.json"
    # }
    #
    # Example `configs/models/model_a.json`:
    # {
    #   "model_name": "ResNet50",
    #   "depth": 50
    # }

    # Create dummy files for demonstration
    os.makedirs('configs/train', exist_ok=True)
    os.makedirs('configs/models', exist_ok=True)
    with open('configs/train/config1.json', 'w') as f:
        json.dump({
            "experiment_name": "json_experiment",
            "batch_size": 8,
            "model_config": "../models/model_a.json" # Using relative path
        }, f, indent=4)
    with open('configs/models/model_a.json', 'w') as f:
        json.dump({
            "model_name": "ResNet50",
            "depth": 50
        }, f, indent=4)

    print("--- Running Argument Parser ---")
    try:
        args = set_args()
        print("\n--- Final Parsed Config Object ---")
        print(f"Experiment Name: {args.experiment_name}")
        print(f"Log Directory (from argparse default): {args.log_dir}")
        print(f"Batch Size (from JSON): {args.batch_size}")
        print(f"Model Name (from nested JSON): {args.model_config.model_name}")
        print(f"Model Depth (from nested JSON): {args.model_config.depth}")
        
        # To see the full object structure:
        # import pprint
        # pprint.pprint(vars(args))

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the default config file 'configs/train/config1.json' exists.")
    
# v1: only support config file
# import argparse
# # from addict import Dict
# import json
def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

# from types import SimpleNamespace
# def dict_to_namespace(d):
#     """Recursively convert a dictionary into a SimpleNamespace object"""
#     if not isinstance(d, dict):
#         return d
#     # Recursively call this function for each value in the dictionary
#     # Then pass the transformed key-value pairs to SimpleNamespace
#     return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

# def parse_args(parser):
#     entry = parser.parse_args()
#     json_path = entry.config
#     args = json_to_args(json_path)
#     args_dict = args.__dict__
#     for index, (key, value) in enumerate(vars(entry).items()):
#         args_dict[key] = value
#     args = dict_to_namespace(args_dict)
#     return args


