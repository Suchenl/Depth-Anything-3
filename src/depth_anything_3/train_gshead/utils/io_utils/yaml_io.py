import yaml
from collections import OrderedDict
from types import SimpleNamespace

def load_yaml_config(path, keep_order=False, as_namespace=False):
    """
    Reads a YAML configuration file, optionally returning in dict / OrderedDict / Namespace format.    
    Args:    
        path (str): Path to the YAML file.
        keep_order (bool): If True, preserves order (uses OrderedDict).
        as_namespace (bool): If True, returns a SimpleNamespace (dot-accessible).

    Returns:
        dict | OrderedDict | SimpleNamespace
    """
    # Use a custom Loader if order preservation is required
    if keep_order:
        _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

        def dict_representer(dumper, data):
            return dumper.represent_dict(data.items())

        def dict_constructor(loader, node):
            return OrderedDict(loader.construct_pairs(node))

        class OrderedLoader(yaml.SafeLoader): pass
        class OrderedDumper(yaml.SafeDumper): pass
        OrderedDumper.add_representer(OrderedDict, dict_representer)
        OrderedLoader.add_constructor(_mapping_tag, dict_constructor)

        Loader = OrderedLoader
    else:
        Loader = yaml.SafeLoader

    with open(path, 'r') as f:
        data = yaml.load(f, Loader=Loader)

    if as_namespace:
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(v) for v in d]
            else:
                return d
        data = dict_to_namespace(data)

    return data
