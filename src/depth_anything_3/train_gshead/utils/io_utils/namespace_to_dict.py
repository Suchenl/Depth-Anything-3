def namespace_to_dict(ns):
    if not isinstance(ns, object) or isinstance(ns, (str, int, float, bool)):
        return ns
    # hasattr check is to handle non-namespace objects that might be in the structure
    if hasattr(ns, '__dict__'):
            return {key: namespace_to_dict(value) for key, value in vars(ns).items()}
    elif isinstance(ns, list):
            return [namespace_to_dict(item) for item in ns]
    return ns # return as is if not a namespace or list