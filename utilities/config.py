import yaml
from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    """
    Loads a YAML config file into a SimpleNamespace for dot-access.

    Args:
        path (str): Path to YAML file

    Returns:
        SimpleNamespace: Config object with attribute access
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)