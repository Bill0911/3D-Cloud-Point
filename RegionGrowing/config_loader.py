import os
import yaml

class ConfigError(Exception):
    pass

def _read_yaml(path):
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse YAML '{path}': {e}")


def load_config(default_path="configs/default.yaml", override_path=None):
    """
    Loads the main configuration.
    If override_path is provided, it merges in values on top of default.yaml.
    """

    cfg = _read_yaml(default_path)
    if cfg is None:
        raise ConfigError("default.yaml is empty or invalid.")

    # Optional additional config file
    if override_path is not None:
        if os.path.exists(override_path):
            override = _read_yaml(override_path)
            cfg = _merge(cfg, override)
        else:
            raise ConfigError(f"Override config not found: {override_path}")

    _validate(cfg)
    return cfg


def _merge(base, override):
    """
    Recursively merges two dictionaries.
    override takes precedence over base.
    """
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate(cfg):
    """
    Ensures required sections exist.
    Expand as needed.
    """
    required_sections = ["classification", "segmentation", "measurement"]

    for section in required_sections:
        if section not in cfg:
            raise ConfigError(f"Missing required config section: '{section}'")

    # Basic key validation
    cls = cfg["classification"]
    if "voxel_size" not in cls or "angle_threshold" not in cls:
        raise ConfigError("classification config missing keys")

    seg = cfg["segmentation"]
    if "grid_size" not in seg or "min_room_points" not in seg:
        raise ConfigError("segmentation config missing keys")

    mea = cfg["measurement"]
    if "grid_size" not in mea:
        raise ConfigError("measurement config missing keys")
