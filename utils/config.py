import torch
import yaml

_ACTIVATION_FNS = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "ELU": torch.nn.ELU,
}


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_policy_kwargs(policy_kwargs: dict) -> dict:
    """Convert string keys in policy_kwargs to their SB3-expected Python types."""
    kwargs = dict(policy_kwargs)
    if "activation_fn" in kwargs:
        name = kwargs["activation_fn"]
        if name not in _ACTIVATION_FNS:
            raise ValueError(f"Unknown activation_fn '{name}'. Valid options: {list(_ACTIVATION_FNS)}")
        kwargs["activation_fn"] = _ACTIVATION_FNS[name]
    if "net_arch" in kwargs:
        kwargs["net_arch"] = list(kwargs["net_arch"])
    return kwargs
