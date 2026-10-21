"""Load trusted, native RNAvolve checkpoints without module or key remapping.

Only load checkpoints from trusted sources; PyTorch pickle loading executes code.
"""

import torch


def load_checkpoint(path, map_location="cpu"):
    """Load a native Lightning payload or bare state dict."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Expected a checkpoint dictionary")
    if "state_dict" in payload and not isinstance(payload["state_dict"], dict):
        raise TypeError("Expected a state_dict dictionary")
    return payload
