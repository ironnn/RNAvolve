"""Inference and publication benchmark helpers."""

__all__ = ["RNAvolvePredictor"]


def __getattr__(name):
    if name == "RNAvolvePredictor":
        from .inference_api import RNAvolvePredictor

        return RNAvolvePredictor
    raise AttributeError(name)
