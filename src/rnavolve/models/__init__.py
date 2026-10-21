"""Lazy model exports; checkpoint-key utilities also work in CPU-only CI."""

from importlib import import_module

_EXPORTS = {
    "RNAvolveConfig": ".configuration",
    "RNAvolve": ".mamba_backbone",
    "RNAvolveForMaskedLM": ".mamba_backbone",
    "RNAvolveForRegression": ".mamba_backbone",
    "RNAvolveForSequenceClassification": ".mamba_backbone",
    "RNAvolveTokenizer": ".tokenization",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
