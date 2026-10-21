"""Lightweight structural check for the canonical deployed TE checkpoint."""

from __future__ import annotations

import torch
import pytest

from rnavolve.evaluation.human_te import CHECKPOINT_ROOT, _checkpoint_state


def test_canonical_te_checkpoint_deserializes() -> None:
    checkpoint = CHECKPOINT_ROOT / "te_student/final_tdc.ckpt"
    state = _checkpoint_state(checkpoint, torch.device("cpu"))
    assert isinstance(state, dict)
    assert len(state) == 357


@pytest.mark.parametrize("relative", [
    "te_student/w0.ckpt", "te_student/w1.ckpt", "te_student/w2.ckpt",
    "te_student/final_tdc.ckpt", "half_life/final_half_life.ckpt",
])
def test_native_config_without_custom_unpickler(relative) -> None:
    payload = torch.load(CHECKPOINT_ROOT / relative, map_location="cpu", weights_only=False)
    config = payload["hyper_parameters"]["config"]
    assert type(config).__module__ == "rnavolve.models.configuration"
    assert type(config).__name__ == "RNAvolveConfig"
    assert config.model_type == "rnavolve"
    assert config._target_ == "rnavolve.models.configuration.RNAvolveConfig"
    assert len(payload["state_dict"]) == 357


def test_native_backbone_keys() -> None:
    state = torch.load(CHECKPOINT_ROOT / "backbone/pretrained_step13500.pt", weights_only=False)
    assert len(state) == 334
    assert "rnavolve.backbone.embeddings.word_embeddings.embedding.weight" in state
    assert all(key.startswith(("rnavolve.", "lm_head.")) for key in state)
