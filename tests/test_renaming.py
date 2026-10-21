"""Guard native RNAvolve package names and checkpoint round trips."""

from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnavolve.models.checkpoint import load_checkpoint


def test_native_checkpoint_round_trip(tmp_path):
    state = OrderedDict({"rnavolve.backbone.weight": torch.arange(6).reshape(2, 3)})
    state._metadata = {"rnavolve": {"version": 1}}
    path = tmp_path / "weights.pt"
    torch.save(state, path)
    result = load_checkpoint(path)
    assert torch.equal(result["rnavolve.backbone.weight"], state["rnavolve.backbone.weight"])
    assert result._metadata == {"rnavolve": {"version": 1}}
    torch.save({"state_dict": state, "epoch": 4}, path)
    payload = load_checkpoint(path)
    assert payload["epoch"] == 4
    assert list(payload["state_dict"]) == ["rnavolve.backbone.weight"]


def test_checkpoint_requires_mapping(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError, match="dictionary"):
        load_checkpoint(path)


def test_current_package_and_branding():
    import rnavolve
    assert rnavolve.__version__
    packages = {path.parent.name for path in (ROOT / "src").glob("*/__init__.py")}
    assert packages == {"rnavolve"}
    assert 'name = "rnavolve"' in (ROOT / "pyproject.toml").read_text()
    assert (ROOT / "README.md").read_text().startswith("# RNAvolve\n")
