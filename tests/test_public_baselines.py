"""Keep the public baseline roster consistent across entry points and figures."""

import csv
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BACKBONES = {"codonbert", "dnabert2", "lucaone", "mrnabert", "orthrus", "rnafm"}
DISPLAY_NAMES = {"CodonBERT", "DNABERT-2", "LucaOne", "mRNABERT", "Orthrus", "RNA-FM"}


def models_in_csv(relative_path):
    with (ROOT / relative_path).open() as handle:
        rows = list(csv.DictReader(handle))
    models = [row["model"] for row in rows]
    assert len(models) == len(set(models))
    return set(models)


def test_public_model_registries():
    from external_models.common.backbones import ALL_KEYS, FULL_FT_KEYS

    assert set(ALL_KEYS) == BACKBONES | {"rnavolve"}
    assert set(FULL_FT_KEYS) == BACKBONES
    path = ROOT / "external_models/inference/evaluate_te_checkpoint.py"
    spec = importlib.util.spec_from_file_location("public_te_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.ALL_MODELS) == BACKBONES | {"ribonn", "utrlm"}
    config = yaml.safe_load((ROOT / "benchmarks/configs/external_llm_te.yaml").read_text())
    assert set(config["models"]) == BACKBONES | {"ribonn", "utrlm", "rnavolve_human_mouse"}


@pytest.mark.parametrize("task, excluded", [
    ("mean_ribosome_load", {"codonbert"}),
    ("mrna_half_life", set()),
    ("expression_level", set()),
])
def test_fig2_public_roster(task, excluded):
    path = f"assets/manuscript_figures/fig2_compact/F2B_{task}.csv"
    assert models_in_csv(path) == (BACKBONES | {"rnavolve"}) - excluded


def test_fig3_public_roster():
    path = "assets/manuscript_figures/full/fig3_current_nm/F3F_heldout_human_TE_prediction_R2.csv"
    assert models_in_csv(path) == DISPLAY_NAMES | {"RiboNN", "UTR-LM", "RNAvolve"}


@pytest.mark.parametrize("name", [
    "panel_a_crispr_on_target", "panel_b_crispr_off_target",
    "panel_c_rna_switches", "panel_d_rna_modification",
])
def test_extended_data_public_roster(name):
    assert models_in_csv(f"figures/extended_data/ext_fig2/data/{name}.csv") == BACKBONES | {"rnavolve"}
