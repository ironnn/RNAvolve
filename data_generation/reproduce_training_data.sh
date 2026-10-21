#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}

echo "[1/3] Generating K80 evolutionary mutants"
RNAVOLVE_GENERATION_CONFIG="$ROOT/data_generation/evolutionary_variants/configs/k80.yaml" "$PYTHON" -m rnavolve.synthetic.k80

echo "[2/3] Generating FM UTR-replacement mutants"
RNAVOLVE_GENERATION_CONFIG="$ROOT/data_generation/flow_matching/configs/flow_matching.yaml" "$PYTHON" -m rnavolve.synthetic.flow_matching

echo "[3/3] Merging K80 and FM mutants"
RNAVOLVE_GENERATION_CONFIG="$ROOT/data_generation/evolutionary_variants/configs/k80.yaml" "$PYTHON" -m rnavolve.synthetic.merge

echo "Done: assets/processed/k80/k80_mutations.pkl"
