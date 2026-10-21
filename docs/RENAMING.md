# RNAvolve naming and native checkpoint migration

The public model name is **RNAvolve**. The Python distribution and package are
`rnavolve`; classes use the `RNAvolve` prefix and environment variables use
`RNAVOLVE_`. Install the renamed package with `python -m pip install -e .`.

The change covers public imports, entry points, configuration keys, figure labels,
benchmark model labels and prediction-column names, source-table filenames,
documentation, and CI. The local checkout directory is unchanged; the Git remote
now points to `ironnn/RNAvolve`. Independent historical experiment directories are
outside scope.

## Native checkpoints

All seven released artifacts have been checked for prior-brand fields. The six
neural checkpoints were reserialized with native `RNAvolveConfig` references,
`rnavolve.models.configuration` module paths, and `rnavolve` backbone keys. All
2,119 state-dict tensors match the originals exactly in shape, dtype, and value.
The fitted teacher already had no prior-brand fields and is byte-identical.

The loader in `src/rnavolve/models/checkpoint.py` uses ordinary `torch.load` with
no pickle aliases or key remapping. Backbone extraction and SFT initialization use
strict state-dict loading. Architecture, reward settings, prediction logic, and
numeric source data are unchanged. The six neural files and their `.zst` archives
have new hashes, recorded in the refreshed checksum files and
`assets/checkpoints/NATIVE_MIGRATION.json`.

Original binaries and literal source-location records are backed up outside this
checkout. Public provenance uses run identifiers and logical private-source names
instead of private checkout names. The smoke-data builder takes an explicit
`--pretrain-root`; it has no historical checkout-name default. Git history and
independent experiment directories are not rewritten by this work.
Private archived checkpoints must be migrated separately before optional reruns.

Import the native package as follows:

```python
from rnavolve.evaluation import RNAvolvePredictor

predictor = RNAvolvePredictor("final_tdc", device="cuda:0")
```

## Validation performed

- The initial text/binary scan covered 634 public files and all seven decompressed
  checkpoint archives, including serialized configuration metadata and state-dict
  keys. This scan does not detect labels rendered as PDF outlines or image pixels;
  the figure-export follow-up below closes the observed Fig. 2 omission.
- Compared three real human smoke transcripts before/after renaming for `w0`,
  `w1`, `w2`, `final_tdc`, and `hl_new`: exactly identical predictions.
- Compared a real-transcript backbone embedding and three teacher-feature rows:
  identical embedding SHA-256 and teacher predictions.
- Verified all seven compressed and decompressed checkpoint hashes, all 103
  manifest assets, and all 53 tracked CSVs. CSV changes are name-only replacements;
  numeric values and sequence data are unchanged.
- Ran CPU core checks, GPU inference/routing smoke, gene-disjoint split checks,
  Fig. 2 checks/redraw, Fig. 3 value checks/redraw, Fig. 4 strict audit, and the
  Fig. 5 CPU RL rollout/update/save-load/export smoke.
- Ran two warm-started backbone pretraining steps and one three-row SFT epoch,
  including validation and checkpoint saving; these are plumbing checks, not
  scientific training results.

The equivalence runner is `tests/smoke/test_rename_equivalence.py`; pass a saved
`--reference` JSON to compare outputs exactly. It requires decompressed weights
and an NVIDIA GPU. `tests/test_renaming.py` guards native package names and loading in
the CPU CI job.

## Plotting and optional workflows

The 2026-10-22 visual follow-up found that the archived Fig. 2 PDF retained the
prior backbone label even though its filename and plotting source were renamed.
It has now been regenerated from the checked frozen source tables, with all three
Panel B labels reading `RNAvolve backbone`. The standalone Fig. 3B title now reads
`RNAvolve`; the `0412` run identifier remains in data keys for traceability.
The generated Fig. 2 and Fig. 3B PDFs retain searchable text. Regression tests
require their current display labels and reject prior-brand labels; CI installs
Poppler to run these checks. Rendered exports are also inspected visually.

The Fig. 3B standalone redraw now accepts the shipped merged prediction table and
uses the frozen full-precision annotations from its configuration. Compact plotting
predictions are not used to silently replace the manuscript's recorded coefficients.
The raw-input recalculation branch still requires the original per-model inputs.

The RNAvolve adapter in the external-backbone harness and its common-split training
entry point now use the public package rather than unpublished development modules.
Full historical benchmark/training reruns still require the non-distributed input
tables documented in `BENCHMARKS.md`; renaming does not make those assets available.
