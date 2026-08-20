# Validation archive scope

This directory contains the compact public evidence used by the revised
manuscript and response to reviewers.

## Public contents

- JSON summaries record every numerical value quoted in the revision.
- Selected NPZ arrays reproduce the NaCl and three-material polar figures.
- Compact constrained IFC tensors support the FC2/FC3/FC4 validation tables.
- QE/DFPT input files and parsed dynamical-matrix summaries document settings.

The top-level `reviewer_evidence_summary.json`,
`matched_sampling_summary.json`, and `production_sotc_nac_summary.json` provide
the shortest audit path for the reviewer-requested controls.

## Excluded restart data

Large QE scratch trees, wave-function restart files, scheduler logs, and local
checkpoint datasets are intentionally ignored by Git. They are operational
restart artifacts rather than compact submission data. The public drivers and
input files specify how to regenerate them; the manuscript does not claim that
these large restart directories are distributed in the repository.

Excluded data include:

- `scratch/`, `thermal_qe/`, and material-specific QE checkpoint directories;
- scheduler `.out` and `.err` files;
- redundant dense intermediate tensors that are not used in reported values.

## Regeneration

```bash
python3 codes/validation/summarize_real_anharmonic_ifcs.py
python3 codes/validation/plot_ifc34_diagnostics.py
python3 codes/validation/plot_nacl_loto.py
python3 codes/validation/plot_matched_sampling_ablation.py
python3 codes/validation/plot_sotc_polar_nac.py
```
