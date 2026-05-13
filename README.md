# SQTC — Special Quasirandom Thermal Cells

A first-principles framework for computing high-temperature phonon dispersions, density of states, and thermodynamic properties from **purposefully designed small supercells**, without large stochastic ensembles or molecular dynamics.

---

## Overview

Standard high-temperature phonon methods (TDEP, SSCHA, AIMD) require large supercells — typically 54–500+ atoms — making DFT cost prohibitive for many systems. SQTC extends the **Special Quasirandom Structures (SQS)** philosophy to the thermal problem:

> Just as an SQS encodes the statistical signature of a random alloy in the smallest possible cell, an SQTC encodes the **displacement–displacement correlations** of a thermally disordered crystal in a small, self-consistently optimised cell.

A single SQTC with 32–64 atoms replaces hundreds of stochastic snapshots while reproducing phonon properties to within a few percent of experiment.

---

## Method Summary

1. **Correlator design** — target displacement correlations $C_{ij}^{\alpha\beta}(\mathbf{R}, T)$ are computed from a trial phonon model using the quantum coth formula.
2. **Cell construction** — atom displacements are optimised (via gradient descent on a correlator mismatch cost) so that a single periodic cell best matches the target correlator hierarchy up to cutoff $r_\text{cut}$.
3. **DFT forces** — VASP computes forces on the displaced cell.
4. **IFC extraction** — interatomic force constants are fitted by ridge regression:
$$\boldsymbol{\Phi} = (\mathbf{G}^T\mathbf{G} + \alpha\mathbf{I})^{-1}\mathbf{G}^T\mathbf{F}$$
5. **Self-consistency** — the new phonon model updates the correlator targets; steps 1–4 repeat until convergence (typically 2–8 VASP calls total).
6. **Thermodynamics** — phonon DOS → heat capacity $C_V$, ZPE, vibrational entropy $S_\text{vib}$, free energy $F_\text{vib}$.

Full derivations are in [Theory/SQTC_framework.md](Theory/SQTC_framework.md) and [Theory/SQTC_derivations.md](Theory/SQTC_derivations.md).

---

## Repository Structure

```
codes/
  sqtc/              ← Python package
    runner.py        ← Self-consistent SQTC loop
    postprocessor.py ← CLI post-processing (phonons + thermodynamics)
    cell_design.py   ← Correlator-optimised cell construction
    correlators.py   ← Quantum coth correlator targets
    ifc_extractor.py ← Ridge-regression IFC fitting
    phonons.py       ← Phonon dispersion, DOS, thermodynamics
    qha.py           ← Quasi-harmonic approximation
    vasp_io.py       ← POSCAR / OUTCAR I/O
    mock_forces.py   ← Analytic test forces (He, H₂)
    constants.py     ← Physical constants
  examples/          ← Driver scripts for all benchmarked systems
  tests/             ← pytest suite

Theory/
  SQTC_framework.md  ← Theoretical framework (motivation, formulation, benchmarks)
  SQTC_derivations.md ← Detailed derivations (IFC regression, symmetry, QHA, ...)

slurm_*.sh           ← SLURM job scripts (MPCDF Viper cluster)
RUNNING.md           ← Step-by-step usage guide
requirements.txt     ← Python dependencies
```

---

## Benchmarked Systems

| Material | Structure | $T_\text{design}$ | $\omega_\text{max}$ (THz) | $T_D$ (K) | $C_V$ (J/mol·K) | Expt. $T_D$ (K) |
|---|---|---|---|---|---|---|
| Au | FCC | 1200 K | 4.74 | 156 | 24.6 | 165 |
| Cu | FCC | 1200 K | 8.16 | 343 | 24.4 | 315 |
| Al | FCC | 300 K | 10.47 | 429 | 24.2 | 390–428 |
| Mo | BCC | 1500 K | 9.38 | 432 | 23.9 | 380–470 |
| Ti | BCC | 1200 K | 5.71 | 359 | 23.8 | 380 |
| NaCl | Rocksalt | 700 K | 6.53 | 305 | 48.2 | 321 |
| MgO | Rocksalt | 1500 K | 21.3 | 940 | 47.6 | 940 |
| PbTe | Rocksalt | 700 K | 3.34 | 135 | 48.5 | 130–160 |

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |
| spglib | ≥ 2.1 |
| ase | ≥ 3.22 |
| pytest | ≥ 7.0 |

VASP (≥ 5.4) is required for DFT force calculations. The mock-force drivers (`run_sqtc_he.py`, `run_sqtc_h2.py`) work without VASP.

### Running without VASP (H₂ or He test)

```bash
python codes/examples/run_sqtc_he.py
python codes/examples/run_sqtc_h2.py
```

### Submitting a VASP calculation (SLURM)

```bash
sbatch slurm_au.sh        # fcc Au at 1200 K
sbatch slurm_mgo.sh       # rocksalt MgO at 1500 K
```

Results are written to `sqtc_<material>_vasp_run/`.

### Post-processing

```bash
# Auto-detect structure from spglib:
python codes/sqtc/postprocessor.py --run-dir sqtc_au_vasp_run

# Specify structure explicitly:
python codes/sqtc/postprocessor.py --run-dir sqtc_nacl_vasp_run --structure rocksalt --r-cutoff 4.3

# Enable central-force symmetry projection:
python codes/sqtc/postprocessor.py --run-dir sqtc_cu_vasp_run --symmetrize-bonds
```

Output is saved to `<run_dir>/postproc/`: phonon band structure, DOS plot, and `phonon_results.json`.

See [RUNNING.md](RUNNING.md) for the full usage guide.

### Running tests

```bash
pytest codes/tests/ -v
```

---

## Key Design Choices

**Single cell per iteration** — only one DFT calculation per SQTC step (vs. dozens in TDEP/SSCHA), making convergence cheap.

**Ridge regression** — Tikhonov regularisation suppresses noise amplification in under-determined IFC fits. Default $\alpha = 10^{-3}$.

**Central-force symmetry projection** (`symmetrize_bonds`) — for high-symmetry structures (FCC, rocksalt), IFC tensors are projected onto the $A\hat{R}_\alpha\hat{R}_\beta + B(\delta_{\alpha\beta} - \hat{R}_\alpha\hat{R}_\beta)$ subspace, reducing free parameters and improving stability. Auto-detected via spglib.

**Quantum correlators** — displacement targets use the exact quantum coth formula, capturing zero-point motion at all temperatures.

---

## Theory

The full theoretical background is in the [Theory/](Theory/) directory:

- **[SQTC_framework.md](Theory/SQTC_framework.md)** — motivation, formulation, correlator hierarchy, convergence analysis, benchmark results, implementation status
- **[SQTC_derivations.md](Theory/SQTC_derivations.md)** — detailed derivations: IFC ridge regression, symmetry projections, QHA, quantum thermodynamics, postprocessor pipeline

---

## License

This project is not yet licensed. Contact the author before use or redistribution.
