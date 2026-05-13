# SQTC: Running and Post-Processing Guide

## Overview

Each SQTC calculation follows two steps:

1. **Run** — submit a SLURM job that generates thermally displaced snapshots, calls VASP for DFT forces, and iterates the SQTC self-consistency loop. Results are saved in `<run_dir>/`.
2. **Post-process** — refit the IFCs from all converged iterations and compute the phonon band structure, DOS, and thermodynamic properties. Output goes to `<run_dir>/postproc/`.

---

## Step 1: Running SQTC

### Directory structure

```
codes/examples/run_sqtc_<material>_vasp.py   ← Python driver for each material
slurm_<material>.sh                          ← SLURM submission script
```

### Submitting a job

```bash
sbatch slurm_au.sh       # fcc Au at 1200 K
sbatch slurm_cu.sh       # fcc Cu at 1200 K
sbatch slurm_mo.sh       # bcc Mo at 1500 K
sbatch slurm_mgo.sh      # rocksalt MgO at 1500 K
sbatch slurm_nacl.sh     # rocksalt NaCl at 700 K
sbatch slurm_w.sh        # bcc W at 2000 K
sbatch slurm_si.sh       # diamond-cubic Si at 1000 K
sbatch slurm_lif.sh      # rocksalt LiF at 800 K
sbatch slurm_ni.sh       # fcc Ni at 1200 K
sbatch slurm_mg.sh       # hcp Mg at 700 K
sbatch slurm_pbte.sh     # rocksalt PbTe at 700 K
sbatch slurm_al_qha.sh   # fcc Al QHA (3 volumes) at 300 K
```

### What each SLURM script does

Every `slurm_*.sh` script:
- Loads `mkl`, `intel`, `impi` modules
- Activates the Python venv at `.venv/`
- Runs `python3 codes/examples/run_sqtc_<material>_vasp.py`

VASP is called via `srun` inside the Python driver; no external job arrays are needed.

### Key parameters in each driver script

| Parameter | Description |
|-----------|-------------|
| `a` (Å) | Primitive-cell lattice constant at `T_design` (thermal expansion accounted for) |
| `T_design` (K) | SQTC design temperature — sets displacement magnitudes |
| `n_atoms_sc` | Supercell size (atoms). Must satisfy supercell side > 2 × `r_cutoff` |
| `r_cutoff` (Å) | IFC cutoff radius — determines how many NN shells are included |
| `ridge_alpha` | Tikhonov regularisation strength (default `1e-3`) |

### Completed runs

| Run directory | Material | Structure | `T_design` | Supercell | `r_cutoff` | Converged | Iterations |
|---|---|---|---|---|---|---|---|
| `sqtc_au_vasp_run` | Au | fcc | 1200 K | 32 atoms | 4.5 Å | ✓ | 3 |
| `sqtc_cu_vasp_run` | Cu | fcc | 1200 K | 32 atoms | 4.5 Å | — | 8 |
| `sqtc_al_fast_vasp_run` | Al | fcc | 300 K | 32 atoms | 4.5 Å | ✓ | 2 |
| `sqtc_fccni_vasp_run` | Ni | fcc | 1200 K | 32 atoms | 4.5 Å | ✓ | 2 |
| `sqtc_bccmo_vasp_run` | Mo | bcc | 1500 K | 64 atoms | 5.0 Å | ✓ | 2 |
| `sqtc_bccw_vasp_run` | W | bcc | 2000 K | 64 atoms | 5.0 Å | ✓ | 2 |
| `sqtc_bccti_vasp_run` | Ti | bcc | 1200 K | 32 atoms | 4.0 Å | ✓ | 3 |
| `sqtc_mgo_vasp_run` | MgO | rocksalt | 1500 K | 64 atoms | 3.2 Å | ✓ | 2 |
| `sqtc_nacl_vasp_run` | NaCl | rocksalt | 700 K | 64 atoms | 4.3 Å | ✓ | 2 |
| `sqtc_lif_vasp_run` | LiF | rocksalt | 800 K | 64 atoms | 3.5 Å | ✓ | 2 |
| `sqtc_pbte_vasp_run` | PbTe | rocksalt | 700 K | 64 atoms | 5.0 Å | ✓ | 7 |
| `sqtc_si_vasp_run` | Si | zincblende | 1000 K | 64 atoms | 4.5 Å | ✓ | 6 |
| `sqtc_hcpmg_vasp_run` | Mg | hcp | 700 K | 72 atoms | 5.5 Å | ✓ | 2 |
| `sqtc_al_qha_run` | Al | fcc (×3 vol) | 300 K | 32 atoms | 4.5 Å | ✓ | — |

---

## Step 2: Post-processing

Post-processing refits the IFCs from all converged snapshots and computes:
- Phonon band structure along standard high-symmetry paths
- Phonon DOS
- Harmonic thermodynamic properties: $C_V(T)$, $S_\text{vib}(T)$, $F_\text{vib}(T)$, MSD$(T)$, Debye-Waller $B(T)$, caloric Debye temperature $\Theta_D(T)$

### Generic postprocessor (any material)

```bash
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir <run_dir> \
    --structure <fcc|bcc|sc|hcp|rocksalt|zincblende>
```

All parameters are **auto-detected** from the run directory:
- Lattice constant `a` from the first POSCAR
- Elements and masses from POSCAR species line + IUPAC 2021 table
- `T_design` from `sqtc_results.json`
- `r_cutoff` from `sqtc_results.json` (fallback: analytical NN-shell midpoint)

Output is written to `<run_dir>/postproc/`.

### Post-processing commands for all completed runs

#### fcc metals

```bash
# fcc Au — 1200 K (T/T_melt = 0.90, extreme anharmonicity)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_au_vasp_run --structure fcc

# fcc Cu — 1200 K (T/T_melt = 0.88, strongly anharmonic)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_cu_vasp_run --structure fcc

# fcc Al — 300 K (harmonic benchmark)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_al_fast_vasp_run --structure fcc

# fcc Ni — 1200 K (T/T_C = 1.9, paramagnetic)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_fccni_vasp_run --structure fcc
```

#### bcc metals

```bash
# bcc Mo — 1500 K (T/T_melt = 0.52, refractory)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_bccmo_vasp_run --structure bcc

# bcc W — 2000 K (T/T_melt = 0.54, strongly refractory)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_bccw_vasp_run --structure bcc

# bcc Ti — 1200 K (above α→β transition at 1155 K)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_bccti_vasp_run --structure bcc
```

#### Rocksalt compounds

```bash
# MgO — 1500 K (low anharmonicity, stiff ionic)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_mgo_vasp_run --structure rocksalt

# NaCl — 700 K (moderate anharmonicity)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_nacl_vasp_run --structure rocksalt

# LiF — 800 K (T/T_D ≈ 1.1, near classical onset)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_lif_vasp_run --structure rocksalt

# PbTe — 700 K (strongly anharmonic chalcogenide)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_pbte_vasp_run --structure rocksalt
```

#### Other structures

```bash
# Si — 1000 K (zincblende, T/T_melt = 0.59)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_si_vasp_run --structure zincblende

# hcp Mg — 700 K (T/T_melt = 0.76)
.venv/bin/python codes/sqtc/postprocessor.py \
    --run-dir sqtc_hcpmg_vasp_run --structure hcp
```

### Optional CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lattice a [c]` | auto | Override lattice constant(s) in Å |
| `--elements E1 [E2]` | auto | Override element symbols |
| `--mass m1 [m2]` | auto | Override atomic masses in amu |
| `--T-design T` | auto | Override design temperature in K |
| `--r-cutoff R` | auto | Override IFC cutoff radius in Å |
| `--ridge-alpha α` | auto | Override Tikhonov regularisation |
| `--label "text"` | run dir name | Plot title / legend label |
| `--T-min T` | 10 | Start of temperature scan [K] |
| `--T-max T` | max(1.05·T_design, 1000) | End of temperature scan [K] |
| `--T-step dT` | 10 | Temperature step [K] |
| `--q-mesh-cv N N N` | 20 20 20 | q-mesh for $C_V$ integration |
| `--q-mesh-dos N N N` | 30 30 30 | q-mesh for DOS and band structure |
| `--out-dir path` | `<run_dir>/postproc` | Output directory |

---

## Post-processing output files

### Standard run (`<run_dir>/postproc/`)

| File | Description |
|------|-------------|
| `<label>_phonons.pdf/png` | Phonon band structure + DOS side by side |
| `<label>_thermal.pdf/png` | $C_V(T)$, $S_\text{vib}(T)$, $\Theta_D(T)$, MSD$(T)$ |
| `phonon_bandstructure.npz` | Band structure data: q-paths, frequencies |
| `phonon_dos.npz` | DOS data: frequencies, g(ω) |
| `thermal_properties.npz` | Full temperature-resolved thermodynamics |
| `thermal_summary.json` | Human-readable JSON with all thermodynamic quantities |

### QHA run (`sqtc_al_qha_run/postproc/`)

All standard outputs above, plus:

| File | Description |
|------|-------------|
| `al_qha_comparison.pdf/png` | QHA vs harmonic $C_P(T)$, $\alpha(T)$, $B(T)$ comparison |
| `al_phonon_dos_volumes.pdf/png` | Phonon DOS at the three volumes $V^-$, $V^0$, $V^+$ |
| `al_gruneisen.pdf/png` | Mode Grüneisen parameter $\gamma(\omega)$ |
| `al_thermal_properties_V0.pdf/png` | $C_V(T)$, $S_\text{vib}(T)$ at equilibrium volume |
| `al_qha_summary.json` | QHA results: $\alpha(T)$, $C_P(T)$, $B(T)$, $V(T)$ |
| `qha_results.npz` | Full QHA arrays |

### `thermal_summary.json` fields

```json
{
  "label": "...",
  "elements": ["Au"],
  "T_design_K": 1200.0,
  "ZPE_eV": 0.0207,
  "TD_spectral_K": 279.5,
  "temperatures": [
    {
      "T_K": 300.0,
      "Cv_jmolk": 24.94,
      "Svib_jmolk": 47.2,
      "TD_caloric": 165.0,
      "MSD_ang2": 0.018,
      "DW_B_ang2": 0.48,
      "Fvib_meV": -74.3
    },
    ...
  ]
}
```

---

## Checking convergence

SQTC iterates until the heat capacity $C_V$ changes by less than 0.05 J/(mol·K) between iterations. Check the final state:

```bash
python3 -m json.tool sqtc_au_vasp_run/sqtc_results.json | \
    grep -E '"converged"|"n_iterations"|"T_D_effective"|"C_V_jmolk"'
```

Example output:
```
"T_D_effective": 223.4,
"C_V_jmolk": 24.90,
"converged": true,
"n_iterations": 3,
```

---

## Running with a custom script

For materials not in `codes/examples/`, or to customise the workflow:

```python
from sqtc import SQTCRunner
import numpy as np

a = 4.13  # Å at T_design
prim_cell = 0.5 * a * np.array([[0,1,1],[1,0,1],[1,1,0]])  # fcc
prim_pos   = np.array([[0.0, 0.0, 0.0]])

runner = SQTCRunner(
    prim_cell     = prim_cell,
    prim_pos      = prim_pos,
    masses_amu    = [196.967],
    elements      = ["Au"],
    work_dir      = "my_au_run",
    T             = 1200.0,
    T_D           = 160.0,       # seed Debye temperature for displacement magnitude
    n_atoms_sc    = 32,
    r_cutoff      = 4.5,
    ridge_alpha   = 1e-3,
    vasp_cmd      = "srun --ntasks=64 /path/to/vasp_std",
    pp_base_dir   = "/path/to/pseudopotentials",
    n_iterations  = 5,
)
runner.run()
```
