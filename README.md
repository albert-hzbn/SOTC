# SOTC — Special Optimal Thermal Cell

A first-principles framework for computing high-temperature phonon dispersions,
density of states, and thermodynamic properties from **purposefully designed
small supercells** — without large stochastic ensembles or molecular dynamics.

---

## Overview

Standard high-temperature phonon methods (TDEP, SSCHA, AIMD) require large
supercells — typically 54–500+ atoms — making DFT prohibitively expensive for
many systems. SOTC extends the **Special Quasirandom Structures (SQS)**
philosophy to the thermal problem:

> Just as an SQS encodes the statistical signature of a random alloy in the
> smallest possible periodic cell, an SOTC encodes the **thermal
> displacement–displacement correlations** of a disordered crystal in a small,
> self-consistently optimised cell.

A single SOTC with 32–64 atoms replaces hundreds of stochastic snapshots while
reproducing phonon properties to within a few percent of experiment.

---

## Method Summary

1. **Correlator design** — target displacement–displacement correlations
   $C_2(R, T)$ are computed from a trial phonon model using the quantum
   $\coth$ formula.
2. **Cell optimisation** — atomic displacements are optimised (gradient descent
   on a correlator mismatch cost $\mathcal{Q}$) so that one periodic cell best
   matches the target correlator hierarchy up to cutoff $r_\text{cut}$.
3. **DFT forces** — a single DFT call (Quantum ESPRESSO) computes
   forces on the displaced supercell.
4. **IFC extraction** — interatomic force constants (IFCs) are fitted by ridge
   regression with acoustic sum rule enforcement:
   $\hat{\boldsymbol{\Phi}} = -(\mathbf{G}^T\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{G}^T\mathbf{F}$
5. **Self-consistency** — the new phonon model updates the correlator targets;
   steps 1–4 repeat until convergence (typically 2–8 DFT calls total).
6. **Thermodynamics** — phonon DOS → $C_V(T)$, ZPE, $S_\text{vib}(T)$,
   $F_\text{vib}(T)$, MSD$(T)$, Debye–Waller $B(T)$.  The isochoric heat
   capacity over an arbitrary temperature range is obtained by applying the
   harmonic Einstein/Debye formula to the SOTC-renormalised frequencies —
   no QHA is needed for $C_V(T)$.

---

## Repository Structure

```
codes/
  sotc/                    ← Python package (import as `from sotc import ...`)
    runner.py              ←   Self-consistent SOTC loop (SOTCRunner)
    postprocessor.py       ←   CLI post-processor: phonons + thermodynamics
    cell_design.py         ←   Correlator-optimised cell construction (HNF)
    correlators.py         ←   Quantum coth correlator targets
    ifc_extractor.py       ←   Ridge-regression IFC fitting + ASR enforcement
    anharmonic_ifc.py      ←   Mode-space and reference finite-difference fc3/fc4
    anharmonic_io.py       ←   phono3py, ShengBTE, and FourPhonon IFC exporters
    symfc_fit.py           ←   Symmetry/sum-rule constrained compact fc2/fc3/fc4
    phonons.py             ←   Phonon dispersion, DOS, thermodynamics
    qha.py                 ←   Quasi-harmonic approximation (α, C_P, B, V(T))
    qe_io.py               ←   pw.x I/O via ASE Espresso (QE backend)
    qe_dfpt.py             ←   ph.x Born charges, dielectric tensor, LO–TO NAC
    gpaw_io.py             ←   GPAW force calculator
    mock_forces.py         ←   Analytic test forces (He, H₂) — no DFT needed
    constants.py           ←   Physical constants
  benchmarks/              ← Ready-to-run driver scripts for all systems
  tests/                   ← pytest test suite
  validation/              ← Independent FC3/FC4 and real QE DFPT campaigns

validation_results/        ← Compact QE/IFC validation outputs and summaries

pseudopotentials/          ← Bundled SSSP efficiency + ONCV UPF files for QE
manuscript/                ← Paper source (RevTeX4-2 + BibTeX)
requirements.txt           ← Python dependencies
README.md                  ← this file
```

The manuscript sources are kept separate: `manuscript_original.tex`/`.pdf` archive the original submission, `manuscript_clean.tex`/`.pdf` are the uncolored revised submission, and `manuscript_updated.tex`/`.pdf` are the selectively marked revision. The marked source contains only the revised text plus blue color commands; hidden deleted text and `latexdiff` tracking macros have been removed.

---

## Installation

### Requirements

| Requirement | Min. version | Notes |
|---|---|---|
| Python | 3.9 | |
| NumPy | 1.24 | |
| SciPy | 1.10 | |
| Matplotlib | 3.7 | |
| spglib | 2.1 | structure auto-detection, bond symmetrisation |
| ASE | 3.22 | required for the QE force calculator |
| pytest | 7.0 | test suite |
| **QE `pw.x`** | 7.0 | DFT forces via Quantum ESPRESSO |

QE is only needed for real calculations; the mock-force drivers work
without either.

### Step 1 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
```

### Step 2 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Add the package to your Python path

No install step is needed. Add `codes/` to `PYTHONPATH` once per shell
session (or put it in your `.bashrc`):

```bash
export PYTHONPATH="$PWD/codes:$PYTHONPATH"
```

### Step 4 — Verify

```bash
pytest codes/tests/ -v
```

All tests use built-in Lennard-Jones / Aziz mock potentials and pass without QE.

### Production real-material FC3/FC4 validation

The optional validation environment adds phonopy, phono3py, symfc, and HDF5:

```bash
pip install -r requirements-validation.txt
```

The production driver computes 3×3×3-supercell FC3 and FC4 tensors for fcc Al
at 300 K, bcc W at 1500 K, or rocksalt NaCl at 700 K:

```bash
export QE_PW_CMD="mpiexec -np 128 /path/to/pw.x"
export SSSP_DIR="$PWD/pseudopotentials"
python3 codes/validation/run_real_anharmonic_ifcs.py al
```

Each campaign uses 100 paired quantum-canonical snapshots, an 80/20
pair-independent train/test split, joint symmetry- and acoustic-sum-rule
constrained fits, and explicit snapshot-count, cutoff, plane-wave-cutoff, and
k-mesh controls. QE states are checkpointed locally so interrupted campaigns
can be restarted. The public repository contains the validation scripts,
compact summaries, selected figure arrays, constrained IFC tensors, and final
figures needed to audit the revised manuscript. Large QE scratch directories
and restart checkpoints are deliberately excluded from Git; they can be
regenerated from the archived inputs and drivers.

After all three campaigns finish, regenerate the cross-material acceptance
report directly from their machine-readable archives:

```bash
python3 codes/validation/summarize_real_anharmonic_ifcs.py
```

Regenerate the manuscript validation plots from the same archives:

```bash
python3 codes/validation/plot_ifc34_diagnostics.py
python3 codes/validation/plot_nacl_loto.py
python3 codes/validation/plot_matched_sampling_ablation.py
python3 codes/validation/plot_sotc_polar_nac.py
```

---

## Quick Start — fcc Al with Quantum ESPRESSO (300 K, harmonic benchmark)

Al at 300 K is weakly anharmonic ($\gamma \approx 2.17$, $T_D \approx 390$ K)
and converges in 3–4 iterations. It is the standard first test for a new QE
setup.

### What you need

- `pw.x` binary built with MPI (e.g. `~/Softwares/qe-7.5/bin/pw.x`)
- `pseudopotentials/Al.pbe-n-kjpaw_psl.1.0.0.UPF` — already included

### 1. Set environment variables

```bash
export QE_PW_CMD="mpirun -np 8 ~/Softwares/qe-7.5/bin/pw.x"
export SSSP_DIR="$PWD/pseudopotentials"
```

### 2. Run the SOTC self-consistent loop

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/codes:$PYTHONPATH"

python3 codes/benchmarks/run_sotc_al_qe.py
```

To restart a partially completed run:

```bash
python3 codes/benchmarks/run_sotc_al_qe.py --restart
```

Intermediate results are written after every iteration to
`sotc_al_qe_run/sotc_results.json`. Scratch files go to
`sotc_al_qe_run/qe_scratch/snap_NNNN/`.

### 3. Check convergence

```bash
python3 -m json.tool sotc_al_qe_run/sotc_results.json \
    | grep -E '"converged"|"n_iterations"|"T_D_effective"|"C_V_jmolk"'
```

Expected output:

```
"T_D_effective": 429.1,
"C_V_jmolk": 24.21,
"converged": true,
"n_iterations": 3,
```

### 4. Post-process

```bash
python3 codes/sotc/postprocessor.py \
    --run-dir sotc_al_qe_run \
    --structure fcc          \
    --calculator qe
```

All parameters (lattice constant, elements, masses, $T_\text{design}$,
$r_\text{cut}$) are **auto-detected** from `sotc_results.json` and the
first snapshot directory. Output goes to `./sotc_al_qe_run/postproc/`:

| File | Contents |
|---|---|
| `al_qe_phonons.pdf` | Phonon band structure + projected DOS |
| `al_qe_thermal.pdf` | $C_V(T)$, $S_\text{vib}(T)$, $\Theta_D(T)$, MSD$(T)$ |
| `phonon_bandstructure.npz` | q-path, frequencies (THz) |
| `phonon_dos.npz` | Frequency grid, $g(\omega)$ |
| `thermal_properties.npz` | Full temperature-resolved thermodynamics |
| `thermal_summary.json` | Human-readable JSON summary |

---

## Example 1 — fcc Cu with QE (strongly anharmonic, 1200 K)

Cu at 1200 K ($T/T_\text{melt} = 0.88$) is a demanding anharmonicity test;
SOTC needs 6–8 iterations to converge the renormalised IFCs.

### QE settings

| Parameter | Value | Notes |
|---|---|---|
| Pseudopotential | `Cu.paw.z_11.ld1.psl.v1.0.0-low.upf` | PAW, 11 val. e⁻ |
| `ecutwfc` | 45 Ry (≈ 612 eV) | SSSP efficiency |
| `ecutrho` | 360 Ry | 8 × ecutwfc (PAW) |
| k-mesh (32-atom SC) | 2×2×2 | scales from 8×8×8 on primitive cell |
| Smearing | Marzari–Vanderbilt cold | σ = 0.02 Ry |
| SCF threshold | 1×10⁻⁸ Ry | forces accurate to ~0.1 meV/Å |

### Run

```bash
export QE_PW_CMD="mpirun -np 64 ~/Softwares/qe-7.5/bin/pw.x"
export SSSP_DIR="$PWD/pseudopotentials"

python3 codes/benchmarks/run_sotc_cu_qe.py
```

### Python API

```python
import numpy as np
from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

a_cu = 3.67    # Å at ~1200 K (thermally expanded)

prim_cell = 0.5 * a_cu * np.array([
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.],
])

qe_calc = QEForceCalculator(
    species=["Cu"],
    pseudopotentials={"Cu": "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf"},
    pseudo_dir="pseudopotentials/",
    ecutwfc=45.0,                       # Ry
    ecutrho=360.0,                      # Ry  (8 × ecutwfc for PAW)
    kpts=(2, 2, 2),
    smearing="marzari-vanderbilt",
    degauss=0.02,                       # Ry
    conv_thr=1e-8,                      # Ry
    pw_cmd="mpirun -np 64 pw.x",
    workdir="sotc_cu_qe_run/qe_scratch",
    prefix="cu",
)

runner = SOTCRunner(
    element="Cu",
    mass_amu=63.546,
    prim_cell=prim_cell,
    prim_positions=np.array([[0., 0., 0.]]),
    T=1200.0,
    T_D=310.0,           # seed Debye temperature [K]
    n_atoms_sc=32,
    force_calculator=qe_calc,
    r_cutoff=4.5,        # Å — 1st + 2nd NN shells
    r_max_corr=9.0,      # Å — correlator fitting range
    n_ensemble=8,
    ridge_alpha=1e-3,
    symmetrize_bonds=True,
    work_dir="sotc_cu_qe_run",
    verbosity=1,
)

results = runner.run(
    n_sc_iterations=10,
    epsilon_conv=0.003,   # J/(mol·K) — convergence on ΔC_V
    min_iterations=4,
    mixing=0.20,
    T_values=np.array([100., 300., 500., 700., 900., 1200.]),
    q_mesh_cv=(8, 8, 8),
)

print(f"Converged : {results['converged']}")
print(f"T_D       : {results['T_D_effective']:.1f} K  (expt: 315 K)")
print(f"C_V(300K) : {results['C_V_scan'][1]:.2f} J/(mol·K)  (expt: ~24.5)")
```

### Post-process

```bash
python3 codes/sotc/postprocessor.py \
    --run-dir sotc_cu_qe_run \
    --structure fcc          \
    --calculator qe          \
    --T-max 1400
```

---

## Example 2 — MgO rocksalt with QE (insulator, 1500 K)

MgO is a stiff ionic insulator ($T_D \approx 740$ K, $\gamma \approx 1.5$).
Key differences from metals: no smearing, higher `ecutwfc` (O drives the
cutoff), Γ-only k-mesh for the 64-atom supercell.

### QE settings

| Parameter | Value | Notes |
|---|---|---|
| Pseudopotentials | `Mg.pbe-n-kjpaw_psl.0.3.0.UPF`, `O.pbe-n-kjpaw_psl.0.1.UPF` | PAW |
| `ecutwfc` | 80 Ry | O drives the cutoff (Mg needs ~35 Ry, O needs ~80 Ry) |
| `ecutrho` | 640 Ry | 8 × ecutwfc for PAW |
| k-mesh (64-atom SC) | Γ-only (1×1×1) | insulator, large supercell |
| Occupations | `fixed` | **no smearing** for insulators |
| SCF threshold | 1×10⁻⁸ Ry | |

### Python API

```python
import numpy as np
from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

a_mgo = 4.26   # Å at ~1500 K

prim_cell = 0.5 * a_mgo * np.array([
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.],
])

# Rocksalt basis: Mg at origin, O at face-centre
prim_pos = np.array([
    [0.0,       0.0, 0.0],   # Mg
    [a_mgo/2,   0.0, 0.0],   # O
])

qe_calc = QEForceCalculator(
    species=["Mg", "O"],
    pseudopotentials={
        "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
        "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
    },
    pseudo_dir="pseudopotentials/",
    ecutwfc=80.0,
    ecutrho=640.0,
    kpts=(1, 1, 1),
    smearing=None,           # insulator — fixed occupations
    conv_thr=1e-8,
    pw_cmd="mpirun -np 128 pw.x",
    workdir="sotc_mgo_qe_run/qe_scratch",
    prefix="mgo",
)

runner = SOTCRunner(
    element="MgO",
    mass_amu=[24.305, 15.999],
    prim_cell=prim_cell,
    prim_positions=prim_pos,
    T=1500.0,
    T_D=740.0,
    n_atoms_sc=64,
    force_calculator=qe_calc,
    r_cutoff=3.2,            # Å — Mg–O and Mg–Mg 1st NN
    r_max_corr=9.0,
    n_ensemble=3,
    ridge_alpha=1e-3,
    symmetrize_bonds=True,
    work_dir="sotc_mgo_qe_run",
)

results = runner.run(
    n_sc_iterations=10,
    epsilon_conv=0.003,
    min_iterations=4,
    mixing=0.30,
    T_values=np.array([300., 600., 900., 1200., 1500.]),
    q_mesh_cv=(8, 8, 8),
)
```

### Run

```bash
python3 codes/benchmarks/run_sotc_mgo_qe.py
```

### Post-process

```bash
python3 codes/sotc/postprocessor.py \
    --run-dir sotc_mgo_qe_run \
    --structure rocksalt      \
    --calculator qe
```

---

## Example 3 — NaCl rocksalt with QE (ionic, USPP, 700 K)

NaCl uses ultrasoft pseudopotentials (USPP) from the SSSP library. The
`ecutrho` factor is the same (8×) but note `smearing=None` since NaCl is
an insulator.

### QE settings

| Parameter | Value | Notes |
|---|---|---|
| Pseudopotentials | `na_pbe_v1.5.uspp.F.UPF`, `cl_pbe_v1.4.uspp.F.UPF` | USPP |
| `ecutwfc` | 60 Ry | Cl drives the cutoff |
| `ecutrho` | 480 Ry | 8 × ecutwfc for USPP |
| k-mesh (64-atom SC) | Γ-only (1×1×1) | insulator |
| Occupations | `fixed` | no smearing |

```bash
python3 codes/benchmarks/run_sotc_nacl_qe.py
```

Post-process:

```bash
python3 codes/sotc/postprocessor.py \
    --run-dir sotc_nacl_qe_run \
    --structure rocksalt        \
    --calculator qe
```

---

## Quasi-harmonic API (outside the revised manuscript validation)

The underconverged three-volume QHA result from the original manuscript was
withdrawn and is not part of the revised validation. No QHA cost or accuracy
claim is made here.

The current `SOTCQuasiHarmonic` API requires at least five directly calculated
volumes, uses static reference-lattice energies, distinguishes volumetric and
cubic linear expansion, includes the external-pressure term explicitly, and
forbids phonon-free-energy extrapolation outside the direct volume grid. Users
must still demonstrate convergence with respect to the volume grid, EOS model,
electronic settings, and interpolation before interpreting a QHA result.

The safeguards are covered by:

```bash
pytest codes/tests/test_qha_convergence.py -v
```

---

## All QE Benchmark Scripts

All scripts live in `codes/benchmarks/`.

| Script | Material | Structure | $T_\text{design}$ | Supercell |
|---|---|---|---|---|
| `run_sotc_al_qe.py` | Al | FCC | 300 K | 32 atoms |
| `run_sotc_cu_qe.py` | Cu | FCC | 1200 K | 32 atoms |
| `run_sotc_ag_qe.py` | Ag | FCC | 1200 K | 32 atoms |
| `run_sotc_au_qe.py` | Au | FCC | 1200 K | 32 atoms |
| `run_sotc_ni_qe.py` | Ni | FCC | 1200 K | 32 atoms |
| `run_sotc_mo_qe.py` | Mo | BCC | 1500 K | 64 atoms |
| `run_sotc_w_qe.py` | W | BCC | 2000 K | 64 atoms |
| `run_sotc_mgo_qe.py` | MgO | Rocksalt | 1500 K | 64 atoms |
| `run_sotc_nacl_qe.py` | NaCl | Rocksalt | 700 K | 64 atoms |
| `run_sotc_pbte_qe.py` | PbTe | Rocksalt | 700 K | 64 atoms |

---

## Post-processor CLI Reference

```bash
python3 codes/sotc/postprocessor.py --run-dir <dir> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--run-dir PATH` | **required** | Path to the completed SOTC run directory |
| `--structure STR` | auto (spglib) | `fcc`, `bcc`, `sc`, `hcp`, `rocksalt`, `zincblende` |
| `--calculator STR` | auto | `qe` |
| `--qe-scratch-dir PATH` | auto | Override QE scratch directory |
| `--lattice a [c]` | auto | Lattice constant(s) in Å |
| `--elements E1 [E2]` | auto | Element symbols per basis atom |
| `--mass m1 [m2]` | auto | Atomic masses in amu |
| `--T-design T` | auto | Design temperature in K |
| `--r-cutoff R` | 6.0 | IFC cutoff radius in Å |
| `--ridge-alpha α` | 1e-3 | Tikhonov regularisation strength |
| `--symmetrize-bonds` | auto | Project IFCs onto central-force subspace |
| `--T-min T` | 10 | Start of temperature scan (K) |
| `--T-max T` | max(1.05·T_design, 1000) | End of temperature scan (K) |
| `--T-step dT` | 10 | Temperature step (K) |
| `--q-mesh-cv N N N` | 20 20 20 | q-mesh for $C_V$ integration |
| `--q-mesh-dos N N N` | 30 30 30 | q-mesh for DOS and band structure |
| `--out-dir PATH` | `<run_dir>/postproc` | Output directory |
| `--label TEXT` | run-dir name | Plot title / legend label |

Output files written to `<run_dir>/postproc/`:

| File | Contents |
|---|---|
| `<label>_phonons.pdf/png` | Phonon band structure + DOS |
| `<label>_thermal.pdf/png` | $C_V(T)$, $S_\text{vib}(T)$, $\Theta_D(T)$, MSD$(T)$ |
| `phonon_bandstructure.npz` | Band-structure arrays (q-path, frequencies) |
| `phonon_dos.npz` | DOS: frequency grid, $g(\omega)$ |
| `thermal_properties.npz` | Full temperature-resolved thermodynamics |
| `thermal_summary.json` | Human-readable JSON (see format below) |

---

## Key Design Choices

**Designed configurations per iteration** — the archived benchmarks use
material-specific ensembles of one to eight displaced configurations per
iteration plus an undisplaced reference. Force-call counts are reported
directly and are not converted into a universal external-method speedup.

**Ridge regression** — Tikhonov regularisation with default $\lambda = 10^{-3}$
eV/Å² suppresses noise amplification in under-determined IFC fits.

**Bond symmetrisation** — for high-symmetry structures (FCC, rocksalt), IFC
tensors are projected onto the central-force subspace
$A\hat{R}_\alpha\hat{R}_\beta + B(\delta_{\alpha\beta} - \hat{R}_\alpha\hat{R}_\beta)$,
reducing free parameters and improving stability. Auto-detected via spglib;
override with `--symmetrize-bonds`.

**Quantum correlators** — displacement targets use the exact quantum $\coth$
formula, correctly capturing zero-point motion at all temperatures.

**Temperature dependence without QHA** — once the self-consistent IFCs are
converged at $T_{\text{design}}$, $C_V(T')$ across an arbitrary temperature
range is computed by the harmonic sum

$$
C_V(T') = k_B N_A \frac{1}{N_q} \sum_{\mathbf{q},s}
\left(\frac{\hbar\omega_s(\mathbf{q})}{2 k_B T'}\right)^{2}
\text{csch}^2\left(\frac{\hbar\omega_s(\mathbf{q})}{2 k_B T'}\right)
$$

using the effective-FC2 frequencies $\{\omega_s\}$ fitted at
$T_\text{design}$. This is a fixed-volume, independent-phonon evaluation using
one fitted spectrum; it is not a calculation of the full temperature evolution
of anharmonic phonons. Thermal expansion, $C_P(T)$, and bulk modulus require a
separate, converged multi-volume workflow and are outside the revised
manuscript's demonstrated scope.

---

## `thermal_summary.json` Format

```json
{
  "label": "Cu QE 1200 K",
  "elements": ["Cu"],
  "T_design_K": 1200.0,
  "ZPE_eV": 0.0491,
  "TD_spectral_K": 343.0,
  "temperatures": [
    {
      "T_K": 300.0,
      "Cv_jmolk": 24.44,
      "Svib_jmolk": 38.1,
      "TD_caloric": 309.5,
      "MSD_ang2": 0.021,
      "DW_B_ang2": 0.53,
      "Fvib_meV": -89.2
    }
  ]
}
```

---

## Checking Convergence

SOTC iterates until $|\Delta C_V| < \varepsilon_\text{conv}$ (default
0.003 J/(mol·K)) between successive iterations. Inspect any run with:

```bash
python3 -m json.tool <run_dir>/sotc_results.json \
    | grep -E '"converged"|"n_iterations"|"T_D_effective"|"C_V_jmolk"'
```

---

## Pseudopotentials

SSSP efficiency UPF files for all benchmarked elements are included in
`pseudopotentials/`. For additional elements download from:

- [Materials Cloud — SSSP](https://www.materialscloud.org/discover/sssp)
- [PseudoDojo](http://www.pseudo-dojo.org/)

---

## Running Tests

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/codes:$PYTHONPATH"
pytest codes/tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
