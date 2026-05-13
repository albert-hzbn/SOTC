#!/usr/bin/env python3
"""
SQTC + GPAW example: fcc Al at 300 K  (harmonic regime)
=========================================================
Al is a textbook weakly-anharmonic metal.  At T = 300 K  (T/T_D ≈ 0.77,
T/T_melt ≈ 0.32) the phonon frequencies are barely renormalized relative to
the harmonic values.  This script provides the harmonic GPAW reference for
the Al-vs-Pb comparison in compare_al_pb_gpaw.py.

GPAW settings
-------------
  Plane-wave cutoff : 450 eV  (standard for Al PAW)
  XC functional     : PBE
  k-mesh (supercell): 2×2×2 (scales from ~8×8×8 on 1-atom prim. cell)
  Smearing          : Fermi-Dirac, σ = 0.10 eV
  SCF convergence   : ΔE < 1×10⁻⁷ eV/electron

The 32-atom fcc supercell is ≈ 11.4×11.4×11.4 Å; for this size a (2,2,2)
k-mesh gives forces accurate to ~1 meV/Å, which is sufficient for SQTC IFC
regression.

Usage
-----
    # Local (for testing — very slow without parallelism):
    python codes/examples/run_sqtc_al_gpaw.py

    # On a cluster: see slurm_gpaw.sh

Output
------
    sqtc_al_gpaw_run/sqtc_results.json
    sqtc_al_gpaw_run/phonon_bandstructure.npz
    sqtc_al_gpaw_run/phonon_dos.npz
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sqtc import SQTCRunner
from sqtc.gpaw_io import GPAWForceCalculator

# ── Physical parameters: fcc Al ──────────────────────────────────────────────

a_al  = 4.05          # Å  (DFT-PBE equilibrium lattice constant)
M_Al  = 26.9815385    # amu
T_D_Al = 390.0        # K   (experimental Debye temperature)
T_design = 300.0      # K   design temperature (harmonic regime)

N_SC = 32             # target supercell size (atoms)

prim_cell_al = 0.5 * a_al * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])                    # fcc primitive cell (3×3) [Å]

prim_pos_al = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("sqtc_al_gpaw_run")

# ── GPAW calculator ───────────────────────────────────────────────────────────
# Pass one formula unit; GPAWForceCalculator tiles it to the actual supercell
# size chosen by SQTCRunner (which may differ slightly from N_SC).

gpaw_calc = GPAWForceCalculator(
    species=["Al"],  # tiled automatically to match supercell
    cutoff=450.0,
    xc="PBE",
    kpts=(2, 2, 2),
    smearing_width=0.10,
    convergence_energy=1e-7,
    eigensolver="rmm-diis",
    txt=str(WORK_DIR / "gpaw_al.out"),
    workdir=WORK_DIR / "gpaw_restarts",
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print(" SQTC + GPAW: fcc Al at 300 K")
print("=" * 68)
print(f"  Supercell size  : {N_SC} atoms")
print(f"  r_cutoff        : 4.5 Å  (1st + 2nd NN for fcc Al)")
print(f"  T_design        : {T_design} K")
print(f"  T_D (input)     : {T_D_Al} K")
print(f"  GPAW cutoff     : {gpaw_calc.cutoff:.0f} eV")
print(f"  k-mesh (SC)     : {gpaw_calc.kpts}")
print()

runner = SQTCRunner(
    element="Al",
    mass_amu=M_Al,
    prim_cell=prim_cell_al,
    prim_positions=prim_pos_al,
    T=T_design,
    T_D=T_D_Al,
    n_atoms_sc=N_SC,
    force_calculator=gpaw_calc,
    r_cutoff=4.5,           # Å  — includes 1st (2.86 Å) and 2nd (4.05 Å) NN
    r_max_corr=8.0,
    n_ensemble=5,
    work_dir=WORK_DIR,
    verbosity=1,
    ridge_alpha=1e-3,
    symmetrize_bonds=True,  # recommended for fcc monatomic with small ensemble
)

results = runner.run(
    n_sc_iterations=4,
    epsilon_conv=0.003,
    mixing=0.40,
    T_values=np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0,
                       350.0, 400.0, 500.0, 600.0, 700.0, 800.0]),
    q_mesh_cv=(8, 8, 8),
)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print(" Al GPAW result summary")
print("─" * 60)
print(f"  Converged          : {results['converged']}")
print(f"  Iterations         : {results['n_iterations']}")
print(f"  T_D (spectral)     : {results['T_D_effective']:.1f} K  "
      f"(exp: {T_D_Al:.0f} K)")
print(f"  ZPE / atom         : {results['ZPE_eV']*1e3:.2f} meV")
print(f"  C_V(300 K)         : {results['C_V_jmolk']:.3f} J/(mol·K)  "
      f"(3R = 24.94)")
print(f"  Unstable fraction  : {results['unstable_fraction']*100:.1f} %")
print()
print("  C_V scan:")
print(f"  {'T [K]':>8}  {'C_V [J/(mol·K)]':>16}")
print("  " + "-" * 28)
for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
    print(f"  {T_v:8.1f}  {cv_v:16.4f}")
print()
print(f"  Results saved → {WORK_DIR / 'sqtc_results.json'}")
