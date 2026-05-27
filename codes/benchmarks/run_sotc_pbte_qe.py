#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: PbTe rocksalt at 700 K  (high anharmonicity)
=================================================================================
PbTe is the most strongly anharmonic benchmark: large Grüneisen parameter
(γ ≈ 2.3), very soft acoustic modes, and a narrow PBE band gap (~0.19 eV).
SQTC typically requires 5-8 iterations to converge the renormalised IFCs.

QE settings  (SSSP efficiency pseudopotentials)
-----------------------------------------------
  Pb : Pb.pbe-dn-kjpaw_psl.0.2.2.UPF  (PAW)
  Te : Te_pbe_v1.uspp.F.UPF            (USPP)
  ecutwfc         : 44 Ry  (Pb PAW drives the cutoff; Te USPP = 30 Ry)
  ecutrho         : 352 Ry  (8 × ecutwfc for PAW/USPP mix)
  k-mesh (SC)     : 1×1×1  (64-atom supercell, Γ-only)
  Occupations     : fixed  (PBE gap ~0.19 eV; treated as insulator at Γ)
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/examples/run_sotc_pbte_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: PbTe rocksalt ───────────────────────────────────────

a_pbte   = 6.51           # Å  (experimental PbTe at ~700 K)
M_Pb     = 207.2          # amu
M_Te     = 127.6          # amu
T_D_PbTe = 130.0          # K  (seed from low-T Cp; T/T_D = 5.4 at 700 K)
T_design = 700.0          # K

N_SC = 64                 # 64-atom rocksalt supercell (32 f.u.)

# Rocksalt = FCC Bravais + 2-atom basis
prim_cell_pbte = 0.5 * a_pbte * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_pbte = np.array([
    [0.0,       0.0, 0.0],   # Pb at origin
    [a_pbte/2,  0.0, 0.0],   # Te at (a/2, 0, 0)
])

WORK_DIR = Path("examples/sotc_pbte_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

# Default: pseudopotentials/ bundled alongside this repository
# Override with the SSSP_DIR environment variable if needed.
SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
SSP_DIR = SSP_DIR  # alias kept for clarity

PB_UPF = "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF"
TE_UPF = "Te_pbe_v1.uspp.F.UPF"

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────
# PbTe has a small PBE gap (~0.19 eV); treat as insulator with fixed occupations.
# The Γ-only k-mesh avoids any metallic crossings in the supercell BZ.

qe_calc = QEForceCalculator(
    species=["Pb", "Te"],
    pseudopotentials={"Pb": PB_UPF, "Te": TE_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=44.0,                               # Ry  (Pb PAW drives cutoff)
    ecutrho=352.0,                              # Ry  (8 × ecutwfc)
    kpts=(1, 1, 1),                             # Γ-only for 64-atom supercell
    smearing=None,                              # fixed occupations (narrow-gap semi)
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="pbte",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: PbTe rocksalt at 700 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms (32 f.u.)")
    print(f"  r_cutoff         : 5.0 Å  (captures Pb-Te and Pb-Pb NN shells)")
    print(f"  T_design         : {T_design} K  (T/T_D = {T_design/T_D_PbTe:.1f})")
    print(f"  T_D (input)      : {T_D_PbTe} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}  (Γ-only)")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()

    runner = SOTCRunner(
        element="Pb",
        mass_amu=M_Pb,
        elements=["Pb", "Te"],
        masses_amu=[M_Pb, M_Te],
        prim_cell=prim_cell_pbte,
        prim_positions=prim_pos_pbte,
        T=T_design,
        T_D=T_D_PbTe,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=5.0,
        r_max_corr=10.0,
        n_ensemble=5,  # increased from 3; T/T_D=5.4 needs more data per iter
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.10,           # very conservative: T/T_D=5.4 prone to divergence spikes
        T_values=np.array([100., 200., 300., 500., 700.]),
        q_mesh_cv=(8, 8, 8),
    )

    print("\n" + "─" * 60)
    print(" SQTC PbTe benchmark summary")
    print("─" * 60)
    print(f"  Converged  : {results['converged']}")
    print(f"  Iterations : {results['n_iterations']}")
    print(f"  T_D        : {results['T_D_effective']:.1f} K  (expt: ~130 K)")
    print()
    print(f"  {'T [K]':>8}  {'C_V [J/mol/K]':>14}  {'Expt':>10}")
    print("  " + "-" * 38)
    expt = {100.: "~42.0", 200.: "~48.5", 300.: "~50.0", 500.: "~52.0", 700.: "~54.0"}
    for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T_v:8.0f}  {cv_v:14.4f}  {expt.get(T_v, '—'):>10}")
    print()
    print(f"  Results saved to {WORK_DIR}/sotc_results.json")


if __name__ == "__main__":
    main()
