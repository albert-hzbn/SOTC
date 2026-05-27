#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO benchmark: NaCl rocksalt at 700 K  (moderate anharmonicity)
=====================================================================================
NaCl is the prototypical ionic compound benchmark.  The LO-TO splitting is not
included in the short-range IFC framework; SQTC recovers TO-only phonons.

QE settings  (SSSP efficiency pseudopotentials)
-----------------------------------------------
  Na : na_pbe_v1.5.uspp.F.UPF  (USPP)
  Cl : cl_pbe_v1.4.uspp.F.UPF  (USPP)
  ecutwfc         : 60 Ry  (max of Na=40, Cl=60)
  ecutrho         : 480 Ry  (8 × ecutwfc for USPP)
  k-mesh (SC)     : 1×1×1  (64-atom rocksalt supercell, Γ-only)
  Occupations     : fixed  (insulator, no smearing)
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Usage
-----
    QE_PW_CMD="mpirun -np 64 pw.x" python3 codes/examples/run_sotc_nacl_qe.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: NaCl rocksalt ───────────────────────────────────────

a_nacl   = 5.73           # Å  (experimental NaCl at ~700 K)
M_Na     = 22.9898        # amu
M_Cl     = 35.453         # amu
T_D_NaCl = 310.0          # K  (seed; calorimetric expt ≈ 321 K)
T_design = 700.0          # K

N_SC = 64                 # 64-atom rocksalt supercell (32 f.u.)

# Rocksalt = FCC Bravais + 2-atom basis
prim_cell_nacl = 0.5 * a_nacl * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_nacl = np.array([
    [0.0,        0.0, 0.0],   # Na at origin
    [a_nacl/2,   0.0, 0.0],   # Cl at (a/2, 0, 0)
])

WORK_DIR = Path("examples/sotc_nacl_qe_run")

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

NA_UPF = "na_pbe_v1.5.uspp.F.UPF"
CL_UPF = "cl_pbe_v1.4.uspp.F.UPF"

PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────
# NaCl is an insulator (band gap ~8.9 eV): use smearing=None → occupations='fixed'

qe_calc = QEForceCalculator(
    species=["Na", "Cl"],
    pseudopotentials={"Na": NA_UPF, "Cl": CL_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=60.0,                               # Ry  (Cl drives the cutoff)
    ecutrho=480.0,                              # Ry  (8 × ecutwfc for USPP)
    kpts=(1, 1, 1),                             # Γ-only for 64-atom supercell
    smearing=None,                              # insulator: fixed occupations
    conv_thr=1.0e-8,                            # Ry
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="nacl",
    n_parallel=1,
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: NaCl rocksalt at 700 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms (32 f.u.)")
    print(f"  r_cutoff         : 4.3 Å  (captures Na-Cl and Na-Na shells)")
    print(f"  T_design         : {T_design} K  (T/T_D = {T_design/T_D_NaCl:.1f})")
    print(f"  T_D (input)      : {T_D_NaCl} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}  (Γ-only, insulator)")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print()
    print("  NOTE: LO-TO splitting not included; TO-only T_D is expected")
    print()

    runner = SOTCRunner(
        element="Na",
        mass_amu=M_Na,
        elements=["Na", "Cl"],
        masses_amu=[M_Na, M_Cl],
        prim_cell=prim_cell_nacl,
        prim_positions=prim_pos_nacl,
        T=T_design,
        T_D=T_D_NaCl,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.3,
        r_max_corr=9.0,
        n_ensemble=3,
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=10,
        epsilon_conv=0.003,
        min_iterations=4,
        mixing=0.30,
        T_values=np.array([200., 300., 400., 700., 900.]),
        q_mesh_cv=(6, 6, 6),
    )

    print("\n" + "─" * 60)
    print(" SQTC NaCl benchmark summary")
    print("─" * 60)
    print(f"  Converged  : {results['converged']}")
    print(f"  Iterations : {results['n_iterations']}")
    print(f"  T_D        : {results['T_D_effective']:.1f} K  (expt: 321 K)")
    print()
    print(f"  {'T [K]':>8}  {'C_V [J/mol/K]':>14}  {'Expt':>10}")
    print("  " + "-" * 38)
    expt = {200.: "~44.0", 300.: "~49.0", 400.: "~49.9", 700.: "~50.5", 900.: "~51.2"}
    for T_v, cv_v in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T_v:8.0f}  {cv_v:14.4f}  {expt.get(T_v, '—'):>10}")
    print()
    print("  Classical limit 6R = 49.883 J/(mol·K)  [per formula unit]")
    print(f"  Results saved to {WORK_DIR}/sotc_results.json")


if __name__ == "__main__":
    main()
