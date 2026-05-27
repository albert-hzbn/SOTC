#!/usr/bin/env python3
"""
SQTC + Quantum ESPRESSO example: fcc Al at 300 K  (harmonic regime)
=====================================================================
Demonstrates SQTC self-consistent phonon calculation for fcc Al using
Quantum ESPRESSO (pw.x) as the force engine via the ASE Espresso interface.

Physical background
-------------------
  Al is weakly anharmonic (Grüneisen γ ≈ 2.17, T_D ≈ 390 K).
  At 300 K the phonon self-renormalization is < 1 %, so this run serves as
  a clean harmonic benchmark.  SQTC converges in 3–4 iterations.

QE settings  (SSSP efficiency pseudopotential)
-----------------------------------------------
  Pseudopotential : Al.pbe-n-kjpaw_psl.1.0.0.UPF  (PAW, SSSP efficiency)
  ecutwfc         : 40 Ry  (≈ 544 eV)  [SSSP recommended]
  ecutrho         : 320 Ry (8 × ecutwfc)
  XC functional   : PBE  (encoded in the PAW pseudopotential)
  k-mesh (SC)     : 2×2×2  (scales from 8×8×8 on the primitive cell)
  Smearing        : Marzari-Vanderbilt cold smearing, σ = 0.02 Ry
  SCF convergence : ΔE_tot < 1×10⁻⁸ Ry

Lattice constant
----------------
  a = 4.05 Å  (DFT-PBE equilibrium; match your relaxed value if different).
  To use the experimental a = 4.0495 Å change a_al below.

Paths
-----
  QE binary     : ~/Softwares/qe-7.5/bin/pw.x
                  (override with env var QE_PW_CMD or pw_cmd= argument)
  Pseudopotentials: pseudopotentials/  (bundled in workspace root)
                  (override with SSSP_DIR env var or pseudo_dir= argument)

Usage
-----
    # Local (serial pw.x, slow — useful for testing):
    python codes/examples/run_sotc_al_qe.py

    # Restart after partial run (skips completed SQTC iterations):
    python codes/examples/run_sotc_al_qe.py --restart

    # On a cluster with MPI pw.x (edit pw_cmd below or set QE_PW_CMD):
    QE_PW_CMD="mpirun -np 32 pw.x" python codes/examples/run_sotc_al_qe.py

Output
------
    sotc_al_qe_run/sotc_results.json      — converged results
    sotc_al_qe_run/qe_scratch/snap_NNNN/  — per-snapshot pw.x directories
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "../"))

from sotc import SOTCRunner
from sotc.qe_io import QEForceCalculator

# ── Physical parameters: fcc Al ──────────────────────────────────────────────

a_al     = 4.05           # Å  (DFT-PBE equilibrium lattice constant)
M_Al     = 26.9815385     # amu
T_D_Al   = 390.0          # K   (experimental Debye temperature)
T_design = 300.0          # K   design temperature for SQTC correlators

N_SC = 32                 # target supercell size (atoms)

prim_cell_al = 0.5 * a_al * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])                         # fcc primitive cell [Å], rows = lattice vectors

prim_pos_al = np.array([[0.0, 0.0, 0.0]])

WORK_DIR = Path("examples/sotc_al_qe_run")

# ── QE settings ───────────────────────────────────────────────────────────────

# Default paths — override with environment variables if needed.
# Default: pseudopotentials/ bundled alongside this repository
# Override with the SSSP_DIR environment variable if needed.
SSP_DIR = Path(
    os.environ.get(
        "SSSP_DIR",
        str(Path(__file__).parent.parent.parent / "pseudopotentials"),
    )
)
AL_UPF = "Al.pbe-n-kjpaw_psl.1.0.0.UPF"

# pw.x command.  The binary was compiled with Intel MPI; it requires
# 'mpiexec' even for serial runs.  Set $QE_PW_CMD to override.
#
# Before running, load the required modules:
#   module load mkl/2025.2 intel/2025.3 impi/2021.17
#   export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
#   export I_MPI_HYDRA_BOOTSTRAP=ssh   # needed outside SLURM on login node
#
# For a SLURM job with 32 MPI ranks:
#   export QE_PW_CMD="mpirun -np 32 pw.x"
PW_CMD = os.environ.get(
    "QE_PW_CMD",
    "mpiexec -np 1 " + os.path.expanduser("~/Softwares/qe-7.5/bin/pw.x"),
)

# ── QE force calculator ───────────────────────────────────────────────────────

qe_calc = QEForceCalculator(
    species=["Al"],                             # tiled to actual supercell size
    pseudopotentials={"Al": AL_UPF},
    pseudo_dir=SSP_DIR,
    ecutwfc=40.0,                               # Ry  (SSSP efficiency recommendation)
    ecutrho=320.0,                              # Ry  (8 × ecutwfc, standard for PAW)
    kpts=(2, 2, 2),                             # 2×2×2 for 32-atom fcc supercell
    smearing="marzari-vanderbilt",              # cold smearing, good for metals
    degauss=0.02,                               # Ry  (≈ 0.27 eV)
    conv_thr=1.0e-8,                            # Ry  (tight: accurate forces)
    pw_cmd=PW_CMD,
    workdir=WORK_DIR / "qe_scratch",
    prefix="al",
    n_parallel=1,  # serial pw.x; increase if using multiple small QE jobs
)

# ── SQTC runner ───────────────────────────────────────────────────────────────

def main(restart: bool = False) -> None:
    results_file = WORK_DIR / "sotc_results.json"
    if restart and results_file.exists():
        print(f"  Found existing results at {results_file} — skipping run.")
        return

    print("\n" + "=" * 68)
    print(" SQTC + Quantum ESPRESSO: fcc Al at 300 K")
    print("=" * 68)
    print(f"  Supercell size   : {N_SC} atoms")
    print(f"  r_cutoff         : 4.5 Å  (1st + 2nd NN for fcc Al)")
    print(f"  T_design         : {T_design} K")
    print(f"  T_D (input)      : {T_D_Al} K")
    print(f"  ecutwfc          : {qe_calc.ecutwfc:.0f} Ry"
          f"  ({qe_calc.ecutwfc * 13.6057:.0f} eV)")
    print(f"  ecutrho          : {qe_calc.ecutrho:.0f} Ry")
    print(f"  k-mesh (SC)      : {qe_calc.kpts}")
    print(f"  pw_cmd           : {qe_calc.pw_cmd}")
    print(f"  pseudo_dir       : {qe_calc.pseudo_dir}")
    print()

    runner = SOTCRunner(
        element="Al",
        mass_amu=M_Al,
        prim_cell=prim_cell_al,
        prim_positions=prim_pos_al,
        T=T_design,
        T_D=T_D_Al,
        n_atoms_sc=N_SC,
        force_calculator=qe_calc,
        r_cutoff=4.5,           # Å  — includes 1st (2.86 Å) and 2nd (4.05 Å) NN
        r_max_corr=8.0,
        n_ensemble=5,
        work_dir=WORK_DIR,
        verbosity=1,
        ridge_alpha=1e-3,
        symmetrize_bonds=True,
    )

    results = runner.run(
        n_sc_iterations=4,
        epsilon_conv=0.003,
        mixing=0.40,
        T_values=np.array([50., 100., 150., 200., 250., 300.,
                           350., 400., 500., 600., 700., 800.]),
        q_mesh_cv=(8, 8, 8),
    )

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n" + "─" * 60)
    print(" Al QE result summary")
    print("─" * 60)
    print(f"  Converged          : {results['converged']}")
    print(f"  Iterations         : {results['n_iterations']}")
    print(f"  T_D (spectral)     : {results['T_D_effective']:.1f} K"
          f"  (exp: {T_D_Al:.0f} K)")
    print(f"  T_D (caloric)      : {results['T_D_caloric']:.1f} K")
    print(f"  C_V(300 K)         : {results['C_V_jmolk']:.3f} J/(mol·K)")
    print(f"  ZPE                : {results['ZPE_eV']:.4f} eV")
    print(f"  MSD(300 K)         : {results['MSD_ang2']:.4f} Å²")
    print(f"  Unstable fraction  : {results['unstable_fraction']:.3f}")
    print(f"  pw.x calls         : {qe_calc._n_calls}")
    print()
    print(f"  Results saved → {WORK_DIR / 'sotc_results.json'}")

    print("\n  C_V(T):")
    print(f"  {'T [K]':>8}  {'C_V [J/(mol·K)]':>18}")
    print("  " + "-" * 28)
    for T, cv in zip(results["T_values"], results["C_V_scan"]):
        print(f"  {T:8.1f}  {cv:18.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                        help="Skip if sotc_results.json already exists.")
    args = parser.parse_args()
    main(restart=args.restart)
