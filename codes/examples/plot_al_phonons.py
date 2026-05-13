#!/usr/bin/env python3
"""
Post-processing: phonon band structure and DOS for fcc Al from existing SQTC run.

Reads force/displacement data from the last (best) iteration of the completed
sqtc_al_fast_vasp_run and rebuilds the IFCExtractor without re-running VASP.
Saves:
  sqtc_al_fast_vasp_run/phonon_bandstructure.npz
  sqtc_al_fast_vasp_run/phonon_dos.npz
  sqtc_al_fast_vasp_run/phonon_bandstructure.pdf
  sqtc_al_fast_vasp_run/phonon_dos.pdf
  sqtc_al_fast_vasp_run/phonon_combined.pdf
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from sqtc import SQTCRunner
from sqtc.ifc_extractor import IFCExtractor
from sqtc.phonons import PhononCalculator

import json

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = Path(__file__).parent.parent.parent / "sqtc_al_fast_vasp_run"

# ── Find selected iteration from results JSON ─────────────────────────────────
results_json = WORK_DIR / "sqtc_results.json"
if results_json.exists():
    with open(results_json) as f:
        saved = json.load(f)
    selected_it = saved.get("selected_iteration", None)
else:
    selected_it = None

# ── Al structure (FCC, a = 4.05 Å, same as run script) ────────────────────────
a_al = 4.05  # Å
prim_cell_al = 0.5 * a_al * np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
prim_pos_al = np.array([[0.0, 0.0, 0.0]])
M_Al_amu = 26.9815385
R_CUTOFF = 4.5   # Å — same as used in the SQTC run
RIDGE_ALPHA = 1e-3

# ── Find the last completed iteration directory ───────────────────────────────
# ── Find the correct iteration directory (0-indexed = selected_iteration - 1) ─
iter_dirs = sorted(WORK_DIR.glob("iter_*"))
if not iter_dirs:
    raise RuntimeError(f"No iter_* directories found in {WORK_DIR}")

if selected_it is not None:
    target_name = f"iter_{selected_it - 1:02d}"
    matching = [d for d in iter_dirs if d.name == target_name]
    best_iter = matching[0] if matching else iter_dirs[-1]
else:
    best_iter = iter_dirs[-1]
print(f"Using iteration: {best_iter.name}  (selected_iteration={selected_it})")

snap_dirs = sorted(best_iter.glob("snap_*"))
if not snap_dirs:
    raise RuntimeError(f"No snap_* directories found in {best_iter}")
print(f"Found {len(snap_dirs)} snapshots")

# ── Parse POSCAR and forces from each snapshot ────────────────────────────────
try:
    from ase.io import read as ase_read
except ImportError:
    raise ImportError("ASE is required: pip install ase")

all_positions = []   # displaced positions for each snap
all_forces = []      # DFT forces for each snap

for snap_dir in snap_dirs:
    poscar = snap_dir / "POSCAR"
    vasprun = snap_dir / "vasprun.xml"
    if not poscar.exists() or not vasprun.exists():
        print(f"  WARNING: skipping {snap_dir.name} — missing POSCAR or vasprun.xml")
        continue

    atoms_poscar = ase_read(str(poscar), format="vasp")
    atoms_vasp   = ase_read(str(vasprun), format="vasp-xml")

    all_positions.append(atoms_poscar.get_positions())   # displaced positions
    all_forces.append(atoms_vasp.get_forces())           # DFT forces [eV/Å]

n_snaps = len(all_positions)
print(f"Successfully read {n_snaps} snapshots")

# ── Reconstruct exact supercell geometry from the runner ─────────────────────
# This avoids error in the mean-position estimate and ensures IFCExtractor's
# pair-grouping works correctly (pairs with same min-image R grouped as one class).
_runner = SQTCRunner(
    element="Al",
    mass_amu=M_Al_amu,
    prim_cell=prim_cell_al,
    prim_positions=prim_pos_al,
    T=300.0,
    T_D=390.0,
    n_atoms_sc=32,
    r_cutoff=R_CUTOFF,
    r_max_corr=8.0,
    work_dir=WORK_DIR,
    verbosity=0,
    ridge_alpha=RIDGE_ALPHA,
    symmetrize_bonds=True,
)
_H, sc_cell, eq_positions, masses = _runner._choose_supercell()
N_atoms = len(eq_positions)
print(f"Supercell: {N_atoms} atoms (reconstructed from runner)")
del _runner

# ── Compute displacements = displaced - equilibrium ───────────────────────────
displacements = [pos - eq_positions for pos in all_positions]

# Equilibrium forces ≈ 0 for a well-relaxed VASP calculation;
# forces in vasprun.xml are F(u), so ΔF = F(u) - F(0) ≈ F(u).
# Subtract the mean as a safe residual correction.
mean_force = np.mean(all_forces, axis=0)
delta_forces = [f - mean_force for f in all_forces]

# ── Build and fit IFCExtractor ────────────────────────────────────────────────
extractor = IFCExtractor(
    supercell_positions=eq_positions,
    supercell_cell=sc_cell,
    masses_amu=masses,
    r_cutoff=R_CUTOFF,
    symmetrise=True,
    ridge_alpha=RIDGE_ALPHA,
    symmetrize_bonds=True,
)
extractor.fit(displacements, delta_forces)
report = extractor.fit_report(displacements, delta_forces)
print(f"IFC fit:  RMSE = {report['rmse_ev_ang']:.4f} eV/Å  "
      f"R² = {report['r2']:.4f}  rank = {report['rank']}")

# ── Build PhononCalculator ────────────────────────────────────────────────────
phonon_calc = PhononCalculator(
    ifc_extractor=extractor,
    prim_positions=prim_pos_al,
    prim_cell=prim_cell_al,
    masses_amu=np.array([M_Al_amu]),
)

# ── Phonon band structure ──────────────────────────────────────────────────────
print("Computing phonon band structure ...")
bs = phonon_calc.compute_band_structure(n_points_per_segment=80)
np.savez(
    WORK_DIR / "phonon_bandstructure.npz",
    distances=bs["distances"],
    frequencies_thz=bs["frequencies_thz"],
    q_points=bs["q_points"],
    labels=np.array(bs["labels"], dtype=str),
    label_positions=bs["label_positions"],
)
print(f"  Path: {' - '.join(bs['labels'])}")
print(f"  Max frequency: {bs['frequencies_thz'].max():.2f} THz")

# ── Phonon DOS ────────────────────────────────────────────────────────────────
print("Computing phonon DOS (30×30×30 mesh) ...")
dos = phonon_calc.compute_dos(q_mesh=(30, 30, 30), n_bins=400, sigma_thz=0.05)
np.savez(
    WORK_DIR / "phonon_dos.npz",
    frequencies_thz=dos["frequencies_thz"],
    dos=dos["dos"],
)

# ── Plot: band structure ───────────────────────────────────────────────────────
fig_bs, ax_bs = plt.subplots(figsize=(7, 5))
dists = bs["distances"]
freqs = bs["frequencies_thz"]
n_branches = freqs.shape[1]

for b in range(n_branches):
    ax_bs.plot(dists, freqs[:, b], color="steelblue", lw=1.2, alpha=0.9)

for lpos in bs["label_positions"]:
    ax_bs.axvline(x=lpos, color="gray", lw=0.7, ls="--")
ax_bs.axhline(y=0, color="gray", lw=0.5, ls="-")

ax_bs.set_xticks(bs["label_positions"])
label_tex = [lbl.replace("G", r"$\Gamma$") for lbl in bs["labels"]]
ax_bs.set_xticklabels(label_tex, fontsize=12)
ax_bs.set_xlim(dists[0], dists[-1])
ax_bs.set_ylabel("Frequency (THz)", fontsize=12)
ax_bs.set_title("Phonon Band Structure — fcc Al (SQTC, 300 K)", fontsize=12)
ax_bs.tick_params(axis="x", length=0)
ax_bs.grid(axis="y", lw=0.3, alpha=0.5)
fig_bs.tight_layout()
fig_bs.savefig(WORK_DIR / "phonon_bandstructure.pdf", dpi=200)
fig_bs.savefig(WORK_DIR / "phonon_bandstructure.png", dpi=200)
print(f"  Saved: {WORK_DIR / 'phonon_bandstructure.pdf'}")

# ── Plot: DOS ──────────────────────────────────────────────────────────────────
fig_dos, ax_dos = plt.subplots(figsize=(5, 5))
ax_dos.fill_betweenx(dos["frequencies_thz"], dos["dos"], alpha=0.35, color="coral")
ax_dos.plot(dos["dos"], dos["frequencies_thz"], color="firebrick", lw=1.5)
ax_dos.axhline(y=0, color="gray", lw=0.5)
ax_dos.set_xlabel("DOS (states/THz/f.u.)", fontsize=12)
ax_dos.set_ylabel("Frequency (THz)", fontsize=12)
ax_dos.set_title("Phonon DOS — fcc Al (SQTC, 300 K)", fontsize=12)
ax_dos.set_ylim(bottom=0)
ax_dos.set_xlim(left=0)
fig_dos.tight_layout()
fig_dos.savefig(WORK_DIR / "phonon_dos.pdf", dpi=200)
fig_dos.savefig(WORK_DIR / "phonon_dos.png", dpi=200)
print(f"  Saved: {WORK_DIR / 'phonon_dos.pdf'}")

# ── Plot: combined (band + DOS side by side) ────────────────────────────────────
fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)

# Band structure
for b in range(n_branches):
    ax1.plot(dists, freqs[:, b], color="steelblue", lw=1.2, alpha=0.9)
for lpos in bs["label_positions"]:
    ax1.axvline(x=lpos, color="gray", lw=0.7, ls="--")
ax1.axhline(y=0, color="gray", lw=0.5)
ax1.set_xticks(bs["label_positions"])
ax1.set_xticklabels(label_tex, fontsize=12)
ax1.set_xlim(dists[0], dists[-1])
ax1.set_ylabel("Frequency (THz)", fontsize=12)
ax1.set_title("fcc Al — SQTC phonons at 300 K", fontsize=12)
ax1.tick_params(axis="x", length=0)
ax1.grid(axis="y", lw=0.3, alpha=0.5)

# DOS (horizontal)
ax2.fill_betweenx(dos["frequencies_thz"], dos["dos"], alpha=0.35, color="coral")
ax2.plot(dos["dos"], dos["frequencies_thz"], color="firebrick", lw=1.5)
ax2.axhline(y=0, color="gray", lw=0.5)
ax2.set_xlabel("DOS", fontsize=11)
ax2.set_xlim(left=0)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis="y", length=0)
ax2.grid(axis="y", lw=0.3, alpha=0.5)

y_max = max(freqs.max(), dos["frequencies_thz"][dos["dos"] > 0.01 * dos["dos"].max()].max()) * 1.05
ax1.set_ylim(bottom=min(-0.5, freqs.min() * 1.1), top=y_max)

fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12, wspace=0.05)
fig.savefig(WORK_DIR / "phonon_combined.pdf", dpi=200)
fig.savefig(WORK_DIR / "phonon_combined.png", dpi=200)
print(f"  Saved: {WORK_DIR / 'phonon_combined.pdf'}")

print("\nDone.")
print(f"Output directory: {WORK_DIR}")
