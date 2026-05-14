#!/usr/bin/env python3
"""
Comparison: Al phonon DOS + thermal properties — VASP vs Quantum ESPRESSO
==========================================================================
Produces a 2×2 figure:
  Top-left  : Phonon DOS (normalised)
  Top-right : Phonon band structure  Γ–X–W–K–Γ–L
  Bottom-left : C_V(T) comparison + Dulong–Petit limit
  Bottom-right: Mean-square displacement <u²>(T)

Usage
-----
    python codes/examples/compare_al_vasp_qe.py

Requires that both runs have been post-processed:
    sqtc_al_fast_vasp_run/postproc/
    sqtc_al_qe_run/postproc/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
VASP_DIR = ROOT / "sqtc_al_fast_vasp_run" / "postproc"
QE_DIR   = ROOT / "sqtc_al_qe_run"   / "postproc"
OUT_PATH = ROOT / "sqtc_al_vasp_vs_qe.pdf"
OUT_PNG  = ROOT / "sqtc_al_vasp_vs_qe.png"

# ── Colours ───────────────────────────────────────────────────────────────────
C_VASP = "#1f77b4"   # blue
C_QE   = "#d62728"   # red
LW = 1.8

# ── Experimental reference data (Al) ─────────────────────────────────────────
# Flubacher et al. (1959) + Ditmars et al. (1985)
EXP_T  = np.array([  50,   100,   150,   200,   250,   300,   400,   600,   800,  933])
EXP_CV = np.array([5.91, 12.99, 17.68, 20.46, 22.02, 23.01, 24.14, 24.81, 25.18, 25.46])

# ── Helper: load thermal npz ──────────────────────────────────────────────────

def load_thermal(pp_dir: Path):
    d = np.load(pp_dir / "thermal_properties.npz")
    return d

def load_dos(pp_dir: Path):
    d = np.load(pp_dir / "phonon_dos.npz")
    freq = d["frequencies_thz"]
    dos  = d.get("dos_total", d.get("dos"))
    return freq, dos

def load_bands(pp_dir: Path):
    """Returns (distances, frequencies, special_x, special_labels) or None."""
    p = pp_dir / "phonon_bandstructure.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    dist = d["distances"]
    # Key may be 'frequencies' or 'frequencies_thz'
    freq = d["frequencies_thz"] if "frequencies_thz" in d else d["frequencies"]
    sx   = d.get("label_positions", d.get("special_x",     d.get("special_points_x",    None)))
    sl   = d.get("labels",          d.get("special_labels", d.get("special_point_labels", None)))
    return dist, freq, sx, sl

# ── Load data ─────────────────────────────────────────────────────────────────
th_v = load_thermal(VASP_DIR)
th_q = load_thermal(QE_DIR)

freq_v, dos_v = load_dos(VASP_DIR)
freq_q, dos_q = load_dos(QE_DIR)

# Normalise DOS so that ∫g(ω)dω = 1 (per atom, 3 modes → normalisation = 3)
dw_v = (freq_v[-1] - freq_v[0]) / (len(freq_v) - 1)
dw_q = (freq_q[-1] - freq_q[0]) / (len(freq_q) - 1)
dos_v_n = dos_v / (dos_v.sum() * dw_v)
dos_q_n = dos_q / (dos_q.sum() * dw_q)

bands_v = load_bands(VASP_DIR)
bands_q = load_bands(QE_DIR)

T_v  = th_v["T_K"]
Cv_v = th_v["Cv_jmolk"]
msd_v = th_v["MSD_ang2"]
ZPE_v = float(th_v["ZPE_eV"])
TD_v  = float(th_v["TD_spectral"])

T_q  = th_q["T_K"]
Cv_q = th_q["Cv_jmolk"]
msd_q = th_q["MSD_ang2"]
ZPE_q = float(th_q["ZPE_eV"])
TD_q  = float(th_q["TD_spectral"])

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Al phonon properties: VASP vs Quantum ESPRESSO (SQTC)", fontsize=13, fontweight="bold")

# ── Top-left: Phonon DOS ──────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(freq_v, dos_v_n, color=C_VASP, lw=LW, label="VASP")
ax.plot(freq_q, dos_q_n, color=C_QE,   lw=LW, label="QE (pw.x)")
ax.set_xlabel("Frequency (THz)", fontsize=11)
ax.set_ylabel("DOS (arb. units)", fontsize=11)
ax.set_title("Phonon DOS", fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
# Vertical markers at ω_max
ax.axvline(freq_v.max(), color=C_VASP, lw=0.8, ls="--", alpha=0.6,
           label=f"$\\omega_{{max}}$ VASP = {freq_v.max():.2f} THz")
ax.axvline(freq_q.max(), color=C_QE,   lw=0.8, ls="--", alpha=0.6,
           label=f"$\\omega_{{max}}$ QE = {freq_q.max():.2f} THz")
ax.legend(fontsize=8.5, loc="upper left")

# ── Top-right: Band structure ─────────────────────────────────────────────────
ax = axes[0, 1]
if bands_q is not None:
    dist_q, bfreq_q, sx_q, sl_q = bands_q
    for branch in bfreq_q.T:
        ax.plot(dist_q, branch, color=C_QE, lw=LW * 0.8, alpha=0.85)
    ax.plot([], [], color=C_QE, lw=LW, label="QE (pw.x)")
if bands_v is not None:
    dist_v, bfreq_v, sx_v, sl_v = bands_v
    for branch in bfreq_v.T:
        ax.plot(dist_v, branch, color=C_VASP, lw=LW * 0.8, alpha=0.7, ls="--")
    ax.plot([], [], color=C_VASP, lw=LW, ls="--", label="VASP")

# High-symmetry labels from whichever dataset has them
_sx = sx_q if (bands_q is not None and sx_q is not None) else (sx_v if bands_v is not None else None)
_sl = sl_q if (bands_q is not None and sl_q is not None) else (sl_v if bands_v is not None else None)
if _sx is not None and _sl is not None:
    for x in _sx:
        ax.axvline(x, color="0.5", lw=0.7)
    ax.set_xticks(list(_sx))
    _labels = [str(l) if not hasattr(l, "item") else l.item() for l in _sl]
    _labels = [l.replace("GAMMA", "Γ").replace("G", "Γ") for l in _labels]
    ax.set_xticklabels(_labels, fontsize=9)

ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_ylabel("Frequency (THz)", fontsize=11)
ax.set_title("Phonon band structure", fontsize=11)
ax.set_ylim(bottom=-0.3)
ax.legend(fontsize=9, loc="upper left")

# ── Bottom-left: C_V(T) ───────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(T_v, Cv_v, color=C_VASP, lw=LW, label="VASP")
ax.plot(T_q, Cv_q, color=C_QE,   lw=LW, label="QE (pw.x)")
ax.scatter(EXP_T, EXP_CV, marker="o", s=30, color="k", zorder=5, label="Experiment")
ax.axhline(3 * 8.314, color="0.55", lw=1.0, ls="--", label="Dulong–Petit (3R)")
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("$C_V$ (J mol⁻¹ K⁻¹)", fontsize=11)
ax.set_title("Heat capacity", fontsize=11)
ax.set_xlim(0, max(T_v.max(), T_q.max()))
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)

# Print table header
print(f"\n{'='*72}")
print(f"  Al  VASP vs QE — thermal properties comparison")
print(f"{'='*72}")
print(f"  ω_max  : VASP {freq_v.max():.3f} THz   QE {freq_q.max():.3f} THz")
print(f"  ZPE    : VASP {ZPE_v*1000:.2f} meV   QE {ZPE_q*1000:.2f} meV")
print(f"  T_D    : VASP {TD_v:.1f} K      QE {TD_q:.1f} K")
print(f"\n  {'T [K]':>7}  {'Cv VASP':>10}  {'Cv QE':>10}  {'ΔCv':>8}  {'Cv exp':>10}")
print(f"  {'-'*55}")
for T_c in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800]:
    cv_v_i = float(np.interp(T_c, T_v, Cv_v))
    cv_q_i = float(np.interp(T_c, T_q, Cv_q))
    if T_v.min() <= T_c <= T_v.max() and T_q.min() <= T_c <= T_q.max():
        cv_e = float(np.interp(T_c, EXP_T, EXP_CV)) if EXP_T.min() <= T_c <= EXP_T.max() else float("nan")
        print(f"  {T_c:7.0f}  {cv_v_i:10.4f}  {cv_q_i:10.4f}  {cv_q_i-cv_v_i:+8.4f}  "
              f"{'---':>10}" if np.isnan(cv_e) else
              f"  {T_c:7.0f}  {cv_v_i:10.4f}  {cv_q_i:10.4f}  {cv_q_i-cv_v_i:+8.4f}  {cv_e:10.4f}")

# ── Bottom-right: MSD(T) ──────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(T_v, msd_v, color=C_VASP, lw=LW, label="VASP")
ax.plot(T_q, msd_q, color=C_QE,   lw=LW, label="QE (pw.x)")
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("$\\langle u^2 \\rangle$ (Å²)", fontsize=11)
ax.set_title("Mean-square displacement", fontsize=11)
ax.set_xlim(0, max(T_v.max(), T_q.max()))
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
fig.savefig(OUT_PNG,  dpi=150, bbox_inches="tight")
print(f"\n  Saved -> {OUT_PATH}")
print(f"  Saved -> {OUT_PNG}")
