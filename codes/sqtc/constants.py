"""
Physical constants in SI units.
All internal calculations use SI; conversion helpers are provided for
common materials-science unit systems (eV/Å/amu).
"""

from __future__ import annotations

import numpy as np

# ── Fundamental constants (SI) ────────────────────────────────────────────────
HBAR    = 1.054571817e-34   # J·s
KB      = 1.380649e-23      # J/K
NA      = 6.02214076e23     # mol⁻¹
R_GAS   = KB * NA            # J/(mol·K) = 8.314...
ME      = 9.1093837015e-31  # kg  (electron mass)

# ── Unit conversions ──────────────────────────────────────────────────────────
EV_TO_J   = 1.602176634e-19   # 1 eV  → J
J_TO_EV   = 1.0 / EV_TO_J
AMU_TO_KG = 1.66053906660e-27 # 1 amu → kg
KG_TO_AMU = 1.0 / AMU_TO_KG
ANG_TO_M  = 1.0e-10           # 1 Å   → m
M_TO_ANG  = 1.0e10
THZ_TO_RAD_S = 2.0 * np.pi * 1.0e12  # 1 THz → rad/s
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S

# ── Derived convenience values ────────────────────────────────────────────────
HBAR_EV_S  = HBAR * J_TO_EV       # ℏ in eV·s  (6.5821e-16)
KB_EV_K    = KB   * J_TO_EV       # k_B in eV/K (8.6173e-5)

# Force-constant conversion: 1 eV/Å² → N/m
EV_ANG2_TO_SI = EV_TO_J / ANG_TO_M**2   # ≈ 16.022 N/m

# ── Atomic masses (kg) for common elements ────────────────────────────────────
MASS_H  = 1.00794  * AMU_TO_KG
MASS_D  = 2.01410  * AMU_TO_KG
MASS_HE = 4.002602 * AMU_TO_KG
MASS_NE = 20.1797  * AMU_TO_KG
MASS_AR = 39.948   * AMU_TO_KG

# AMU values (for phonopy/ASE compatibility)
MASS_H_AMU  = 1.00794
MASS_D_AMU  = 2.01410
MASS_HE_AMU = 4.002602

# ── Frequently used thermal scale factors ─────────────────────────────────────
def hbar_omega_over_kBT(omega_rad_s: float, T_K: float) -> float:
    """Dimensionless quantum parameter x = ℏω/(k_B T)."""
    if T_K < 1e-9:
        return np.inf
    return HBAR * omega_rad_s / (KB * T_K)


def coth(x: float | np.ndarray) -> float | np.ndarray:
    """
    Hyperbolic cotangent with safe handling of x → 0.
    coth(x) = 1/tanh(x).  At x→0, coth(x) → 2/x (classical limit).
    """
    x = np.asarray(x, dtype=float)
    result = np.where(np.abs(x) < 1e-6, 2.0 / (x + 1e-300), 1.0 / np.tanh(x))
    return result


def bose_einstein(omega_rad_s: float | np.ndarray, T_K: float) -> float | np.ndarray:
    """Bose–Einstein occupation n_B(ω,T) = 1/(exp(ℏω/k_BT)−1)."""
    omega = np.asarray(omega_rad_s, dtype=float)
    if T_K < 1e-9:
        return np.zeros_like(omega)
    x = HBAR * omega / (KB * T_K)
    return np.where(x > 700, 0.0, 1.0 / (np.exp(np.clip(x, 0, 700)) - 1.0))
