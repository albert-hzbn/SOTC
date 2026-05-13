"""
Mock force calculators for testing SQTC without a VASP licence.

Implements pair-potential force evaluation for:
  - He–He  : Aziz HFD-B potential (Aziz et al. 1979, Mol. Phys. 61, 1487)
              Well depth ε = 0.944 meV,  σ = 2.56 Å
  - H₂–H₂ : Silvera–Goldman isotropic potential (SG78)
              (Silvera & Goldman 1978, J. Chem. Phys. 69, 4209)
  - Generic: Lennard-Jones 12-6

All forces are in eV/Å; the interface mirrors VASP output so the
SQTC runner can substitute mock_forces → VASP transparently.

Usage
-----
    lj = LennardJonesForces(epsilon_ev=0.0944e-3, sigma_ang=2.56, r_cut=8.0)
    forces = lj.compute(positions, cell)  # (N,3) eV/Å
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .constants import EV_TO_J, ANG_TO_M


class LennardJonesForces:
    """
    Lennard-Jones 12-6 pair potential forces.

    V(r) = 4ε [(σ/r)¹² − (σ/r)⁶]

    Parameters
    ----------
    epsilon_ev : float  [eV]    well depth
    sigma_ang  : float  [Å]    zero-crossing distance
    r_cut      : float  [Å]    cutoff radius
    """

    def __init__(
        self,
        epsilon_ev: float,
        sigma_ang: float,
        r_cut: float = 10.0,
    ):
        self.eps = epsilon_ev
        self.sig = sigma_ang
        self.r_cut = r_cut

    # index of (0,0,0) in itertools.product([-1,0,1], repeat=3)
    _CENTER_IMG = 13

    def _build_rvec(self, positions: np.ndarray, cell: np.ndarray):
        """
        Build all pairwise displacement vectors including periodic images.

        Returns
        -------
        R_vec : (N, N, 27, 3)  [Å]
        r     : (N, N, 27)     distances [Å]
        valid : (N, N, 27)     bool mask — excludes self at origin + r outside [r_min, r_cut]
        """
        N = len(positions)
        images = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)  # (27, 3)
        img_shifts = images @ cell  # (27, 3)
        # diff0[i,j] = pos[j] - pos[i]
        diff0 = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 3)
        R_vec = diff0[:, :, np.newaxis, :] + img_shifts[np.newaxis, np.newaxis, :, :]  # (N, N, 27, 3)
        r = np.linalg.norm(R_vec, axis=-1)  # (N, N, 27)
        # Exclude self-interaction at the (0,0,0) image
        self_mask = np.zeros((N, N, 27), dtype=bool)
        self_mask[np.arange(N), np.arange(N), self._CENTER_IMG] = True
        valid = (~self_mask) & (r >= 1e-6) & (r <= self.r_cut)
        return R_vec, r, valid

    def _lj_dV_dr(self, r_safe: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """LJ dV/dr on safe distances (already masked)."""
        sr6 = (self.sig / r_safe) ** 6
        return np.where(valid, 4.0 * self.eps * (-12.0 * sr6 ** 2 + 6.0 * sr6) / r_safe, 0.0)

    def compute(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> np.ndarray:
        """
        Compute forces on all atoms using minimum-image convention.

        Parameters
        ----------
        positions : (N, 3) float [Å]  — Cartesian positions
        cell      : (3, 3) float [Å]  — lattice (rows = vectors)

        Returns
        -------
        forces : (N, 3) float [eV/Å]
        """
        R_vec, r, valid = self._build_rvec(positions, cell)
        r_safe = np.where(valid, r, 1.0)
        dV_dr = self._lj_dV_dr(r_safe, valid)          # (N, N, 27)
        R_hat = R_vec / r_safe[:, :, :, np.newaxis]    # (N, N, 27, 3)
        # forces[i] = Σ_{j,img} dV_dr[i,j,img] * R_hat[i,j,img]
        forces = np.einsum('ijk,ijkl->il', dV_dr, R_hat)
        return forces

    def energy(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> float:
        """Total LJ potential energy [eV]."""
        R_vec, r, valid = self._build_rvec(positions, cell)
        r_safe = np.where(valid, r, 1.0)
        sr6 = np.where(valid, (self.sig / r_safe) ** 6, 0.0)
        E_mat = np.where(valid, 4.0 * self.eps * (sr6 ** 2 - sr6), 0.0)
        # Each pair (i,j,img) and (j,i,-img) are counted twice → divide by 2
        return float(E_mat.sum()) / 2.0


class AzizHFDPotential(LennardJonesForces):
    """
    Aziz HFD-B potential for ⁴He–⁴He interactions.

    Parameters from: Aziz R.A. et al. (1979) Mol. Phys. 61, 1487.
    Also used in: Aziz R.A. & Slaman M.J. (1991) J. Chem. Phys. 94, 8047.

    V(x) = ε { A exp(−αx − βx²) − [C₆/x⁶ + C₈/x⁸ + C₁₀/x¹⁰] F(x) }
    where x = r/r_m, and F(x) is a damping function:
        F(x) = exp(−(D/x − 1)²)  if x < D, else 1.

    For testing purposes, we use the simpler parameterisation from the
    original 1979 paper (HFD-B model).
    """

    # HFD-B parameters (Aziz 1979)
    _A    = 0.5448504e6    # dimensionless
    _alpha = 13.353384     # dimensionless
    _beta  = 0.0           # not used in original HFD-B
    _C6   = 1.3732412      # dimensionless
    _C8   = 0.4253785      # dimensionless
    _C10  = 0.178100       # dimensionless
    _D    = 1.241314       # dimensionless (damping onset)
    _r_m  = 2.9673         # Å  (potential minimum position)
    _eps  = 10.8 * 8.617333e-5  # K → eV  (well depth = 10.8 K)

    def __init__(self, r_cut: float = 12.0):
        # Call parent with approximate LJ params for the r_cut mechanism
        super().__init__(epsilon_ev=self._eps, sigma_ang=2.56, r_cut=r_cut)

    def _V_and_dV(self, r: float) -> Tuple[float, float]:
        """Compute V(r) [eV] and dV/dr [eV/Å] for the HFD-B potential."""
        x = r / self._r_m
        eps = self._eps

        # Repulsion
        V_rep = self._A * np.exp(-self._alpha * x)

        # Dispersion with damping
        if x < self._D:
            F = np.exp(-((self._D / x - 1.0) ** 2))
        else:
            F = 1.0

        V_disp = -(self._C6 / x**6 + self._C8 / x**8 + self._C10 / x**10) * F
        V = eps * (V_rep + V_disp)

        # Derivative (numerical for simplicity)
        dx = 1e-5
        x2 = (r + dx) / self._r_m
        if x2 < self._D:
            F2 = np.exp(-((self._D / x2 - 1.0) ** 2))
        else:
            F2 = 1.0
        V2_rep = self._A * np.exp(-self._alpha * x2)
        V2_disp = -(self._C6 / x2**6 + self._C8 / x2**8 + self._C10 / x2**10) * F2
        V2 = eps * (V2_rep + V2_disp)
        dV_dr = (V2 - V) / dx

        return V, dV_dr

    def _dV_dr_vec(self, r_arr: np.ndarray) -> np.ndarray:
        """Vectorized dV/dr [eV/Å] for an array of distances."""
        x = r_arr / self._r_m
        x2 = (r_arr + 1e-5) / self._r_m
        x_safe  = np.where(x  > 0, x,  1e-30)
        x2_safe = np.where(x2 > 0, x2, 1e-30)
        F  = np.where(x  < self._D, np.exp(-np.clip((self._D / x_safe  - 1.0) ** 2, 0, 500)), 1.0)
        F2 = np.where(x2 < self._D, np.exp(-np.clip((self._D / x2_safe - 1.0) ** 2, 0, 500)), 1.0)
        V_rep  = self._A * np.exp(-self._alpha * x)
        V2_rep = self._A * np.exp(-self._alpha * x2)
        V_disp  = -(self._C6 / x_safe ** 6  + self._C8 / x_safe ** 8  + self._C10 / x_safe ** 10)  * F
        V2_disp = -(self._C6 / x2_safe ** 6 + self._C8 / x2_safe ** 8 + self._C10 / x2_safe ** 10) * F2
        V  = self._eps * (V_rep  + V_disp)
        V2 = self._eps * (V2_rep + V2_disp)
        return (V2 - V) / 1e-5

    def compute(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> np.ndarray:
        """Forces [eV/Å] from the HFD-B potential (vectorised)."""
        R_vec, r, valid = self._build_rvec(positions, cell)
        # HFD-B has a harder core: exclude r < 0.5 Å
        valid = valid & (r >= 0.5)
        r_safe = np.where(valid, r, 1.0)
        dV_dr = np.where(valid, self._dV_dr_vec(r_safe), 0.0)
        R_hat = R_vec / r_safe[:, :, :, np.newaxis]
        return np.einsum('ijk,ijkl->il', dV_dr, R_hat)

    def energy(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> float:
        """Total HFD-B potential energy [eV]."""
        R_vec, r, valid = self._build_rvec(positions, cell)
        valid = valid & (r >= 0.5)
        r_safe = np.where(valid, r, 1.0)
        x = r_safe / self._r_m
        x_safe = np.where(x > 0, x, 1e-30)
        F = np.where(x < self._D, np.exp(-np.clip((self._D / x_safe - 1.0) ** 2, 0, 500)), 1.0)
        V_rep  = self._A * np.exp(-self._alpha * x)
        V_disp = -(self._C6 / x_safe ** 6 + self._C8 / x_safe ** 8 + self._C10 / x_safe ** 10) * F
        E_mat = np.where(valid, self._eps * (V_rep + V_disp), 0.0)
        return float(E_mat.sum()) / 2.0


class SilveraGoldmanPotential(LennardJonesForces):
    """
    Silvera–Goldman isotropic potential for H₂–H₂ (centre-of-mass).

    V(r) = exp(α − β r − γ r²) − (C₆/r⁶ + C₈/r⁸ + C₁₀/r¹⁰) f_c(r)

    Parameters from: Silvera I.F. & Goldman V.V. (1978)
    J. Chem. Phys. 69, 4209.  Also see Silvera (1980) Rev. Mod. Phys. 52, 393.

    Units in the original paper: energies in K (×k_B), distances in bohr.
    Converted here to eV and Å.
    """

    # SG parameters in atomic units (Hartree, bohr) — Silvera & Goldman (1978)
    _alpha    =  1.713    # constant (dimensionless exponent pre-factor)
    _beta_bohr = 1.5671  # bohr⁻¹
    _gamma_bohr2 = 0.00993  # bohr⁻²
    _C6_habohr6  = 12.14    # Ha·bohr⁶
    _C8_habohr8  = 215.2    # Ha·bohr⁸
    _C10_habohr10 = 4813.9  # Ha·bohr¹⁰
    _r_c_bohr   = 8.321    # bohr  (cutoff in damping)

    _BOHR_TO_ANG = 0.529177   # bohr → Å
    _HARTREE_TO_EV = 27.2114  # Ha → eV

    def __init__(self, r_cut: float = 8.0):
        super().__init__(epsilon_ev=5.2e-3, sigma_ang=2.96, r_cut=r_cut)

    def _V_and_dV(self, r_ang: float) -> Tuple[float, float]:
        """V [eV] and dV/dr [eV/Å] at distance r [Å]."""
        r = r_ang / self._BOHR_TO_ANG  # Å → bohr
        Ha2eV = self._HARTREE_TO_EV
        b2a   = self._BOHR_TO_ANG

        exp_rep = np.exp(self._alpha - self._beta_bohr * r - self._gamma_bohr2 * r**2)

        # Damping function
        if r < self._r_c_bohr:
            fc = np.exp(-((self._r_c_bohr / r - 1.0) ** 2))
        else:
            fc = 1.0

        V_disp = -(
            self._C6_habohr6  / r**6  +
            self._C8_habohr8  / r**8  +
            self._C10_habohr10 / r**10
        ) * fc

        V_ha = exp_rep + V_disp
        V_eV = V_ha * Ha2eV

        # Numerical derivative with respect to r [Å]
        dr = 1e-5  # Å
        r2 = (r_ang + dr) / b2a
        exp_rep2 = np.exp(self._alpha - self._beta_bohr * r2 - self._gamma_bohr2 * r2**2)
        if r2 < self._r_c_bohr:
            fc2 = np.exp(-((self._r_c_bohr / r2 - 1.0) ** 2))
        else:
            fc2 = 1.0
        V2_ha = exp_rep2 - (self._C6_habohr6/r2**6 + self._C8_habohr8/r2**8 + self._C10_habohr10/r2**10) * fc2
        dV_dr = (V2_ha - V_ha) * Ha2eV / dr  # eV/Å

        return V_eV, dV_dr

    def _dV_dr_vec(self, r_arr: np.ndarray) -> np.ndarray:
        """Vectorized dV/dr [eV/Å] for an array of distances [Å]."""
        b2a = self._BOHR_TO_ANG
        Ha2eV = self._HARTREE_TO_EV
        dr_ang = 1e-5  # Å

        r_bohr  = r_arr / b2a
        r2_bohr = (r_arr + dr_ang) / b2a

        exp_rep  = np.exp(self._alpha - self._beta_bohr * r_bohr  - self._gamma_bohr2 * r_bohr  ** 2)
        exp_rep2 = np.exp(self._alpha - self._beta_bohr * r2_bohr - self._gamma_bohr2 * r2_bohr ** 2)

        rc = self._r_c_bohr
        r_s  = np.where(r_bohr  > 0, r_bohr,  1e-30)
        r2_s = np.where(r2_bohr > 0, r2_bohr, 1e-30)
        fc  = np.where(r_bohr  < rc, np.exp(-np.clip((rc / r_s  - 1.0) ** 2, 0, 500)), 1.0)
        fc2 = np.where(r2_bohr < rc, np.exp(-np.clip((rc / r2_s - 1.0) ** 2, 0, 500)), 1.0)

        V_disp  = -(self._C6_habohr6 / r_s  ** 6 + self._C8_habohr8 / r_s  ** 8 + self._C10_habohr10 / r_s  ** 10) * fc
        V2_disp = -(self._C6_habohr6 / r2_s ** 6 + self._C8_habohr8 / r2_s ** 8 + self._C10_habohr10 / r2_s ** 10) * fc2

        V_ha  = exp_rep  + V_disp
        V2_ha = exp_rep2 + V2_disp
        return (V2_ha - V_ha) * Ha2eV / dr_ang  # eV/Å

    def compute(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> np.ndarray:
        """Forces [eV/Å] from the Silvera-Goldman potential (vectorised)."""
        R_vec, r, valid = self._build_rvec(positions, cell)
        # SG potential has repulsive core; exclude r < 1.0 Å
        valid = valid & (r >= 1.0)
        r_safe = np.where(valid, r, 1.0)
        dV_dr = np.where(valid, self._dV_dr_vec(r_safe), 0.0)
        R_hat = R_vec / r_safe[:, :, :, np.newaxis]
        return np.einsum('ijk,ijkl->il', dV_dr, R_hat)


class EinsteinBackground:
    """
    Wrap a pair-potential calculator with an isotropic harmonic (Einstein)
    spring that pulls each atom back to its equilibrium site with stiffness
    ``k_spring`` [eV/Å²].

    This stabilizes quantum crystals (e.g. solid He) where the bare classical
    pair potential is mechanically unstable at the experimental lattice
    parameter.  Physically, the spring represents the quantum zero-point
    pressure that expands the lattice beyond the classical equilibrium.

    The equilibrium positions are set lazily on the first ``compute()`` call:
    they are taken to be the first set of positions passed in (which must be
    the undisplaced supercell).  Alternatively, pass ``eq_positions``
    explicitly at construction time.

    Parameters
    ----------
    base_calc : AzizHFDPotential | SilveraGoldmanPotential | LennardJonesForces
        Underlying pair-potential force calculator.
    k_spring : float  [eV/Å²]
        Isotropic spring stiffness per atom.  Use a small positive value
        just sufficient to make the effective IFCs positive.
    eq_positions : (N, 3) ndarray or None
        Equilibrium (undisplaced) supercell positions.  If None, they are
        inferred from the first ``compute()`` call.
    """

    def __init__(self, base_calc, k_spring: float, eq_positions=None):
        self.base = base_calc
        self.k = k_spring
        self.eq_pos = None if eq_positions is None else np.asarray(eq_positions, dtype=float)

    def _get_eq_pos(self, positions: np.ndarray) -> np.ndarray:
        if self.eq_pos is None:
            self.eq_pos = positions.copy()
        return self.eq_pos

    def compute(self, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        forces = self.base.compute(positions, cell)
        eq = self._get_eq_pos(positions)
        displacements = positions - eq
        forces -= self.k * displacements  # F_spring = -k * u
        return forces

    def energy(self, positions: np.ndarray, cell: np.ndarray) -> float:
        E = self.base.energy(positions, cell)
        eq = self._get_eq_pos(positions)
        displacements = positions - eq
        E += 0.5 * self.k * np.sum(displacements ** 2)
        return E


def mock_vasp_results(
    positions: np.ndarray,
    cell: np.ndarray,
    force_calculator,
) -> Dict:
    """
    Produce a VASP-like results dict using a pair-potential calculator.

    Returns
    -------
    dict with 'energy_ev', 'forces_ev_ang', 'stress_kbar', 'converged'.
    """
    forces = force_calculator.compute(positions, cell)
    energy = force_calculator.energy(positions, cell)
    return {
        "energy_ev": energy,
        "forces_ev_ang": forces,
        "stress_kbar": np.zeros((3, 3)),
        "converged": True,
    }
