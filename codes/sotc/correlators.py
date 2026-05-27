"""
Target thermal displacement correlators for SOTC.

Two backends are provided:
  DebyeCorrelator  — analytical Debye model, no phonopy required.
                     Uses the exact quantum formula with the correct
                     prefactor (factor 9, three acoustic branches).
  PhononCorrelator — uses a phonopy Phonon object for realistic
                     phonon dispersions.

Key formula (monatomic solid, harmonic limit):

    C̄₂(R,T) = (ℏ/6NM) Σ_{q,s} [cos(q·R)/ω(q,s)] coth(ℏω(q,s)/2k_BT)

Setting R=0 gives MSD/3.  In the isotropic Debye model:

    ⟨u²⟩_T = (9ℏ²T²)/(2Mk_BT_D³) ∫₀^{T_D/T} x coth(x/2) dx

The factor is 9, not 27.  See derivation:
    g(ω) = 9ω²/ω_D³  (3 branches, normalised so ∫g dω = 3)
    ⟨u²⟩ = (ℏ/2M) ∫₀^{ω_D} g(ω)/ω · coth(ℏω/2k_BT) dω
           = (9ℏ/2Mω_D³) ∫₀^{ω_D} ω coth(ℏω/2k_BT) dω
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import Optional

from .constants import HBAR, KB, coth, AMU_TO_KG, ANG_TO_M


@dataclass
class DebyeCorrelator:
    """
    Debye-model thermal correlators for a monatomic solid.

    Parameters
    ----------
    T_D : float
        Debye temperature [K].
    M_amu : float
        Atomic mass [amu].
    v_D : float, optional
        Debye sound velocity [m/s].  If None, derived from T_D and the
        primitive-cell volume assuming 3 acoustic branches.
    a_latt : float, optional
        Lattice parameter [Å] used to estimate v_D when v_D is None.
    """

    T_D: float
    M_amu: float
    v_D: Optional[float] = None  # m/s
    a_latt: Optional[float] = None  # Å — conventional lattice parameter (kept for back-compat)
    V_per_atom_ang3: Optional[float] = None  # Å³/atom — primitive cell volume (preferred over a_latt)

    def __post_init__(self):
        self._M = self.M_amu * AMU_TO_KG
        self._omega_D = KB * self.T_D / HBAR  # rad/s
        if self.v_D is None:
            if self.V_per_atom_ang3 is not None:
                V_per_atom = self.V_per_atom_ang3 * ANG_TO_M ** 3
                q_D = (6.0 * np.pi ** 2 / V_per_atom) ** (1.0 / 3.0)
                self._v_D = self._omega_D / q_D
            elif self.a_latt is not None:
                # a_latt is the *conventional* cubic lattice parameter [Å].
                # For fcc: V/atom = a³/4; for bcc: a³/2.  Assume fcc as default.
                V_per_atom = (self.a_latt * ANG_TO_M) ** 3 / 4.0
                q_D = (6.0 * np.pi ** 2 / V_per_atom) ** (1.0 / 3.0)
                self._v_D = self._omega_D / q_D
            else:
                self._v_D = 1000.0  # fallback
        else:
            self._v_D = self.v_D

    # ── MSD ───────────────────────────────────────────────────────────────────

    def msd(self, T: float) -> float:
        """
        Mean-square displacement ⟨u²⟩_T [m²] — all three Cartesian components.

        Uses the exact quantum Debye formula:
            ⟨u²⟩ = (9ℏ²T²)/(2Mk_BT_D³) ∫₀^{T_D/T} x coth(x/2) dx

        At T=0: ⟨u²⟩₀ = 9ℏ²/(4Mk_BT_D)  (zero-point motion)
        High-T limit: ⟨u²⟩ → 9k_BT/(Mω_D²)  (classical equipartition)
        """
        if T < 1e-9:
            return 9.0 * HBAR ** 2 / (4.0 * self._M * KB * self.T_D)

        x_D = self.T_D / T

        def integrand(x):
            return x * coth(x / 2.0)

        I, _ = quad(integrand, 0.0, x_D, limit=300, epsabs=1e-12)
        prefactor = 9.0 * HBAR ** 2 * T ** 2 / (2.0 * self._M * KB * self.T_D ** 3)
        return prefactor * I

    def msd_ang2(self, T: float) -> float:
        """Mean-square displacement [Å²]."""
        return self.msd(T) * 1e20

    def rms_displacement_ang(self, T: float) -> float:
        """Root-mean-square displacement √⟨u²⟩ [Å]."""
        return np.sqrt(self.msd_ang2(T))

    # ── Pair correlator ───────────────────────────────────────────────────────

    def pair_correlator(self, R_ang: float, T: float) -> float:
        """
        Scalar trace pair correlator C̄₂(R,T) [m²].

        C̄₂(R,T) = (1/3) Σ_α C^{αα}(R,T)
                 = (3ℏ/2Mω_D³) ∫₀^{ω_D} ω j₀(ωR/v_D) coth(ℏω/2k_BT) dω

        where j₀(x) = sin(x)/x is the zeroth-order spherical Bessel function.

        At R=0: C̄₂(0,T) = ⟨u²⟩/3.
        """
        R = R_ang * ANG_TO_M

        def integrand(omega):
            x = HBAR * omega / (2.0 * KB * T) if T > 1e-9 else np.inf
            coth_val = coth(x)
            if R < 1e-20:
                j0 = 1.0
            else:
                qR = omega * R / self._v_D
                j0 = np.sin(qR) / qR if qR > 1e-9 else 1.0 - qR ** 2 / 6.0
            return omega * j0 * coth_val

        if T < 1e-9:
            # Zero-point: coth → 1
            def integrand_zp(omega):
                if R < 1e-20:
                    return omega
                qR = omega * R / self._v_D
                j0 = np.sin(qR) / qR if qR > 1e-9 else 1.0 - qR ** 2 / 6.0
                return omega * j0

            I, _ = quad(integrand_zp, 0.0, self._omega_D, limit=300, epsabs=1e-14)
        else:
            I, _ = quad(
                integrand,
                1e-8 * self._omega_D,
                self._omega_D,
                limit=500,
                epsabs=1e-14,
            )

        prefactor = 3.0 * HBAR / (2.0 * self._M * self._omega_D ** 3)
        return prefactor * I

    def pair_correlator_ang2(self, R_ang: float, T: float) -> float:
        """Scalar pair correlator [Å²]."""
        return self.pair_correlator(R_ang, T) * 1e20

    def correlator_table(
        self,
        R_values_ang: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """
        Compute C̄₂(R,T) for an array of shell distances [Å].

        Returns
        -------
        C2 : ndarray, shape (len(R_values_ang),)
            Pair correlator in Å².
        """
        return np.array([self.pair_correlator_ang2(R, T) for R in R_values_ang])

    # ── Heat capacity (Debye model) ───────────────────────────────────────────

    def heat_capacity_v(self, T: float) -> float:
        """
        Isochoric heat capacity C_V per atom [J/K] from the Debye model.

        C_V = 9k_B (T/T_D)³ ∫₀^{T_D/T} x⁴ eˣ/(eˣ−1)² dx
        """
        if T < 1e-9:
            return 0.0
        x_D = self.T_D / T

        def integrand(x):
            ex = np.exp(min(x, 700.0))
            return x ** 4 * ex / (ex - 1.0) ** 2

        I, _ = quad(integrand, 0.0, x_D, limit=300, epsabs=1e-12)
        return 9.0 * KB * (T / self.T_D) ** 3 * I

    def heat_capacity_v_jmolk(self, T: float, n_atoms_per_fu: int = 1) -> float:
        """C_V per mole of formula units [J/(mol·K)]."""
        from .constants import NA
        return self.heat_capacity_v(T) * NA * n_atoms_per_fu

    # ── Thermal coherence length ──────────────────────────────────────────────

    def coherence_length(self, T: float, gruneisen: float = 2.0) -> float:
        """
        Thermal coherence length ξ_T [Å].

        ξ_T = ℏ · v_D / (γ · k_B · T)

        This is the thermal de Broglie wavelength of phonons at the dominant
        thermal wavevector k_T = k_B T / (ℏ v_D).  γ is the Grüneisen parameter.
        """
        if T < 1e-9:
            return 1e6  # diverges at T=0
        xi_m = self._v_D * HBAR / (gruneisen * KB * T)
        return xi_m * 1e10  # m → Å


@dataclass
class PhononCorrelator:
    """
    Thermal correlators computed from a phonopy Phonon object.

    Provides the same interface as DebyeCorrelator but uses a realistic
    phonon dispersion on a dense q-mesh.

    Parameters
    ----------
    phonon : phonopy.Phonon
        Fully initialised phonopy object (force constants must be set).
    q_mesh : tuple of int
        Monkhorst-Pack mesh for BZ integration, e.g. (20, 20, 20).
    """

    phonon: object  # phonopy.Phonon; avoid hard import at module level
    q_mesh: tuple = (20, 20, 20)
    _frequencies: np.ndarray = field(default=None, repr=False, init=False)
    _eigenvectors: np.ndarray = field(default=None, repr=False, init=False)
    _weights: np.ndarray = field(default=None, repr=False, init=False)
    _q_points: np.ndarray = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self._init_mesh()

    def _init_mesh(self):
        """Pre-compute frequencies and eigenvectors on the q-mesh."""
        ph = self.phonon
        ph.run_mesh(self.q_mesh, is_eigenvectors=True, with_eigenvectors=True)
        mesh = ph.mesh_dict
        # frequencies in THz → convert to rad/s
        self._frequencies = mesh["frequencies"] * 2.0 * np.pi * 1e12  # rad/s
        # eigenvectors shape: (n_qpoints, 3*n_atoms, 3*n_atoms) — columns are modes
        self._eigenvectors = mesh["eigenvectors"]
        self._weights = mesh["weights"].astype(float)
        self._weights /= self._weights.sum()
        self._q_points = mesh["qpoints"]  # in fractional coords

    def _get_lattice(self):
        """Return real-space lattice vectors [m]."""
        return self.phonon.unitcell.cell * ANG_TO_M  # cell in Å → m

    def msd(self, T: float) -> float:
        """
        ⟨u²⟩_T [m²] summed over all atoms and all branches.
        Uses:
            ⟨u²⟩ = (ℏ/2) Σ_{q,s} [1/(N M_s ω_{qs})] coth(ℏω_{qs}/2k_BT)
        """
        ph = self.phonon
        masses_amu = np.array(ph.unitcell.masses)  # amu
        masses_kg = masses_amu * AMU_TO_KG  # (n_atoms,)
        n_atoms = len(masses_kg)
        freqs = self._frequencies  # (n_q, 3*n_atoms)
        weights = self._weights

        total_msd = 0.0
        for iq, (freq_row, w) in enumerate(zip(freqs, weights)):
            evecs = self._eigenvectors[iq]  # (3*n_atoms, 3*n_atoms)
            for s, omega in enumerate(freq_row):
                if omega < 1e6:  # skip acoustic branches near Γ
                    continue
                x = HBAR * omega / (2.0 * KB * T) if T > 1e-9 else np.inf
                c = coth(x)
                # polarisation vector for mode s: evecs[:, s]
                # e_i^α for atom i: evecs[3i:3i+3, s]
                for i in range(n_atoms):
                    e_i = evecs[3 * i: 3 * i + 3, s]
                    contrib = np.real(np.dot(e_i.conj(), e_i))
                    total_msd += w * contrib * HBAR / (2.0 * masses_kg[i] * omega) * c
        return total_msd

    def msd_ang2(self, T: float) -> float:
        return self.msd(T) * 1e20

    def pair_correlator(self, R_cart_ang: np.ndarray, T: float) -> float:
        """
        C̄₂(R,T) [m²] for lattice vector R_cart_ang [Å] (Cartesian).

        C̄₂(R,T) = (ℏ/6NM) Σ_{q,s} [cos(q·R)/ω_{qs}] coth(ℏω_{qs}/2k_BT)
        (monatomic; for polyatomic the formula generalises per atom pair)
        """
        ph = self.phonon
        masses_amu = np.array(ph.unitcell.masses)
        masses_kg = masses_amu * AMU_TO_KG
        n_atoms = len(masses_kg)
        R_m = np.asarray(R_cart_ang) * ANG_TO_M

        # Get reciprocal lattice for q-point conversion
        cell_m = ph.unitcell.cell * ANG_TO_M  # (3,3) rows = lattice vectors
        rec_lat = 2.0 * np.pi * np.linalg.inv(cell_m).T  # rows = rec. vectors

        freqs = self._frequencies
        weights = self._weights
        q_frac = self._q_points

        total = 0.0
        for iq, (freq_row, w) in enumerate(zip(freqs, weights)):
            q_cart = q_frac[iq] @ rec_lat  # (3,) in rad/m
            phase = np.cos(np.dot(q_cart, R_m))
            evecs = self._eigenvectors[iq]
            for s, omega in enumerate(freq_row):
                if omega < 1e6:
                    continue
                x = HBAR * omega / (2.0 * KB * T) if T > 1e-9 else np.inf
                c = coth(x)
                for i in range(n_atoms):
                    e_i = evecs[3 * i: 3 * i + 3, s]
                    contrib = np.real(np.dot(e_i.conj(), e_i))
                    total += w * phase * contrib * HBAR / (6.0 * masses_kg[i] * omega) * c
        return total

    def pair_correlator_ang2(self, R_cart_ang: np.ndarray, T: float) -> float:
        return self.pair_correlator(R_cart_ang, T) * 1e20

    def heat_capacity_v(self, T: float) -> float:
        """C_V per unit cell [J/K]."""
        if T < 1e-9:
            return 0.0
        freqs = self._frequencies
        weights = self._weights
        cv = 0.0
        for freq_row, w in zip(freqs, weights):
            for omega in freq_row:
                if omega < 1e6:
                    continue
                x = HBAR * omega / (2.0 * KB * T)
                # C_V contribution: k_B (x/sinh(x))² ... standard formula
                sinh_x = np.sinh(min(x, 350.0))
                cv += w * KB * (x / sinh_x) ** 2
        return cv

    def heat_capacity_v_jmolk(self, T: float) -> float:
        from .constants import NA
        return self.heat_capacity_v(T) * NA
