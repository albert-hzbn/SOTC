"""
Phonon dispersion, DOS, and thermodynamic properties from SOTC IFCs.

Given the IFC dictionary returned by IFCExtractor.fit(), this module:
  1. Builds the dynamical matrix D(q) on a dense q-mesh.
  2. Diagonalises D(q) to get phonon frequencies ω(q,s).
  3. Computes the phonon DOS g(ω).
  4. Integrates to get C_V(T), C_p(T) [with QHA correction], S(T), F(T).
  5. Reconstructs the thermal displacement correlators from DFT-level IFCs
     for use in the next SOTC self-consistency iteration.

All frequencies are in rad/s internally; displayed in THz and cm⁻¹.
"""

from __future__ import annotations

import concurrent.futures
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

from .constants import (
    HBAR, KB, NA, AMU_TO_KG, EV_TO_J,
    coth, bose_einstein, THZ_TO_RAD_S, RAD_S_TO_THZ,
)


EV_ANG2_TO_SI = EV_TO_J / (1e-10) ** 2  # eV/Å² → N/m = kg/s²


class PhononCalculator:
    """
    Phonon dispersion and thermodynamic properties from fitted IFCs.

    Parameters
    ----------
    ifc_extractor : IFCExtractor
        A fitted IFCExtractor object (after calling .fit()).
    prim_positions : (n_b, 3) float [Å]
        Atomic positions in the primitive cell.
    prim_cell : (3, 3) float [Å]
        Primitive cell lattice (rows = vectors).
    masses_amu : (n_b,) float [amu]
        Atomic masses for atoms in the primitive cell.
    """

    def __init__(
        self,
        ifc_extractor,
        prim_positions: np.ndarray,
        prim_cell: np.ndarray,
        masses_amu: np.ndarray,
        imag_tolerance_thz: float = 0.05,
        born_charges: Optional[np.ndarray] = None,
        dielectric_tensor: Optional[np.ndarray] = None,
    ):
        self.ifc = ifc_extractor
        self.pos_prim = np.asarray(prim_positions)
        self.cell_prim = np.asarray(prim_cell)
        self.masses_amu = np.asarray(masses_amu)
        self.masses_kg = self.masses_amu * AMU_TO_KG
        self.n_b = len(masses_amu)
        self._rec_lat = 2.0 * np.pi * np.linalg.inv(self.cell_prim).T  # (3,3) rad/Å
        self._imag_tol = float(imag_tolerance_thz) * THZ_TO_RAD_S
        # LO-TO correction: (n_b, 3, 3) Born effective charges [e] and (3, 3) ε∞
        self._born    = np.asarray(born_charges)      if born_charges      is not None else None
        self._eps_inf = np.asarray(dielectric_tensor) if dielectric_tensor is not None else None

    def _classify_omegas(self, omegas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (valid_mask, positive_mask, omega_for_thermo)."""
        valid = omegas > -self._imag_tol
        positive = omegas > self._imag_tol
        omega_use = np.where(omegas >= 0.0, omegas, 0.0)
        return valid, positive, omega_use

    def spectrum_statistics(
        self,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> Dict[str, float]:
        """Basic stability diagnostics for the current phonon spectrum."""
        all_omegas = self._all_frequencies(q_mesh).ravel()
        n_modes = all_omegas.size
        unstable = all_omegas < -self._imag_tol
        near_zero = np.abs(all_omegas) <= self._imag_tol
        return {
            "n_modes": int(n_modes),
            "unstable_fraction": float(np.mean(unstable)) if n_modes else 0.0,
            "near_zero_fraction": float(np.mean(near_zero)) if n_modes else 0.0,
            "min_freq_thz": float(np.min(all_omegas) * RAD_S_TO_THZ) if n_modes else 0.0,
            "max_freq_thz": float(np.max(all_omegas) * RAD_S_TO_THZ) if n_modes else 0.0,
        }

    def thermodynamic_summary(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> Dict[str, float]:
        """
        Compute C_V, ZPE, Debye estimate and spectral stability in one pass.
        """
        all_omegas = self._all_frequencies(q_mesh)  # (n_q, n_branches)
        n_q = all_omegas.shape[0]
        omegas_flat = all_omegas.ravel()
        valid, positive, omega_use = self._classify_omegas(omegas_flat)

        if T < 1e-9:
            cv_fu = 0.0
        else:
            x = HBAR * omega_use / (2.0 * KB * T)
            classical = valid & (x < 1e-10)
            quantum = valid & (x >= 1e-10)
            cv = np.zeros_like(x)
            cv[classical] = KB
            x_q = np.clip(x[quantum], 0, 350.0)
            cv[quantum] = KB * (x_q / np.sinh(x_q)) ** 2
            cv_fu = float(cv.sum()) / n_q

        pos_omegas = omegas_flat[positive]
        zpe = float(np.sum(pos_omegas) * HBAR / 2.0 / n_q / EV_TO_J)

        # Spectral-moment Debye temperature: T-independent, robust estimator.
        # For a Debye model: <ω²> = 3/5 ω_D², so T_D = ℏ√(5/3 <ω²>_pos) / k_B.
        t_debye_spectral = self._spectral_moment_debye_temperature(q_mesh=q_mesh)

        # Calorimetric Debye temperature: meaningful only when T ≈ T_D.
        # When T >> T_D, C_V is near 3R and the inversion is ill-conditioned.
        t_debye_caloric = self._calorimetric_debye_temperature(cv_fu * NA, T)

        stats = self.spectrum_statistics(q_mesh=q_mesh)
        stats.update(
            {
                "cv_fu_jk": cv_fu,
                "cv_jmolk": cv_fu * NA,
                "zpe_ev": zpe,
                "t_debye_k": t_debye_spectral,        # spectral-moment (robust)
                "t_debye_caloric_k": t_debye_caloric,  # calorimetric (T ≈ T_D only)
            }
        )
        return stats

    def _spectral_moment_debye_temperature(
        self, q_mesh: Tuple[int, int, int] = (20, 20, 20)
    ) -> float:
        """
        Compute the second-moment (rms) Debye temperature from the phonon DOS.

        For a Debye model: <ω²> = 3/5 × ω_D²
        So: ω_D = sqrt(5/3 × <ω²>_positive_modes)
            T_D  = ℏ ω_D / k_B

        This estimator is temperature-independent and well-defined for any
        phonon spectrum, including cases where T >> T_D or T << T_D.
        Imaginary modes are excluded from the average.
        """
        all_omegas = self._all_frequencies(q_mesh)
        omegas_flat = all_omegas.ravel()
        pos = omegas_flat > 0.0
        if not np.any(pos):
            return 0.0
        omega2_mean = float(np.mean(omegas_flat[pos] ** 2))
        omega_D = np.sqrt(5.0 / 3.0 * omega2_mean)
        return float(HBAR * omega_D / KB)

    def _calorimetric_debye_temperature(self, cv_jmolk: float, T: float) -> float:
        """
        Invert the Debye C_V integral to find T_D such that the Debye model
        reproduces cv_jmolk at temperature T.

        Uses the standard Debye integral for n_b atoms per primitive cell:
            C_V^Debye = 9 n_b N_A k_B (T/T_D)³ ∫₀^{T_D/T} x⁴eˣ/(eˣ-1)² dx

        Classical limit: C_V → 3 n_b N_A k_B = Dulong-Petit per formula unit.
        The bisection search is bounded between 50 K and 2000 K.

        Note: this estimator is ill-conditioned when T >> T_D (C_V near 3R).
        Use _spectral_moment_debye_temperature for a robust T-independent estimate.
        """
        from scipy.integrate import quad

        def debye_cv(T_D):
            if T_D < 1e-6:
                return 3.0 * self.n_b * NA * KB
            x_D = T_D / T
            def integrand(x):
                if x > 500:
                    return 0.0
                ex = np.exp(x)
                return x**4 * ex / (ex - 1.0)**2
            integral, _ = quad(integrand, 0.0, x_D, limit=200)
            return 9.0 * self.n_b * NA * KB * (T / T_D)**3 * integral

        # Classical limit: C_V → 3 n_b R per mole of formula units
        cv_classical = 3.0 * self.n_b * NA * KB
        # If C_V is ≥ 99% of classical, T_D << T — return small value
        if cv_jmolk >= 0.99 * cv_classical:
            return 10.0

        # Bisection between 50 K and 2000 K
        # debye_cv is a decreasing function of T_D:
        #   large T_D → more quantum → lower C_V
        # So: debye_cv(mid) > cv_jmolk means T_D too small → move lo up
        lo, hi = 50.0, 2000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if debye_cv(mid) > cv_jmolk:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ── Dynamical matrix ──────────────────────────────────────────────────────

    def dynamical_matrix(self, q_frac: np.ndarray) -> np.ndarray:
        """
        Build dynamical matrix D(q) from IFCs.

        D^{αβ}_{ij}(q) = (1/√(M_i M_j)) Σ_R Φ_{ij}^{αβ}(R) exp(iq·R)

        Parameters
        ----------
        q_frac : (3,) float  — q-point in fractional coordinates of rec. cell

        Returns
        -------
        D : (3*n_b, 3*n_b) complex  [eV/Å²/amu → after conversion: rad²/s²]
        """
        q_cart = q_frac @ self._rec_lat  # [rad/Å]
        N_sc = self.ifc.N
        D = np.zeros((3 * self.n_b, 3 * self.n_b), dtype=complex)

        # D(q)_{ab}^{αβ} = (1/√(M_a M_b)) Σ_{j: j%n_b=b} Φ_{a,j}^{αβ} e^{iq·R_{a,j}}
        # Sum only over pairs where i is a primitive-cell atom (i < n_b).
        # Atoms 0..n_b-1 are the reference primitive cell; all their images are j.
        pref_conv = EV_TO_J / ((1e-10) ** 2 * AMU_TO_KG)  # eV/Å²/amu → rad²/s²

        for (i, j), Phi_ij in self.ifc._Phi.items():
            if i >= self.n_b:   # only sum from primitive-cell reference atoms
                continue
            i_prim = i          # i < n_b by construction
            j_prim = j % self.n_b

            # Lattice vector from primitive-cell atom i to supercell image j
            if i <= j and (i, j) in self.ifc._pair_vecs:
                R_vec = self.ifc._pair_vecs[(i, j)]
            elif i > j and (j, i) in self.ifc._pair_vecs:
                R_vec = -self.ifc._pair_vecs[(j, i)]
            else:
                R_vec = np.zeros(3)

            phase = np.exp(1j * np.dot(q_cart, R_vec))
            prefactor = pref_conv / np.sqrt(
                self.masses_amu[i_prim] * self.masses_amu[j_prim]
            )

            D[3*i_prim:3*i_prim+3, 3*j_prim:3*j_prim+3] += (
                prefactor * phase * Phi_ij
            )

        # Non-analytic correction for LO-TO splitting in polar materials
        if self._born is not None and np.linalg.norm(q_cart) > 1e-10:
            D += self._nac_term(q_cart)

        # Enforce Hermitian symmetry
        D = 0.5 * (D + D.conj().T)
        return D

    def _nac_term(self, q_cart: np.ndarray) -> np.ndarray:
        """
        Cochran-Cowley non-analytic correction to D(q) for LO-TO splitting.

        ΔD_{(iα)(jβ)}(q̂) = [4π·e²/(4πε₀·Ω₀)] × (Z*_i·q̂)_α (Z*_j·q̂)_β
                             / (q̂·ε∞·q̂) × pref_conv / √(M_i·M_j)

        Only applied when |q| > 0. q→0 limit is direction-dependent (LO-TO
        splitting); q=0 gives the TO frequency (no NAC).

        Requires self._born  : (n_b, 3, 3) Born effective charge tensors [e]
                 self._eps_inf: (3, 3) high-frequency dielectric tensor [dimensionless]
        """
        q_hat = q_cart / np.linalg.norm(q_cart)        # dimensionless unit vector
        vol_ang3 = abs(np.linalg.det(self.cell_prim))  # primitive cell volume [Å³]
        # e²/(4πε₀) = 14.3996 eV·Å  →  4π × (e²/4πε₀) / Ω₀  [eV/Å²]
        COULOMB_EVA = 14.3996
        C = 4.0 * np.pi * COULOMB_EVA / vol_ang3
        eps_q = float(q_hat @ self._eps_inf @ q_hat)
        # pref_conv: eV/Å²/amu → rad²/s²  (same as in dynamical_matrix)
        pref_conv = EV_TO_J / ((1e-10) ** 2 * AMU_TO_KG)
        D_nac = np.zeros((3 * self.n_b, 3 * self.n_b), dtype=complex)
        for i in range(self.n_b):
            Zq_i = self._born[i] @ q_hat
            for j in range(self.n_b):
                Zq_j = self._born[j] @ q_hat
                coeff = (C * pref_conv
                         / (eps_q * np.sqrt(self.masses_amu[i] * self.masses_amu[j])))
                D_nac[3*i:3*i+3, 3*j:3*j+3] += coeff * np.outer(Zq_i, Zq_j)
        return D_nac

    # ── Phonon frequencies at a single q ─────────────────────────────────────

    def frequencies_at_q(
        self,
        q_frac: np.ndarray,
        return_eigvecs: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Phonon frequencies [rad/s] at q_frac.

        Returns ω (and optionally eigenvectors as columns).
        Imaginary frequencies (unstable) are returned as negative values.
        """
        D = self.dynamical_matrix(q_frac)
        w2, vecs = eigh(D)  # real eigenvalues for Hermitian D

        # Convert eigenvalues ω² → ω (sign-safe, avoids sqrt-of-negative warning)
        abs_omega = np.sqrt(np.abs(w2))
        omega = np.where(w2 >= 0, abs_omega, -abs_omega)
        if return_eigvecs:
            return omega, vecs
        return omega

    def frequencies_thz_at_q(self, q_frac: np.ndarray) -> np.ndarray:
        """Phonon frequencies [THz] at q_frac."""
        return self.frequencies_at_q(q_frac) * RAD_S_TO_THZ

    def frequencies_cm1_at_q(self, q_frac: np.ndarray) -> np.ndarray:
        """Phonon frequencies [cm⁻¹] at q_frac."""
        return self.frequencies_thz_at_q(q_frac) / 0.02998  # THz → cm⁻¹

    # ── Parallel q-mesh frequency table ──────────────────────────────────────

    def _all_frequencies(
        self,
        q_mesh: Tuple[int, int, int],
        return_eigvecs: bool = False,
    ) -> np.ndarray:
        """
        Compute phonon frequencies at every q-point on a uniform mesh.

        Uses ThreadPoolExecutor so scipy.linalg.eigh (LAPACK) runs in
        parallel across cores (it releases the GIL).

        Returns
        -------
        all_omegas : (n_q, 3*n_b) float [rad/s]
            Frequencies at each q-point.  Imaginary → negative values.
        all_evecs  : (n_q, 3*n_b, 3*n_b) complex  (only when return_eigvecs=True)
        """
        q1, q2, q3 = q_mesh
        q_points = [
            np.array([i / q1, j / q2, k / q3])
            for i in range(q1) for j in range(q2) for k in range(q3)
        ]

        def _freq(q):
            return self.frequencies_at_q(q, return_eigvecs=return_eigvecs)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_freq, q_points))

        if return_eigvecs:
            all_omegas = np.array([r[0] for r in results])
            all_evecs  = np.array([r[1] for r in results])
            return all_omegas, all_evecs, q_points
        return np.array(results)  # (n_q, n_branches)

    # ── Dispersion along a path ───────────────────────────────────────────────

    def band_structure(
        self,
        q_path: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phonon dispersion along a q-path.

        Parameters
        ----------
        q_path : (n_q, 3) float — q-points in fractional coordinates

        Returns
        -------
        distances : (n_q,) float — cumulative path distance [2π/Å]
        frequencies_thz : (n_q, 3*n_b) float — phonon frequencies [THz]
        """
        freqs = []
        dists = [0.0]
        rec = self._rec_lat

        for iq, q in enumerate(q_path):
            freqs.append(self.frequencies_thz_at_q(q))
            if iq > 0:
                dq = (q - q_path[iq-1]) @ rec
                dists.append(dists[-1] + np.linalg.norm(dq))

        return np.array(dists), np.array(freqs)

    # ── High-symmetry path auto-detection ────────────────────────────────────

    def _auto_high_symmetry_path(
        self, n_points_per_segment: int = 50
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Automatically determine a standard high-symmetry Brillouin-zone path
        based on primitive cell geometry.

        Detects: BCC, FCC, simple-cubic, HCP, tetragonal, and defaults to a
        generic path for other lattices.

        Returns
        -------
        q_path : (n_q, 3) float  — q-points in fractional coordinates
        labels : list of str     — high-symmetry point labels (length = n_segments+1)
        tick_positions : (n_segments+1,) float — cumulative distances at labels
        """
        cell = self.cell_prim
        a_vecs = cell  # rows are lattice vectors
        lengths = np.array([np.linalg.norm(a_vecs[i]) for i in range(3)])
        # Angles between lattice vectors (degrees)
        def _angle(u, v):
            c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return np.degrees(np.arccos(np.clip(c, -1, 1)))
        a12 = _angle(a_vecs[0], a_vecs[1])
        a13 = _angle(a_vecs[0], a_vecs[2])
        a23 = _angle(a_vecs[1], a_vecs[2])
        all_equal_length = (np.max(lengths) - np.min(lengths)) / np.mean(lengths) < 0.02

        # BCC primitive: |a|=|b|=|c|, angles ≈ 109.47° (arccos(-1/3))
        if all_equal_length and all(abs(a - 109.47) < 2.0 for a in [a12, a13, a23]):
            # BCC: Γ-H-N-Γ-P-H
            special = {
                "G": np.array([0.0, 0.0, 0.0]),
                "H": np.array([0.5, -0.5, 0.5]),
                "N": np.array([0.0, 0.0, 0.5]),
                "P": np.array([0.25, 0.25, 0.25]),
            }
            path_keys = [("G", "H"), ("H", "N"), ("N", "G"), ("G", "P"), ("P", "H")]

        # FCC primitive: |a|=|b|=|c|, angles ≈ 60°
        elif all_equal_length and all(abs(a - 60.0) < 2.0 for a in [a12, a13, a23]):
            # FCC: Γ-X-W-K-Γ-L
            special = {
                "G": np.array([0.0, 0.0, 0.0]),
                "X": np.array([0.5, 0.0, 0.5]),
                "W": np.array([0.5, 0.25, 0.75]),
                "K": np.array([0.375, 0.375, 0.75]),
                "L": np.array([0.5, 0.5, 0.5]),
            }
            path_keys = [("G", "X"), ("X", "W"), ("W", "K"), ("K", "G"), ("G", "L")]

        # HCP: two equal short vectors + one orthogonal long vector, a=b≠c, angle≈120°
        elif (abs(lengths[0] - lengths[1]) / np.mean(lengths[:2]) < 0.02
              and abs(a12 - 120.0) < 2.0
              and abs(a13 - 90.0) < 2.0 and abs(a23 - 90.0) < 2.0):
            # HCP: Γ-M-K-Γ-A-L-H-A
            special = {
                "G": np.array([0.0, 0.0, 0.0]),
                "M": np.array([0.5, 0.0, 0.0]),
                "K": np.array([1/3, 1/3, 0.0]),
                "A": np.array([0.0, 0.0, 0.5]),
                "L": np.array([0.5, 0.0, 0.5]),
                "H": np.array([1/3, 1/3, 0.5]),
            }
            path_keys = [("G", "M"), ("M", "K"), ("K", "G"), ("G", "A"),
                         ("A", "L"), ("L", "H"), ("H", "A")]

        # Simple cubic: three equal orthogonal vectors
        elif (all_equal_length
              and abs(a12 - 90.0) < 2.0 and abs(a13 - 90.0) < 2.0 and abs(a23 - 90.0) < 2.0):
            # SC: Γ-X-M-Γ-R-X
            special = {
                "G": np.array([0.0, 0.0, 0.0]),
                "X": np.array([0.0, 0.5, 0.0]),
                "M": np.array([0.5, 0.5, 0.0]),
                "R": np.array([0.5, 0.5, 0.5]),
            }
            path_keys = [("G", "X"), ("X", "M"), ("M", "G"), ("G", "R"), ("R", "X")]

        else:
            # Generic fallback: Γ-X-M-Γ-Z along reciprocal axes
            special = {
                "G": np.array([0.0, 0.0, 0.0]),
                "X": np.array([0.5, 0.0, 0.0]),
                "M": np.array([0.5, 0.5, 0.0]),
                "Z": np.array([0.0, 0.0, 0.5]),
            }
            path_keys = [("G", "X"), ("X", "M"), ("M", "G"), ("G", "Z")]

        # Build q_path by interpolating between special points
        q_path_list = []
        labels = []
        label_indices = []
        for seg_i, (k0, k1) in enumerate(path_keys):
            q0 = special[k0]
            q1 = special[k1]
            n_seg = n_points_per_segment
            segment = np.array([q0 + t * (q1 - q0) for t in np.linspace(0, 1, n_seg, endpoint=False)])
            if seg_i == 0:
                labels.append(k0)
                label_indices.append(0)
            label_indices.append(len(q_path_list) + n_seg)
            labels.append(k1)
            q_path_list.extend(segment.tolist())

        # Add the final endpoint
        q_path_list.append(special[path_keys[-1][1]].tolist())
        q_path = np.array(q_path_list)

        # Compute cumulative distances for tick positions
        rec = self._rec_lat
        dists = [0.0]
        for iq in range(1, len(q_path)):
            dq = (q_path[iq] - q_path[iq - 1]) @ rec
            dists.append(dists[-1] + np.linalg.norm(dq))
        dists = np.array(dists)
        tick_positions = dists[label_indices]

        return q_path, labels, tick_positions

    def compute_band_structure(
        self,
        n_points_per_segment: int = 50,
        q_path: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute phonon band structure along a high-symmetry path.

        If q_path is None, automatically determines the standard path for the
        crystal system via _auto_high_symmetry_path().

        Returns
        -------
        dict with keys:
            'distances'       : (n_q,) float  [2π/Å] — cumulative path distance
            'frequencies_thz' : (n_q, n_branches) float [THz]
            'q_points'        : (n_q, 3) float — fractional coordinates
            'labels'          : list of str — high-symmetry point labels
            'label_positions' : array of float [2π/Å] — positions of the labels
        """
        if q_path is None:
            q_path, labels, label_positions = self._auto_high_symmetry_path(n_points_per_segment)
        else:
            labels = []
            label_positions = np.array([])

        distances, frequencies_thz = self.band_structure(q_path)

        return {
            "distances": distances,
            "frequencies_thz": frequencies_thz,
            "q_points": q_path,
            "labels": labels,
            "label_positions": label_positions,
        }

    def compute_dos(
        self,
        q_mesh: Tuple[int, int, int] = (30, 30, 30),
        n_bins: int = 300,
        sigma_thz: float = 0.05,
    ) -> Dict:
        """
        Compute phonon density of states on a dense q-mesh.

        Returns
        -------
        dict with keys:
            'frequencies_thz' : (n_bins,) float [THz]
            'dos'             : (n_bins,) float [states/THz/formula unit]
        """
        return self.dos(q_mesh=q_mesh, n_bins=n_bins, sigma_thz=sigma_thz)

    def compute_partial_dos(
        self,
        q_mesh: Tuple[int, int, int] = (30, 30, 30),
        n_bins: int = 300,
        sigma_thz: float = 0.05,
    ) -> Dict:
        """
        Compute atom-projected (partial) phonon DOS using phonon eigenvectors.

        For each basis atom α, the partial DOS is:
            g_α(ω) = (1/N_q) Σ_{q,s} P_{α,s}(q) · δ(ω − ω_{q,s})
        where P_{α,s}(q) = Σ_{xyz} |e_{α,xyz,s}(q)|² is the atom α weight
        in eigenvector s at q-point q.  Eigenvectors are normalised so that
        Σ_α P_{α,s}(q) = 1 for every (q,s), hence Σ_α g_α = g_total.

        Returns
        -------
        dict with keys:
            'frequencies_thz'  : (n_bins,) float [THz]
            'dos_total'        : (n_bins,) float [states/THz/f.u.]
            'dos_partial'      : (n_b, n_bins) float — one row per basis atom
        """
        n_q_tot = q_mesh[0] * q_mesh[1] * q_mesh[2]

        # get_eigvecs returns (all_omegas, all_evecs, q_points)
        all_omegas, all_evecs, _ = self._all_frequencies(q_mesh, return_eigvecs=True)
        # all_omegas : (n_q, 3*n_b)
        # all_evecs  : (n_q, 3*n_b, 3*n_b)  — columns are eigenvectors

        all_freqs_thz = all_omegas * RAD_S_TO_THZ   # (n_q, n_branches)
        n_branches = all_freqs_thz.shape[1]

        # Determine frequency axis from positive modes only
        all_freqs_pos = all_freqs_thz[all_freqs_thz > 1e-6]
        if len(all_freqs_pos) == 0:
            freq_bins = np.linspace(0, 1, n_bins)
            return {
                "frequencies_thz": freq_bins,
                "dos_total": np.zeros(n_bins),
                "dos_partial": np.zeros((self.n_b, n_bins)),
            }
        omega_max = all_freqs_pos.max() * 1.1
        freq_bins = np.linspace(0, omega_max, n_bins)

        dos_partial = np.zeros((self.n_b, n_bins))

        for alpha in range(self.n_b):
            # Atom alpha contributes components 3α, 3α+1, 3α+2 in the eigenvector
            # all_evecs[:, 3α:3α+3, s] is the polarisation of atom α in branch s
            # P_alpha shape: (n_q, n_branches)
            P = np.sum(
                np.abs(all_evecs[:, 3*alpha:3*(alpha+1), :]) ** 2,
                axis=1,
            )  # (n_q, n_branches)

            # Keep only modes with positive frequency
            mask_pos = all_freqs_thz > 1e-6  # (n_q, n_branches)

            # Vectorised Gaussian smearing
            freqs_flat = all_freqs_thz[mask_pos]   # (N_pos,)
            weights_flat = P[mask_pos]              # (N_pos,)

            diff = freq_bins[:, np.newaxis] - freqs_flat[np.newaxis, :]  # (n_bins, N_pos)
            g = np.sum(weights_flat[np.newaxis, :] * np.exp(-0.5 * (diff / sigma_thz) ** 2), axis=1)
            dos_partial[alpha] = g / (sigma_thz * np.sqrt(2 * np.pi) * n_q_tot)

        dos_total = dos_partial.sum(axis=0)

        return {
            "frequencies_thz": freq_bins,
            "dos_total": dos_total,
            "dos_partial": dos_partial,
        }

    # ── Phonon DOS ────────────────────────────────────────────────────────────

    def dos(
        self,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
        n_bins: int = 300,
        sigma_thz: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phonon density of states g(ω) on a uniform q-mesh (Monkhorst-Pack).

        Uses Gaussian smearing with width sigma_thz [THz].

        Returns
        -------
        freq_thz : (n_bins,) float
        dos : (n_bins,) float  [states per THz per formula unit]
        """
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        all_omegas = self._all_frequencies(q_mesh)            # (n_q, n_branches)
        all_freqs = all_omegas.ravel() * RAD_S_TO_THZ
        all_freqs_pos = all_freqs[all_freqs > 1e-6]

        omega_max = all_freqs_pos.max() * 1.1
        freq_bins = np.linspace(0, omega_max, n_bins)

        # Vectorised Gaussian smearing: (n_bins, n_freqs)
        diff = freq_bins[:, np.newaxis] - all_freqs_pos[np.newaxis, :]
        dos_vals = np.sum(np.exp(-0.5 * (diff / sigma_thz) ** 2), axis=1)
        dos_vals /= (sigma_thz * np.sqrt(2 * np.pi) * n_q)

        return {"frequencies_thz": freq_bins, "dos": dos_vals}

    # ── Thermodynamic properties ──────────────────────────────────────────────

    def heat_capacity_v(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """
        Isochoric heat capacity C_V per formula unit [J/K].

        C_V = k_B Σ_{q,s} (ℏω_{qs}/2k_BT)² / sinh²(ℏω_{qs}/2k_BT)
        """
        if T < 1e-9:
            return 0.0
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        all_omegas = self._all_frequencies(q_mesh).ravel()   # (n_q * n_branches,)

        # Skip genuinely unstable modes; keep near-zero modes in the
        # classical limit for numerical robustness.
        valid, _, omega_use = self._classify_omegas(all_omegas)
        x = HBAR * omega_use / (2.0 * KB * T)
        # classical limit when x → 0: (x/sinh x)² → 1
        classical = valid & (x < 1e-10)
        quantum   = valid & (x >= 1e-10)
        cv = np.zeros_like(x)
        cv[classical] = KB
        x_q = np.clip(x[quantum], 0, 350.0)
        cv[quantum] = KB * (x_q / np.sinh(x_q)) ** 2
        return float(cv.sum()) / n_q

    def heat_capacity_v_jmolk(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """C_V per mole [J/(mol·K)]."""
        return self.heat_capacity_v(T, q_mesh) * NA

    def heat_capacity_scan(
        self,
        T_values: np.ndarray,
        q_mesh: Tuple[int, int, int] = (10, 10, 10),
    ) -> np.ndarray:
        """
        C_V [J/(mol·K)] at each temperature in T_values.

        Uses a precomputed frequency mesh for efficiency.
        """
        all_omegas = self._all_frequencies(q_mesh)        # (n_q, n_branches)
        n_q = len(all_omegas)   # number of q-points
        omegas_flat = all_omegas.ravel()                  # (n_q * n_branches,)

        # Exclude strongly imaginary modes (same threshold as heat_capacity_v).
        # Small imaginary / zero modes (|ω| < imag_tol) are treated as
        # classical (x→0 → contribution = KB); large imaginary modes are
        # genuinely unstable and excluded (contribution = 0).
        valid, _, omega_use = self._classify_omegas(omegas_flat)
        cv_vals = np.zeros(len(T_values))

        for it, T in enumerate(T_values):
            if T < 1e-9:
                continue
            x = HBAR * omega_use / (2.0 * KB * T)
            x_c = np.clip(x, 0, 350.0)
            sinh_x = np.sinh(x_c)
            # Classical limit when x → 0: (x/sinh x)² → 1
            # Use safe denominator to avoid 0/0 warning (masked out by np.where anyway)
            sinh_safe = np.where(x < 1e-10, np.ones_like(sinh_x), sinh_x)
            cv_per_mode = np.where(x < 1e-10, 1.0, (x / sinh_safe) ** 2)
            # Zero out excluded (strongly imaginary) modes — matches heat_capacity_v()
            cv_per_mode = np.where(valid, cv_per_mode, 0.0)
            # Sum over all branches at all q-points, normalise by n_q
            cv_vals[it] = KB * NA * np.sum(cv_per_mode) / n_q

        return cv_vals

    def vibrational_entropy(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """
        Vibrational entropy S_vib per formula unit [J/K].

        S = k_B Σ_{q,s} [(x/tanh(x)) - ln(2 sinh(x))]
        where x = ℏω/(2k_BT).
        """
        if T < 1e-9:
            return 0.0
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        all_omegas = self._all_frequencies(q_mesh).ravel()
        omegas = all_omegas[all_omegas > self._imag_tol]
        x = HBAR * omegas / (2.0 * KB * T)
        x_c = np.clip(x, 0, 350.0)
        S = KB * np.sum(x / np.tanh(x_c) - np.log(2.0 * np.sinh(x_c))) / n_q
        return float(S)

    def zero_point_energy(
        self,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """
        Zero-point energy per formula unit [eV].
        ZPE = Σ_{q,s} ℏω_{qs}/2
        """
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        all_omegas = self._all_frequencies(q_mesh).ravel()
        zpe = np.sum(all_omegas[all_omegas > self._imag_tol]) * HBAR / 2.0 / n_q / EV_TO_J
        return float(zpe)

    def debye_temperature_from_dos(
        self,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """
        Effective Debye temperature estimated from the acoustic branches:
        T_D = ℏω_D / k_B  where ω_D is the acoustic cutoff.
        """
        all_omegas = self._all_frequencies(q_mesh)   # (n_q, n_branches)
        max_acoustic = 0.0
        for omegas in all_omegas:
            pos_omegas = omegas[omegas > self._imag_tol]
            if len(pos_omegas) >= 3:
                max_acoustic = max(max_acoustic, np.sort(pos_omegas)[:3].max())
        return HBAR * max_acoustic / KB  # rad/s → K

    def vibrational_free_energy(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> float:
        """
        Vibrational (Helmholtz) free energy F_vib per formula unit [eV].

        F_vib(T) = Σ_{q,s} k_B T ln[ 2 sinh(ħω_{qs}/2k_BT) ]
                 = ZPE + k_B T Σ_{q,s} ln[1 - exp(-ħω_{qs}/k_BT)]

        At T=0 this reduces to ZPE = Σ ħω/2.
        Imaginary/acoustic modes (|ω| ≤ imag_tol) are excluded.

        Used by SOTCQuasiHarmonic to minimise G(T,P,V) = E_0(V) + F_vib(T,V)
        over volume, giving thermal expansion without extra DFT beyond the
        SOTC force evaluations themselves.
        """
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        all_omegas = self._all_frequencies(q_mesh).ravel()
        omegas = all_omegas[all_omegas > self._imag_tol]
        if T < 1e-9:
            # Pure ZPE limit
            return float(np.sum(omegas) * HBAR / 2.0 / n_q / EV_TO_J)
        x = np.clip(HBAR * omegas / (2.0 * KB * T), 0, 350.0)
        F = KB * T * np.sum(np.log(2.0 * np.sinh(x))) / n_q
        return float(F / EV_TO_J)

    def mode_gruneisen_parameters(
        self,
        calc_vp: "PhononCalculator",
        calc_vm: "PhononCalculator",
        dV_over_V: float,
    ) -> np.ndarray:
        """
        Mode Grüneisen parameters γ_{qs} from finite differences of phonon
        frequencies at two volumes V+ and V- (same q-mesh as self).

        γ_{qs} = -(V/ω) ∂ω/∂V ≈ -(1/2δ) [ω(V+)/ω₀ - ω(V-)/ω₀] / (δV/V)

        where δV/V = dV_over_V is the fractional volume step used to generate
        calc_vp (volume V₀(1+δ)) and calc_vm (volume V₀(1-δ)).

        Parameters
        ----------
        calc_vp, calc_vm : PhononCalculator at V₀(1±dV_over_V)
        dV_over_V        : fractional volume change δ (e.g. 0.02 for ±2%)

        Returns
        -------
        gamma : (n_q, n_branches) float  — mode Grüneisen parameters
            Positive = frequency decreases on expansion (normal behaviour).
            q-mesh and branch ordering must match self._all_frequencies().
        """
        q_mesh = (20, 20, 20)
        omega0 = self._all_frequencies(q_mesh)          # (n_q, n_branches)
        omegap = calc_vp._all_frequencies(q_mesh)
        omegam = calc_vm._all_frequencies(q_mesh)

        # Central finite difference; guard against zero frequencies
        safe0 = np.where(omega0 > self._imag_tol, omega0, np.nan)
        gamma = -0.5 * (omegap - omegam) / (safe0 * dV_over_V)
        return gamma   # NaN where ω₀ ≤ imag_tol (acoustic / imaginary)

    # ── Updated correlators for SOTC self-consistency ─────────────────────────

    def updated_correlators(
        self,
        T: float,
        R_values_ang: np.ndarray,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> np.ndarray:
        """
        Compute updated C̄₂(R,T) [Å²] from the DFT-level IFCs.

        Used in the SOTC self-consistency loop to replace the initial Debye
        target with the DFT-derived correlators.

        C̄₂(R,T) = (ℏ/6NM) Σ_{q,s} [j₀(|q||R|) / ω_{qs}] coth(ℏω_{qs}/2k_BT)

        where j₀(x) = sin(x)/x is the isotropic (spherical) average of
        exp(iq·R) over all R directions at fixed |R|.  Imaginary modes
        (ω < imag_tol) are excluded; the runner compensates their missing
        spectral weight via Debye-model blending.
        """
        q1, q2, q3 = q_mesh
        n_q = q1 * q2 * q3
        # Build q-point list locally — avoids requesting unneeded eigenvectors.
        q_points = [
            np.array([i / q1, j / q2, k / q3])
            for i in range(q1) for j in range(q2) for k in range(q3)
        ]
        all_omegas = self._all_frequencies(q_mesh)   # (n_q, n_branches)
        M_avg = np.mean(self.masses_kg)
        R_arr = np.asarray(R_values_ang)              # (n_R,)
        C2 = np.zeros(len(R_arr))
        pref = HBAR / (6.0 * M_avg)                  # constant prefactor [J·s / kg]

        for idx, q_frac in enumerate(q_points):
            q_cart = q_frac @ self._rec_lat            # (3,) rad/Å
            q_norm = float(np.linalg.norm(q_cart))
            omegas = all_omegas[idx]                   # (n_branches,)

            # j₀(q|R|) — spherical average of exp(iq·R) at each shell distance.
            # Avoid division-by-zero at the Γ-point (q_norm = 0) → j₀ = 1.
            qR = q_norm * R_arr                        # (n_R,)
            j0 = np.where(qR > 1e-9, np.sin(qR) / qR, 1.0)   # (n_R,)

            for s, omega in enumerate(omegas):
                if omega < self._imag_tol:
                    # Imaginary/near-zero mode: exclude from the sum here.
                    # The missing spectral weight is recovered in the runner
                    # by blending with the Debye reference (proportional to
                    # the unstable-mode fraction).
                    continue
                x = HBAR * omega / (2.0 * KB * T) if T > 1e-9 else np.inf
                C2 += j0 * (pref * coth(x) / omega)

        C2 /= n_q
        return C2 * 1e20   # m² → Å²

    # ── Quantum displacement amplitudes (ZPM) ────────────────────────────────

    def quantum_displacement_amplitudes(
        self,
        T: float,
        q_mesh: Tuple[int, int, int] = (20, 20, 20),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantum harmonic displacement amplitude Q_{qs}(T) for each mode.

        Q_{qs}^{QM}(T) = √[ ħ/(2ω_{qs}) · coth(ħω_{qs}/2k_BT) ]

        Includes zero-point motion (ZPM):
          T → 0 : Q → √(ħ/2ω)      (ZPE floor, non-zero)
          T → ∞ : Q → √(k_BT)/ω   (classical limit)

        Use in the SOTC runner instead of classical Q = √(k_BT)/ω to
        correctly account for ZPM when T < T_D/2 (e.g. LiF at 800 K,
        hcp Mg at 700 K).

        Parameters
        ----------
        T      : float — temperature [K]
        q_mesh : (3,) int

        Returns
        -------
        frequencies_thz        : (n_q, n_branches) float [THz]
        amplitudes_ang_sqrtamu : (n_q, n_branches) float [Å·√amu]
            Zero for imaginary/acoustic (|ω| ≤ imag_tol) modes.
        """
        all_omegas = self._all_frequencies(q_mesh)   # (n_q, n_branches) [rad/s]
        amplitudes = np.zeros_like(all_omegas)
        pos = all_omegas > self._imag_tol
        omega_pos = all_omegas[pos]
        if T < 1e-9:
            amp_si = np.sqrt(HBAR / (2.0 * omega_pos))
        else:
            x = HBAR * omega_pos / (2.0 * KB * T)
            amp_si = np.sqrt(HBAR * coth(x) / (2.0 * omega_pos))
        # SI [√kg·m] → [Å·√amu]: ×1e10 (m→Å), ÷√AMU_TO_KG (kg½→amu½)
        amplitudes[pos] = amp_si * 1e10 / np.sqrt(AMU_TO_KG)
        return all_omegas * RAD_S_TO_THZ, amplitudes
