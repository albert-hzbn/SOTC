"""
GPAW force calculator for SQTC.

Wraps the GPAW DFT code (via ASE) as an SQTC force calculator,
providing the .compute() / .energy() interface expected by SQTCRunner
and mock_vasp_results().

GPAW is an open-source real-space / plane-wave DFT code:
    https://wiki.fysik.dtu.dk/gpaw

Installation
------------
    pip install gpaw
    gpaw install-data /path/to/gpaw-setups   # downloads PAW datasets

Usage
-----
    from sqtc.gpaw_io import GPAWForceCalculator

    calc = GPAWForceCalculator(
        species=['Al'] * 32,
        cutoff=450,
        xc='PBE',
        kpts=(2, 2, 2),
    )
    runner = SQTCRunner(
        element='Al', ...,
        force_calculator=calc,
    )

Thread safety
-------------
GPAW uses global internal state (libxc, BLAS) that is not thread-safe.
All calls to GPAW are serialised via a module-level lock so that the
SQTCRunner's ThreadPoolExecutor does not trigger simultaneous GPAW
invocations.  Snapshots are evaluated sequentially in terms of GPU/CPU
wall time, but the Python overhead between snapshots runs in parallel.

A result cache ensures that mock_vasp_results() calling .compute() then
.energy() with the same positions does not run two separate DFT jobs.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# Module-level lock: GPAW is not thread-safe; serialise all calculator calls.
_GPAW_LOCK = threading.Lock()


class GPAWForceCalculator:
    """
    In-process GPAW force calculator for SQTC.

    Parameters
    ----------
    species : list of str
        Chemical symbols for all atoms in the supercell, in the same order
        as the positions array passed to compute().  For a monatomic 32-atom
        fcc Al cell: ``['Al'] * 32``.
    cutoff : float [eV]
        Plane-wave energy cutoff.
    xc : str
        Exchange-correlation functional.  Default ``'PBE'``.
    kpts : tuple of int
        Monkhorst-Pack k-point grid for the *supercell*.  A (2,2,2) grid
        is adequate for a 32-atom fcc supercell; scale down from the
        primitive-cell grid proportionally to supercell size.
    smearing_width : float [eV]
        Fermi-Dirac smearing width for metallic occupation.  0.1 eV is
        standard for metals at room temperature.
    convergence_energy : float [eV/electron]
        SCF convergence threshold on the total energy.  1e-7 gives accurate
        forces for IFC regression.
    nbands : int or str, optional
        Number of bands.  GPAW accepts integers or strings like ``'120%'``
        (120 % of the number needed to fill all electrons).  None uses the
        GPAW default.
    eigensolver : str
        GPAW eigensolver.  ``'rmm-diis'`` is fast for metals.
    txt : str or None
        GPAW output file.  ``'-'`` → stdout, ``None`` → suppress all output,
        or a filename (e.g. ``'gpaw_al.out'``).
    workdir : str or Path, optional
        Directory for GPAW ``.gpw`` restart files.  If given, each snapshot
        is written to ``workdir/snap_{n:03d}.gpw`` so that a crashed run can
        be restarted.  Not used by default to keep examples simple.
    extra_kwargs : dict, optional
        Additional keyword arguments passed verbatim to the GPAW constructor.
        Use for advanced settings (parallel, poisson solver, etc.).
    """

    def __init__(
        self,
        species: List[str],
        cutoff: float = 450.0,
        xc: str = "PBE",
        kpts: Tuple[int, int, int] = (2, 2, 2),
        smearing_width: float = 0.1,
        convergence_energy: float = 1e-7,
        nbands: Union[int, str, None] = None,
        eigensolver: str = "rmm-diis",
        txt: Union[str, None] = "-",
        workdir: Union[str, Path, None] = None,
        extra_kwargs: Optional[Dict] = None,
    ):
        self.species = list(species)
        self.cutoff = float(cutoff)
        self.xc = xc
        self.kpts = tuple(kpts)
        self.smearing_width = smearing_width
        self.convergence_energy = convergence_energy
        self.nbands = nbands
        self.eigensolver = eigensolver
        self.txt = txt
        self.workdir = Path(workdir) if workdir is not None else None
        self.extra_kwargs = extra_kwargs or {}

        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)

        # Result cache: avoid running two DFT jobs when mock_vasp_results()
        # calls .compute() immediately followed by .energy() on the same cell.
        self._cache_key: Optional[bytes] = None
        self._cache_forces: Optional[np.ndarray] = None
        self._cache_energy: Optional[float] = None

        self._n_calls: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_gpaw_calc(self) -> "gpaw.GPAW":  # type: ignore[name-defined]
        try:
            from gpaw import GPAW, PW
        except ImportError as exc:
            raise ImportError(
                "GPAW is not installed. "
                "Install with: pip install gpaw && gpaw install-data <dir>"
            ) from exc

        kwargs: Dict = dict(
            xc=self.xc,
            kpts={"size": self.kpts, "gamma": True},
            occupations={"name": "fermi-dirac", "width": self.smearing_width},
            # Disable crystal symmetry: displaced supercells are not symmetric;
            # allowing symmetry reduction with random displacements introduces
            # subtle errors in force evaluation.
            symmetry={"point_group": False, "time_reversal": False},
            convergence={"energy": self.convergence_energy},
            eigensolver=self.eigensolver,
            txt=self.txt,
        )
        if self.nbands is not None:
            kwargs["nbands"] = self.nbands
        kwargs["mode"] = PW(self.cutoff)
        kwargs.update(self.extra_kwargs)

        return GPAW(**kwargs)

    def _make_atoms(self, positions: np.ndarray, cell: np.ndarray):
        from ase import Atoms

        return Atoms(
            symbols=self.species,
            positions=positions,
            cell=cell,
            pbc=True,
        )

    def _run_dft(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Run a single GPAW single-point calculation.

        Returns
        -------
        forces : (N, 3) ndarray [eV/Å]
        energy : float [eV]
        """
        atoms = self._make_atoms(positions, cell)
        atoms.calc = self._make_gpaw_calc()

        with _GPAW_LOCK:
            energy = float(atoms.get_potential_energy())
            forces = np.array(atoms.get_forces(), dtype=float)

        # Write restart file if workdir is set.
        if self.workdir is not None:
            restart_path = self.workdir / f"snap_{self._n_calls:03d}.gpw"
            try:
                atoms.calc.write(str(restart_path))
            except Exception:
                pass  # Non-critical — don't abort the run.

        self._n_calls += 1
        return forces, energy

    def _run_cached(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return cached result if positions/cell match the last call, otherwise
        run a new DFT calculation and update the cache.
        """
        key = positions.tobytes() + cell.tobytes()
        if key == self._cache_key:
            assert self._cache_forces is not None and self._cache_energy is not None
            return self._cache_forces, self._cache_energy

        forces, energy = self._run_dft(positions, cell)
        self._cache_key = key
        self._cache_forces = forces
        self._cache_energy = energy
        return forces, energy

    # ── Public interface (matches mock_forces contract) ───────────────────────

    def compute(self, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """
        Compute atomic forces.

        Parameters
        ----------
        positions : (N, 3) ndarray [Å]
        cell      : (3, 3) ndarray [Å]  (rows = lattice vectors)

        Returns
        -------
        forces : (N, 3) ndarray [eV/Å]
        """
        forces, _ = self._run_cached(positions, cell)
        return forces

    def energy(self, positions: np.ndarray, cell: np.ndarray) -> float:
        """
        Compute total DFT energy.

        Parameters
        ----------
        positions : (N, 3) ndarray [Å]
        cell      : (3, 3) ndarray [Å]

        Returns
        -------
        energy : float [eV]
        """
        _, energy = self._run_cached(positions, cell)
        return energy
