"""
VASP input file writer and output file parser for SOTC single-point calculations.

Writes
------
  POSCAR   — displaced atomic positions
  INCAR    — single-point settings (NSW=0, IBRION=-1)
  KPOINTS  — automatic Monkhorst-Pack mesh
  (POTCAR) — not generated; user must supply or use aflow/vaspkit

Reads
-----
  OUTCAR   — total energy, atomic forces, stress tensor
  vasprun.xml — alternative via ASE (optional)

VASP single-point settings used for SOTC:
  ISTART = 0 / 1 (fresh/restart)
  ISIF = 2       (fixed cell, compute stress)
  EDIFF = 1E-8   (tight energy convergence for accurate forces)
  PREC  = Accurate
  LWAVE = .FALSE. (no wavefunction write)
  LCHARG= .FALSE. (no charge density write)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class VASPWriter:
    """
    Write VASP input files for an SOTC single-point force calculation.

    Parameters
    ----------
    directory : str or Path
        Directory to write VASP files into (created if not exists).
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # ── POSCAR ────────────────────────────────────────────────────────────────

    def write_poscar(
        self,
        cell: np.ndarray,
        species: List[str],
        positions: np.ndarray,
        comment: str = "SOTC displaced cell",
        direct: bool = False,
    ) -> Path:
        """
        Write POSCAR file.

        Parameters
        ----------
        cell : (3, 3) float [Å]  — rows are lattice vectors
        species : list of str   — element symbols, one per atom
        positions : (n_atoms, 3) float [Å] — Cartesian positions
        comment : str
        direct : bool — write positions in fractional coords if True

        Returns
        -------
        Path to the written POSCAR file.
        """
        path = self.directory / "POSCAR"
        unique_species = list(dict.fromkeys(species))  # ordered unique
        counts = [species.count(s) for s in unique_species]

        lines = [comment + "\n"]
        lines.append("1.0\n")
        for vec in cell:
            lines.append(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}\n")
        lines.append("  " + "  ".join(unique_species) + "\n")
        lines.append("  " + "  ".join(str(c) for c in counts) + "\n")

        if direct:
            inv_cell = np.linalg.inv(cell)
            frac_pos = positions @ inv_cell
            lines.append("Direct\n")
            for sym, pos in zip(species, frac_pos):
                lines.append(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}\n")
        else:
            lines.append("Cartesian\n")
            for sym, pos in zip(species, positions):
                lines.append(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}\n")

        path.write_text("".join(lines))
        return path

    def write_displaced_poscar(
        self,
        cell: np.ndarray,
        species: List[str],
        equilibrium_positions: np.ndarray,
        displacements: np.ndarray,
        snapshot_index: int = 0,
    ) -> Path:
        """
        Write POSCAR with equilibrium + SOTC displacement field applied.

        Positions are clipped to the unit cell using minimum image convention.
        """
        displaced = equilibrium_positions + displacements
        comment = f"SOTC snapshot {snapshot_index}"
        return self.write_poscar(cell, species, displaced, comment=comment)

    # ── INCAR ─────────────────────────────────────────────────────────────────

    def write_incar(
        self,
        params: Optional[Dict[str, str]] = None,
        encut: float = 500.0,
        functional: str = "PBE",
        ncore: int = 4,
    ) -> Path:
        """
        Write INCAR for a single-point force/energy calculation.

        Parameters
        ----------
        params : dict, optional
            Additional INCAR tags to override defaults.
        encut : float [eV]
        functional : 'PBE' | 'PBE-D3' | 'LDA'
        ncore : int  parallel decomposition

        Returns
        -------
        Path to INCAR.
        """
        defaults = {
            "SYSTEM": "SOTC single-point",
            "ISTART": "0",
            "ICHARG": "2",
            "ENCUT": f"{encut:.1f}",
            "EDIFF": "1E-8",
            "PREC": "Accurate",
            "NSW": "0",
            "IBRION": "-1",
            "ISIF": "2",
            "ISMEAR": "0",   # Gaussian smearing — correct for insulators/semiconductors
            "SIGMA": "0.05",
            "ISYM": "-1",    # no symmetry for displaced cells
            "ALGO": "Fast",
            "NCORE": str(ncore),
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "LREAL": ".FALSE.",  # exact projection for small cells
        }

        if functional == "PBE-D3":
            defaults["IVDW"] = "12"   # DFT-D3 with Becke-Johnson damping
        elif functional == "LDA":
            pass  # LDA is the default for PAW-LDA POTCARs

        if params:
            defaults.update(params)

        path = self.directory / "INCAR"
        lines = []
        for key, val in defaults.items():
            lines.append(f"  {key} = {val}\n")
        path.write_text("".join(lines))
        return path

    # ── KPOINTS ───────────────────────────────────────────────────────────────

    def write_kpoints(
        self,
        kgrid: Tuple[int, int, int] = (4, 4, 4),
        shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scheme: str = "Monkhorst",
    ) -> Path:
        """Write KPOINTS file."""
        path = self.directory / "KPOINTS"
        lines = [
            "Automatic\n",
            "0\n",
            f"{scheme}\n",
            f"  {kgrid[0]:d}  {kgrid[1]:d}  {kgrid[2]:d}\n",
            f"  {shift[0]:.1f}  {shift[1]:.1f}  {shift[2]:.1f}\n",
        ]
        path.write_text("".join(lines))
        return path

    def write_potcar(
        self,
        species: List[str],
        pp_base_dir: str | Path,
        pp_set: str = "PAW_PBE",
    ) -> Path:
        """
        Assemble POTCAR by concatenating per-element POTCARs.

        Parameters
        ----------
        species : list of str — element symbols in POSCAR order
        pp_base_dir : path — base pseudopotential directory,
            e.g. '/home/albert/softwares/vasp/Pseudopotentials'
        pp_set : str — subdirectory name, e.g. 'PAW_PBE'

        Returns
        -------
        Path to the assembled POTCAR.
        """
        pp_dir = Path(pp_base_dir) / pp_set
        unique = list(dict.fromkeys(species))  # preserves order
        potcar_path = self.directory / "POTCAR"
        with open(potcar_path, "wb") as out:
            for elem in unique:
                src = pp_dir / elem / "POTCAR"
                if not src.exists():
                    raise FileNotFoundError(
                        f"POTCAR not found for element '{elem}': {src}"
                    )
                out.write(src.read_bytes())
        return potcar_path

    def write_all(
        self,
        cell: np.ndarray,
        species: List[str],
        equilibrium_positions: np.ndarray,
        displacements: np.ndarray,
        snapshot_index: int = 0,
        kgrid: Tuple[int, int, int] = (4, 4, 4),
        encut: float = 500.0,
        functional: str = "PBE",
        ncore: int = 4,
        extra_incar: Optional[Dict] = None,
        pp_base_dir: Optional[Path] = None,
        pp_set: str = "PAW_PBE",
    ) -> None:
        """Convenience: write POSCAR + INCAR + KPOINTS (+ POTCAR) in one call."""
        self.write_displaced_poscar(
            cell, species, equilibrium_positions, displacements, snapshot_index
        )
        self.write_incar(
            params=extra_incar, encut=encut, functional=functional, ncore=ncore
        )
        self.write_kpoints(kgrid=kgrid)
        if pp_base_dir is not None:
            self.write_potcar(species, pp_base_dir, pp_set)


class VASPReader:
    """
    Parse VASP output files to extract forces, energy, and stress.

    Parameters
    ----------
    directory : str or Path
        Directory containing VASP output files.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    # ── OUTCAR parser ─────────────────────────────────────────────────────────

    def read_outcar(self) -> Dict:
        """
        Parse OUTCAR for total energy, atomic forces, and stress tensor.

        Returns
        -------
        results : dict with keys:
            'energy_ev'   : float  — total DFT energy [eV]
            'forces_ev_ang' : (n_atoms, 3) ndarray [eV/Å]
            'stress_kbar' : (3, 3) ndarray [kBar]
            'converged'   : bool  — whether SCF converged
        """
        outcar = self.directory / "OUTCAR"
        if not outcar.exists():
            raise FileNotFoundError(f"OUTCAR not found in {self.directory}")

        text = outcar.read_text()
        return {
            "energy_ev": self._parse_energy(text),
            "forces_ev_ang": self._parse_forces(text),
            "stress_kbar": self._parse_stress(text),
            "converged": self._check_convergence(text),
        }

    def _parse_energy(self, text: str) -> float:
        """Extract the last total energy (TOTEN) from OUTCAR."""
        pattern = r"energy  without entropy=\s*([-\d.E+]+)"
        matches = re.findall(pattern, text)
        if not matches:
            pattern = r"TOTEN\s*=\s*([-\d.E+]+)\s*eV"
            matches = re.findall(pattern, text)
        if not matches:
            raise ValueError("Could not parse total energy from OUTCAR")
        return float(matches[-1])

    def _parse_forces(self, text: str) -> np.ndarray:
        """
        Extract the last set of ionic forces from OUTCAR.

        Looks for the block:
            TOTAL-FORCE (eV/Angst)
            ...
            ------- -------...
            x1  y1  z1  Fx1  Fy1  Fz1
            ...
        """
        # Find all force blocks
        pattern = (
            r"TOTAL-FORCE \(eV/Angst\)\s*"
            r"[-\s]+\n"
            r"((?:\s*[-\d.E+]+\s*[-\d.E+]+\s*[-\d.E+]+"
            r"\s*[-\d.E+]+\s*[-\d.E+]+\s*[-\d.E+]+\n)+)"
        )
        blocks = re.findall(pattern, text)
        if not blocks:
            raise ValueError("Could not find TOTAL-FORCE block in OUTCAR")

        last_block = blocks[-1]
        lines = last_block.strip().split("\n")
        forces = []
        for line in lines:
            vals = line.split()
            if len(vals) == 6:
                # cols 3,4,5 are forces
                forces.append([float(vals[3]), float(vals[4]), float(vals[5])])
        return np.array(forces)

    def _parse_stress(self, text: str) -> np.ndarray:
        """Extract the last stress tensor [kBar] from OUTCAR."""
        # Look for:  in kB  XX  YY  ZZ  XY  YZ  ZX
        pattern = r"in kB\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)"
        matches = re.findall(pattern, text)
        if not matches:
            return np.zeros((3, 3))
        vals = [float(v) for v in matches[-1]]
        # xx, yy, zz, xy, yz, zx
        s = np.array([
            [vals[0], vals[3], vals[5]],
            [vals[3], vals[1], vals[4]],
            [vals[5], vals[4], vals[2]],
        ])
        return s

    def _check_convergence(self, text: str) -> bool:
        """Check whether the VASP SCF loop converged."""
        return "aborting loop because EDIFF is reached" in text

    # ── vasprun.xml parser (via ASE, optional) ────────────────────────────────

    def read_vasprun(self) -> Dict:
        """
        Parse vasprun.xml using ASE (requires ase installed).

        Returns same dict format as read_outcar().
        """
        try:
            from ase.io import read as ase_read
        except ImportError as e:
            raise ImportError("ASE is required for vasprun.xml parsing") from e

        vasprun = self.directory / "vasprun.xml"
        if not vasprun.exists():
            raise FileNotFoundError(f"vasprun.xml not found in {self.directory}")

        atoms = ase_read(str(vasprun), format="vasp-xml")
        return {
            "energy_ev": atoms.get_potential_energy(),
            "forces_ev_ang": atoms.get_forces(),
            "stress_kbar": atoms.get_stress(voigt=False) / 0.1,  # GPa → kBar (× 10)
            "converged": True,
        }


class VASPRunner:
    """
    Run VASP calculations for each SOTC snapshot.

    Parameters
    ----------
    vasp_cmd : str or list
        Command to invoke VASP, e.g. 'mpirun -np 8 vasp_std' or
        ['mpirun', '-np', '8', 'vasp_std'].
    base_dir : str or Path
        Base directory; each snapshot gets a subdirectory.
    """

    def __init__(
        self,
        vasp_cmd: str | List[str],
        base_dir: str | Path = "sqtc_calculations",
    ):
        self.vasp_cmd = vasp_cmd if isinstance(vasp_cmd, list) else vasp_cmd.split()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_snapshot(
        self,
        snapshot_index: int,
        cell: np.ndarray,
        species: List[str],
        eq_positions: np.ndarray,
        displacements: np.ndarray,
        kgrid: Tuple[int, int, int] = (4, 4, 4),
        encut: float = 500.0,
        functional: str = "PBE",
        ncore: int = 4,
        potcar_dir: Optional[Path] = None,
        pp_base_dir: Optional[Path] = None,
        pp_set: str = "PAW_PBE",
        extra_incar: Optional[Dict] = None,
        timeout: int = 7200,
    ) -> Dict:
        """
        Prepare inputs, run VASP, and return parsed results for one snapshot.

        Parameters
        ----------
        potcar_dir : Path, optional
            Directory containing a pre-assembled POTCAR to copy in.
        pp_base_dir : Path, optional
            Base pseudopotential directory (e.g. '.../Pseudopotentials').
            When given, POTCARs are assembled automatically from
            ``pp_base_dir / pp_set / <element> / POTCAR`` for each species.
            Takes precedence over potcar_dir.
        extra_incar : dict, optional
            Additional INCAR tags overriding defaults (e.g. KSPACING).
        timeout : int
            Maximum wall-clock time [s] for the VASP job.

        Returns
        -------
        results : dict  (see VASPReader.read_outcar)
        """
        snap_dir = self.base_dir / f"snap_{snapshot_index:04d}"
        writer = VASPWriter(snap_dir)

        # VASP POSCAR requires atoms grouped by species (all Mg, then all O, …).
        # Build a stable sort that groups identical species together while
        # preserving relative atom order within each species block.
        sort_order = sorted(range(len(species)), key=lambda i: species[i])
        inv_order = [0] * len(sort_order)
        for new_i, old_i in enumerate(sort_order):
            inv_order[old_i] = new_i
        species_sorted   = [species[i]     for i in sort_order]
        eq_sorted        = eq_positions[sort_order]
        disp_sorted      = displacements[sort_order]

        writer.write_all(
            cell, species_sorted, eq_sorted, disp_sorted,
            snapshot_index=snapshot_index,
            kgrid=kgrid, encut=encut, functional=functional, ncore=ncore,
            extra_incar=extra_incar,
        )

        # Assemble POTCAR (uses species_sorted so element order matches POSCAR)
        if pp_base_dir is not None:
            writer.write_potcar(species_sorted, pp_base_dir, pp_set)
        elif potcar_dir is not None:
            potcar_src = Path(potcar_dir) / "POTCAR"
            if potcar_src.exists():
                import shutil
                shutil.copy(potcar_src, snap_dir / "POTCAR")

        # Run VASP
        log_path = snap_dir / "vasp.log"
        with open(log_path, "w") as log:
            proc = subprocess.run(
                self.vasp_cmd,
                cwd=str(snap_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"VASP failed for snapshot {snapshot_index} "
                f"(return code {proc.returncode}). See {log_path}"
            )

        reader = VASPReader(snap_dir)
        try:
            result = reader.read_outcar()
        except Exception:
            result = reader.read_vasprun()

        # Undo species-sort: return forces in caller's original atom order.
        result["forces_ev_ang"] = result["forces_ev_ang"][inv_order]
        return result

    def run_ensemble(
        self,
        ensemble_displacements: List[np.ndarray],
        cell: np.ndarray,
        species: List[str],
        eq_positions: np.ndarray,
        **kwargs,
    ) -> List[Dict]:
        """
        Run VASP for all snapshots in the SOTC ensemble sequentially.

        Returns list of result dicts (one per snapshot).
        """
        results = []
        for m, u in enumerate(ensemble_displacements):
            print(f"  Running VASP snapshot {m+1}/{len(ensemble_displacements)} ...", end=" ", flush=True)
            res = self.run_snapshot(m, cell, species, eq_positions, u, **kwargs)
            results.append(res)
            print(f"E = {res['energy_ev']:.6f} eV  converged={res['converged']}")
        return results
