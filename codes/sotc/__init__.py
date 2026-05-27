"""
SOTC — Special Optimal Thermal Cells
=========================================
Python implementation of the SOTC framework for computing phonons and
thermodynamic properties from purposefully designed small supercells.

Algorithm outline
-----------------
1.  Compute target thermal displacement correlators C2(R,T) from the
    harmonic Debye model or phonopy phonons.
2.  Enumerate candidate supercells via Hermite Normal Form (HNF).
3.  Optimize atomic displacement amplitudes {A_qs} to minimise the
    SOTC quality functional Q[C2^SOTC - C2^target].
4.  Write displaced cells to VASP POSCAR files and run single-point DFT.
5.  Extract interatomic force constants (IFCs) by linear regression on
    force-displacement data.
6.  Reconstruct the phonon dispersion, DOS and heat capacity.
7.  Update target correlators and iterate to self-consistency.

References
----------
SOTC framework: SOTC_framework.md / SOTC_derivations.md (this repository)
SQS parent idea: Zunger et al., PRL 65, 353 (1990)
TDEP: Hellman et al., PRB 84, 180301 (2011)
SSCHA: Bianco et al., PRB 96, 014111 (2017)
"""

from .runner import SOTCRunner
from .correlators import DebyeCorrelator, PhononCorrelator
from .cell_design import HNFEnumerator, DisplacementOptimizer
from .qha import SOTCQuasiHarmonic
from .ifc_extractor import IFCExtractor
from .phonons import PhononCalculator
from .vasp_io import VASPWriter, VASPReader
from .mock_forces import LennardJonesForces
from .postprocessor import SOTCPostProcessor, SOTCRunLoader
from .gpaw_io import GPAWForceCalculator
from .qe_io import QEForceCalculator

__version__ = "0.1.0"
__all__ = [
    "SOTCRunner",
    "DebyeCorrelator",
    "PhononCorrelator",
    "HNFEnumerator",
    "DisplacementOptimizer",
    "IFCExtractor",
    "PhononCalculator",
    "VASPWriter",
    "VASPReader",
    "LennardJonesForces",
    "SOTCPostProcessor",
    "SOTCRunLoader",
    "GPAWForceCalculator",
    "QEForceCalculator",
]
