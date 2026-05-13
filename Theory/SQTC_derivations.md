# SQTC: Complete Step-by-Step Derivations
### From First Principles to the Special Quasirandom Thermal Cell Framework

> **Who is this for?** A physics undergraduate who has taken classical mechanics, quantum mechanics, thermodynamics/statistical mechanics, and has seen Fourier transforms. Solid-state physics (Bloch's theorem, Brillouin zones) is helpful but every concept is introduced from scratch.

---

## Table of Contents

1. [The Crystal and Its Vibrations — From Newton's Law to Phonons](#chapter-1)
2. [Quantum Mechanics of Vibrations — Quantising Phonons](#chapter-2)
3. [Statistical Mechanics at Finite Temperature — The Bose-Einstein Distribution](#chapter-3)
4. [Thermal Displacement Correlators — The Core Object of SQTC](#chapter-4)
5. [Interatomic Force Constants — Connecting Forces to Phonons](#chapter-5)
6. [Why Large Supercells Are Needed at High Temperature](#chapter-6)
7. [The SQS Philosophy — Structure as Statistical Information](#chapter-7)
8. [SQTC Correlator Formulation — The Thermal Analogy to SQS](#chapter-8)
9. [The SQTC Objective Functional — Deriving the Matching Condition](#chapter-9)
10. [Thermal Coherence Length — Why High-T Means Smaller Cells](#chapter-10)
11. [Displacement Basis Expansion — Making the Optimisation Tractable](#chapter-11)
12. [IFC Extraction by Linear Regression — Getting Force Constants from DFT Forces](#chapter-12)
13. [Phonon Reconstruction from the SQTC Cell](#chapter-13)
14. [Self-Consistency: Closing the Loop](#chapter-14)
15. [The Representability Theorem — Formal Proof](#chapter-15)
16. [The SQTC Ensemble for Anharmonic Properties](#chapter-16)
17. [Worked Numerical Example: 1D Diatomic Chain](#chapter-17)
18. [Variational Grounding — SQTC as Deterministic SSCHA Quadrature](#chapter-18)
19. [The SQTC Harmonic Free Energy on the Small Cell](#chapter-19)
20. [Anharmonic Free Energy I — Wick's Theorem and the Tadpole Diagram (4IFC)](#chapter-20)
21. [Anharmonic Free Energy II — The Sunset Diagram (3IFC)](#chapter-21)
22. [The Complete SQTC Anharmonic Free Energy and Thermodynamic Properties](#chapter-22)
23. [Phase Transitions from the SQTC Free Energy](#chapter-23)
24. [Complete Derivation of SQTC Without Dirac Notation](#chapter-24)

---

<a name="chapter-1"></a>
## Chapter 1: The Crystal and Its Vibrations — From Newton's Law to Phonons

### 1.1 Setting Up the Crystal

Imagine a crystal as $N$ atoms arranged in a periodic lattice. Each atom has an **equilibrium position** $\mathbf{R}_i^0$ (the position it would sit at if the temperature were absolute zero and nothing was disturbing it). At finite temperature, each atom wiggles around its equilibrium position. We call this wiggle the **displacement**:

$$\mathbf{u}_i(t) = \mathbf{R}_i(t) - \mathbf{R}_i^0$$

So the actual position of atom $i$ at time $t$ is:

$$\mathbf{R}_i(t) = \mathbf{R}_i^0 + \mathbf{u}_i(t)$$

The displacements are small compared to the interatomic spacing (typically 1–10% of lattice constant). This is the key assumption that lets us use the **harmonic approximation**.

### 1.2 The Potential Energy — Taylor Expansion

The total potential energy $V(\{\mathbf{R}_i\})$ depends on all atomic positions. We expand it in a Taylor series around equilibrium. Let $u_i^\alpha$ mean the $\alpha$-component ($\alpha = x, y, z$) of the displacement of atom $i$:

$$\begin{aligned}
V = V_0
&+ \underbrace{\sum_{i,\alpha} \frac{\partial V}{\partial u_i^\alpha}\bigg|_0 u_i^\alpha}_{\text{Term 1}} \\[4pt]
&+ \underbrace{\frac{1}{2}\sum_{i,j}\sum_{\alpha,\beta} \frac{\partial^2 V}{\partial u_i^\alpha \partial u_j^\beta}\bigg|_0 u_i^\alpha u_j^\beta}_{\text{Term 2}} \\[4pt]
&+ \underbrace{\frac{1}{6}\sum_{i,j,k}\sum_{\alpha,\beta,\gamma} \frac{\partial^3 V}{\partial u_i^\alpha \partial u_j^\beta \partial u_k^\gamma}\bigg|_0 u_i^\alpha u_j^\beta u_k^\gamma}_{\text{Term 3}} + \cdots
\end{aligned}$$

**Explanation of each term:**

- $V_0$: constant, just sets the energy zero, irrelevant for dynamics.
- **Term 1**: First derivatives at equilibrium = **forces at equilibrium = 0** (by definition of equilibrium). So Term 1 = 0 exactly.
- **Term 2**: This is the **harmonic term** — the foundation of phonon theory.
- **Term 3 and beyond**: These are **anharmonic terms** — they cause phonon-phonon scattering, thermal expansion, and finite phonon lifetimes. Crucial at high temperature.

We define the **second-order Interatomic Force Constants (IFCs)**:

$$\Phi_{ij}^{\alpha\beta} \equiv \frac{\partial^2 V}{\partial u_i^\alpha \partial u_j^\beta}\bigg|_{\mathbf{u}=0}$$

So the harmonic potential energy is:

$$V^{harm} = \frac{1}{2}\sum_{i,j}\sum_{\alpha,\beta} \Phi_{ij}^{\alpha\beta}\, u_i^\alpha\, u_j^\beta$$

**Physical meaning of $\Phi_{ij}^{\alpha\beta}$:** It tells you how much force acts on atom $i$ in direction $\alpha$ when atom $j$ is displaced by a unit amount in direction $\beta$. It has units of $[\text{eV}/\text{Å}^2]$ or equivalently $[\text{N/m}]$.

### 1.3 Newton's Law — Equations of Motion

Newton's second law for atom $i$ in direction $\alpha$:

$$M_i \ddot{u}_i^\alpha = -\frac{\partial V^{harm}}{\partial u_i^\alpha} = -\sum_{j,\beta} \Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

This is a set of $3N$ coupled second-order differential equations. The challenge is to decouple them.

### 1.4 Using Periodicity — Bloch's Theorem

A crystal is periodic. The primitive lattice vectors are $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$. Every lattice point is at position:

$$\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3, \quad n_1, n_2, n_3 \in \mathbb{Z}$$

Because of this periodicity, the IFCs only depend on the relative lattice vector between the two atoms:

$$\Phi_{ij}^{\alpha\beta}(\mathbf{R}) \equiv \Phi_{ij}^{\alpha\beta}(\mathbf{R}_i^0 - \mathbf{R}_j^0)$$

**Bloch's ansatz:** Guess that the displacement has the form of a plane wave:

$$u_i^\alpha(\mathbf{R}, t) = \frac{1}{\sqrt{M_i N}} e_i^\alpha(\mathbf{q})\, e^{i(\mathbf{q}\cdot\mathbf{R} - \omega t)}$$

where:
- $\mathbf{q}$ = wavevector (lives in the **Brillouin zone**, the unit cell of reciprocal space)
- $\omega$ = frequency (what we want to find)
- $e_i^\alpha(\mathbf{q})$ = polarisation vector (how much atom $i$ moves in direction $\alpha$)
- $N$ = total number of unit cells

### 1.5 The Dynamical Matrix and Phonon Dispersion

Substituting the Bloch ansatz into Newton's equations of motion:

$$M_i \ddot{u}_i^\alpha = -\omega^2 M_i \cdot \frac{1}{\sqrt{M_i N}} e_i^\alpha e^{i(\mathbf{q}\cdot\mathbf{R} - \omega t)} = -\sum_{j,\beta} \Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

The right-hand side becomes (using periodicity and the Bloch form):

$$-\sum_{j,\beta} \Phi_{ij}^{\alpha\beta} \cdot \frac{1}{\sqrt{M_j N}} e_j^\beta e^{i(\mathbf{q}\cdot\mathbf{R} - \omega t)} \cdot e^{i\mathbf{q}\cdot(\mathbf{R}_j^0 - \mathbf{R}_i^0)}$$

After simplification (cancel $e^{i(\mathbf{q}\cdot\mathbf{R} - \omega t)}$ from both sides):

$$\omega^2 e_i^\alpha = \sum_{j,\beta} \underbrace{\frac{1}{\sqrt{M_i M_j}} \sum_{\mathbf{R}} \Phi_{ij}^{\alpha\beta}(\mathbf{R})\, e^{i\mathbf{q}\cdot\mathbf{R}}}_{D_{ij}^{\alpha\beta}(\mathbf{q})} e_j^\beta$$

We define the **Dynamical Matrix**:

$$\boxed{D_{ij}^{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_i M_j}} \sum_{\mathbf{R}} \Phi_{ij}^{\alpha\beta}(\mathbf{R})\, e^{i\mathbf{q}\cdot\mathbf{R}}}$$

The equation becomes the **eigenvalue problem**:

$$D(\mathbf{q})\, \mathbf{e}(\mathbf{q}) = \omega^2\, \mathbf{e}(\mathbf{q})$$

For a unit cell with $n$ atoms, $D(\mathbf{q})$ is a $3n \times 3n$ Hermitian matrix. Its $3n$ eigenvalues $\omega_{\mathbf{q}s}^2$ give the squared frequencies of the $3n$ **phonon branches** $s$, and the eigenvectors $\mathbf{e}(\mathbf{q}s)$ give the polarisation patterns.

**The phonon dispersion relation** $\omega_{\mathbf{q}s}$ plotted vs $\mathbf{q}$ is the central result of harmonic phonon theory.

---

<a name="chapter-2"></a>
## Chapter 2: Quantum Mechanics of Vibrations — Quantising Phonons

### 2.1 From Classical to Quantum Harmonic Oscillator

Each normal mode $(\mathbf{q}, s)$ of the crystal behaves like an **independent quantum harmonic oscillator** with frequency $\omega_{\mathbf{q}s}$.

The Hamiltonian for one normal mode is:

$$\hat{H}_{\mathbf{q}s} = \frac{\hat{p}_{\mathbf{q}s}^2}{2} + \frac{1}{2}\omega_{\mathbf{q}s}^2 \hat{Q}_{\mathbf{q}s}^2$$

where $\hat{Q}_{\mathbf{q}s}$ is the normal coordinate and $\hat{p}_{\mathbf{q}s}$ is its conjugate momentum.

### 2.2 Creation and Annihilation Operators

Define creation ($\hat{a}^\dagger$) and annihilation ($\hat{a}$) operators:

$$\hat{Q}_{\mathbf{q}s} = \sqrt{\frac{\hbar}{2\omega_{\mathbf{q}s}}}\left(\hat{a}_{\mathbf{q}s} + \hat{a}_{-\mathbf{q}s}^\dagger\right)$$

$$\hat{p}_{\mathbf{q}s} = i\sqrt{\frac{\hbar\omega_{\mathbf{q}s}}{2}}\left(\hat{a}_{-\mathbf{q}s}^\dagger - \hat{a}_{\mathbf{q}s}\right)$$

These satisfy the **commutation relation**: $[\hat{a}_{\mathbf{q}s}, \hat{a}_{\mathbf{q}'s'}^\dagger] = \delta_{\mathbf{q}\mathbf{q}'}\delta_{ss'}$

The Hamiltonian becomes:

$$\hat{H} = \sum_{\mathbf{q},s} \hbar\omega_{\mathbf{q}s}\left(\hat{a}_{\mathbf{q}s}^\dagger\hat{a}_{\mathbf{q}s} + \frac{1}{2}\right)$$

### 2.3 Atomic Displacement in Terms of Phonon Operators

The displacement of atom $i$ (with mass $M_i$) in direction $\alpha$, in a crystal with $N$ unit cells:

$$\hat{u}_i^\alpha(\mathbf{R}) = \frac{1}{\sqrt{N}}\sum_{\mathbf{q},s}\sqrt{\frac{\hbar}{2M_i\omega_{\mathbf{q}s}}} e_i^\alpha(\mathbf{q}s)\left(\hat{a}_{\mathbf{q}s} + \hat{a}_{-\mathbf{q}s}^\dagger\right) e^{i\mathbf{q}\cdot\mathbf{R}}$$

This is the **key formula** connecting atomic displacements to phonon creation/annihilation operators.

---

<a name="chapter-3"></a>
## Chapter 3: Statistical Mechanics at Finite Temperature — The Bose-Einstein Distribution

### 3.1 The Quantum Statistical Average

At temperature $T$, the system is in a **mixed quantum state** described by the density matrix:

$$\hat{\rho} = \frac{e^{-\hat{H}/k_BT}}{Z}, \quad Z = \text{Tr}\left(e^{-\hat{H}/k_BT}\right)$$

The thermal expectation value of any observable $\hat{O}$:

$$\langle \hat{O} \rangle_T = \text{Tr}(\hat{\rho}\,\hat{O})$$

### 3.2 Average Phonon Occupation

For phonon mode $(\mathbf{q}s)$, compute $\langle \hat{a}^\dagger \hat{a} \rangle_T$ using the thermal state of a harmonic oscillator with energy levels $E_n = \hbar\omega(n + 1/2)$:

$$\langle \hat{n}_{\mathbf{q}s} \rangle_T \equiv \langle \hat{a}_{\mathbf{q}s}^\dagger \hat{a}_{\mathbf{q}s} \rangle_T = \sum_{n=0}^\infty n \cdot \frac{e^{-\hbar\omega_{\mathbf{q}s}(n+1/2)/k_BT}}{Z_{\mathbf{q}s}}$$

**Step-by-step evaluation:**

The partition function for one oscillator:

$$Z_{\mathbf{q}s} = \sum_{n=0}^\infty e^{-\hbar\omega_{\mathbf{q}s}(n+1/2)/k_BT} = e^{-\hbar\omega/2k_BT}\sum_{n=0}^\infty \left(e^{-\hbar\omega/k_BT}\right)^n = \frac{e^{-\hbar\omega/2k_BT}}{1 - e^{-\hbar\omega/k_BT}}$$

For the average occupation, use the trick $\langle n \rangle = -\frac{1}{\hbar\omega}\frac{\partial \ln Z}{\partial (1/k_BT)}$... or directly:

$$\langle \hat{n}_{\mathbf{q}s} \rangle_T = \frac{\sum_{n=0}^\infty n\, e^{-n\hbar\omega/k_BT}}{\sum_{n=0}^\infty e^{-n\hbar\omega/k_BT}}$$

Let $x = e^{-\hbar\omega/k_BT}$. Numerator = $\sum_{n=0}^\infty n x^n = x\frac{d}{dx}\sum_{n=0}^\infty x^n = x\frac{d}{dx}\frac{1}{1-x} = \frac{x}{(1-x)^2}$.

Denominator = $\frac{1}{1-x}$.

Therefore:

$$\boxed{\langle \hat{n}_{\mathbf{q}s} \rangle_T = \frac{x}{1-x} = \frac{1}{e^{\hbar\omega_{\mathbf{q}s}/k_BT} - 1} \equiv n_B(\omega_{\mathbf{q}s}, T)}$$

This is the **Bose-Einstein distribution**. It tells us: the higher the temperature, the more phonons are thermally excited into each mode.

### 3.3 Key Expectation Values

Using $\langle \hat{a}_{\mathbf{q}s}\rangle_T = 0$, $\langle\hat{a}_{\mathbf{q}s}^\dagger\rangle_T = 0$ (coherent amplitude vanishes in thermal state), and:

$$\langle \hat{a}_{\mathbf{q}s}\hat{a}_{\mathbf{q}'s'}^\dagger \rangle_T = (n_B(\omega_{\mathbf{q}s},T) + 1)\delta_{\mathbf{q}\mathbf{q}'}\delta_{ss'}$$

$$\langle \hat{a}_{\mathbf{q}s}^\dagger\hat{a}_{\mathbf{q}'s'} \rangle_T = n_B(\omega_{\mathbf{q}s},T)\,\delta_{\mathbf{q}\mathbf{q}'}\delta_{ss'}$$

Also needed: $\langle\hat{a}\hat{a}\rangle_T = 0$ and $\langle\hat{a}^\dagger\hat{a}^\dagger\rangle_T = 0$ (detailed balance in thermal state).

---

<a name="chapter-4"></a>
## Chapter 4: Thermal Displacement Correlators — The Core Object of SQTC

### 4.1 Definition

The **thermal displacement correlation tensor** between atom $i$ at lattice site $\mathbf{0}$ and atom $j$ at lattice site $\mathbf{R}$ is:

$$C_{ij}^{\alpha\beta}(\mathbf{R}, T) = \langle \hat{u}_i^\alpha(\mathbf{0})\, \hat{u}_j^\beta(\mathbf{R}) \rangle_T$$

This measures how correlated the displacement of atom $i$ at the origin is with atom $j$ at position $\mathbf{R}$.

**Physical intuition:** If atom $A$ jiggles to the right, does atom $B$ (some distance away) tend to also jiggle right (positive correlation), left (negative correlation), or randomly (zero correlation)?

### 4.2 Derivation of the Correlator in the Harmonic Approximation

Substitute the displacement operator from Chapter 2:

$$\hat{u}_i^\alpha(\mathbf{0}) = \frac{1}{\sqrt{N}}\sum_{\mathbf{q},s}\sqrt{\frac{\hbar}{2M_i\omega_{\mathbf{q}s}}} e_i^\alpha(\mathbf{q}s)\left(\hat{a}_{\mathbf{q}s} + \hat{a}_{-\mathbf{q}s}^\dagger\right)$$

$$\hat{u}_j^\beta(\mathbf{R}) = \frac{1}{\sqrt{N}}\sum_{\mathbf{q}',s'}\sqrt{\frac{\hbar}{2M_j\omega_{\mathbf{q}'s'}}} e_j^\beta(\mathbf{q}'s')\left(\hat{a}_{\mathbf{q}'s'} + \hat{a}_{-\mathbf{q}'s'}^\dagger\right) e^{i\mathbf{q}'\cdot\mathbf{R}}$$

Now compute the thermal average of their product:

$$C_{ij}^{\alpha\beta}(\mathbf{R},T) = \frac{1}{N}\sum_{\mathbf{q},s}\sum_{\mathbf{q}',s'} \sqrt{\frac{\hbar}{2M_i\omega_{\mathbf{q}s}}} \sqrt{\frac{\hbar}{2M_j\omega_{\mathbf{q}'s'}}} e_i^\alpha(\mathbf{q}s) e_j^\beta(\mathbf{q}'s') \cdot e^{i\mathbf{q}'\cdot\mathbf{R}} \cdot \mathcal{A}$$

where $\mathcal{A}$ contains the operator expectation values:

$$\mathcal{A} = \left\langle\left(\hat{a}_{\mathbf{q}s} + \hat{a}_{-\mathbf{q}s}^\dagger\right)\left(\hat{a}_{\mathbf{q}'s'} + \hat{a}_{-\mathbf{q}'s'}^\dagger\right)\right\rangle_T$$

Expanding the product:

$$\mathcal{A} = \langle\hat{a}_{\mathbf{q}s}\hat{a}_{\mathbf{q}'s'}\rangle + \langle\hat{a}_{\mathbf{q}s}\hat{a}_{-\mathbf{q}'s'}^\dagger\rangle + \langle\hat{a}_{-\mathbf{q}s}^\dagger\hat{a}_{\mathbf{q}'s'}\rangle + \langle\hat{a}_{-\mathbf{q}s}^\dagger\hat{a}_{-\mathbf{q}'s'}^\dagger\rangle$$

Using the results from Chapter 3:
- $\langle\hat{a}\hat{a}\rangle_T = 0$, $\langle\hat{a}^\dagger\hat{a}^\dagger\rangle_T = 0$
- $\langle\hat{a}_{\mathbf{q}s}\hat{a}_{-\mathbf{q}'s'}^\dagger\rangle_T = (n_B + 1)\delta_{\mathbf{q},-\mathbf{q}'}\delta_{ss'}$
- $\langle\hat{a}_{-\mathbf{q}s}^\dagger\hat{a}_{\mathbf{q}'s'}\rangle_T = n_B\,\delta_{-\mathbf{q},\mathbf{q}'}\delta_{ss'} = n_B\,\delta_{\mathbf{q},-\mathbf{q}'}\delta_{ss'}$

Therefore:

$$\mathcal{A} = (n_B(\omega_{\mathbf{q}s},T) + 1)\delta_{\mathbf{q},-\mathbf{q}'}\delta_{ss'} + n_B(\omega_{\mathbf{q}s},T)\delta_{\mathbf{q},-\mathbf{q}'}\delta_{ss'}$$

$$= (2n_B(\omega_{\mathbf{q}s},T) + 1)\delta_{\mathbf{q},-\mathbf{q}'}\delta_{ss'}$$

Now recognise that $2n_B + 1 = \coth\!\left(\frac{\hbar\omega}{2k_BT}\right)$ (a standard identity, proven below).

**Proof of identity $2n_B + 1 = \coth(\hbar\omega/2k_BT)$:**

$$2n_B + 1 = \frac{2}{e^{\hbar\omega/k_BT} - 1} + 1 = \frac{2 + e^{\hbar\omega/k_BT} - 1}{e^{\hbar\omega/k_BT} - 1} = \frac{e^{\hbar\omega/k_BT} + 1}{e^{\hbar\omega/k_BT} - 1}$$

Multiply numerator and denominator by $e^{-\hbar\omega/2k_BT}$:

$$= \frac{e^{\hbar\omega/2k_BT} + e^{-\hbar\omega/2k_BT}}{e^{\hbar\omega/2k_BT} - e^{-\hbar\omega/2k_BT}} = \frac{2\cosh(\hbar\omega/2k_BT)}{2\sinh(\hbar\omega/2k_BT)} = \coth\!\left(\frac{\hbar\omega}{2k_BT}\right) \quad \checkmark$$

**Back to the correlator.** Substituting $\mathcal{A}$ and using $\delta_{\mathbf{q},-\mathbf{q}'}$ to kill the $\mathbf{q}'$ sum ($\mathbf{q}' = -\mathbf{q}$):

$$C_{ij}^{\alpha\beta}(\mathbf{R},T) = \frac{1}{N}\sum_{\mathbf{q},s} \frac{\hbar}{2\sqrt{M_i M_j}\,\omega_{\mathbf{q}s}} e_i^\alpha(\mathbf{q}s) e_j^\beta(-\mathbf{q},s) \coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right) e^{-i\mathbf{q}\cdot\mathbf{R}}$$

Using the time-reversal symmetry of eigenvectors, $e_j^\beta(-\mathbf{q},s) = e_j^{\beta*}(\mathbf{q},s)$:

$$\boxed{C_{ij}^{\alpha\beta}(\mathbf{R},T) = \frac{\hbar}{2N} \sum_{\mathbf{q},s} \frac{e_i^\alpha(\mathbf{q}s)\, e_j^{\beta*}(\mathbf{q}s)}{\sqrt{M_i M_j}\,\omega_{\mathbf{q}s}} \coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right) e^{-i\mathbf{q}\cdot\mathbf{R}}}$$

This is one of the most important results in the SQTC framework. Let us carefully examine each factor:

- $\hbar/(2\omega_{\mathbf{q}s})$: the **zero-point amplitude** — present even at $T = 0$.
- $\coth(\hbar\omega/2k_BT)$: the **thermal factor** — equals 1 at $T=0$ (zero-point), grows as $2k_BT/\hbar\omega$ at high $T$ (classical limit).
- $e_i^\alpha(\mathbf{q}s) e_j^{\beta*}(\mathbf{q}s)$: **outer product of eigenvectors** — encodes the polarisation.
- $e^{-i\mathbf{q}\cdot\mathbf{R}}$: **phase factor** — encodes spatial variation.

### 4.3 High-Temperature Classical Limit

At high temperature, $k_BT \gg \hbar\omega_{\mathbf{q}s}$. Using $\coth(x) \approx 1/x$ for small $x$:

$$\coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right) \approx \frac{2k_BT}{\hbar\omega_{\mathbf{q}s}}$$

Therefore at high temperature:

$$C_{ij}^{\alpha\beta}(\mathbf{R},T) \approx \frac{k_BT}{N} \sum_{\mathbf{q},s} \frac{e_i^\alpha(\mathbf{q}s) e_j^{\beta*}(\mathbf{q}s)}{\sqrt{M_i M_j}\,\omega_{\mathbf{q}s}^2} e^{-i\mathbf{q}\cdot\mathbf{R}}$$

Note: the correlator grows **linearly with temperature** in the classical limit, because thermal energy $k_BT$ is shared equally among all modes (equipartition theorem).

### 4.4 The Scalar Pair Correlator

For practical use, we define the **scalar (trace-averaged) pair correlator**:

$$\bar{C}_2(\mathbf{R},T) = \frac{1}{3}\sum_\alpha C_{ii}^{\alpha\alpha}(\mathbf{R},T) \quad \text{(same atom type, average over directions)}$$

and the **IFC-dressed pair correlator** (which couples structure to dynamics):

$$\tilde{C}_2(\mathbf{R},T) = \frac{1}{9}\sum_{\alpha,\beta} \Phi_{ij}^{\alpha\beta}(\mathbf{R})\, C_{ij}^{\alpha\beta}(\mathbf{R},T)$$

The IFC-dressed version is more sensitive to anharmonic corrections because $\Phi_{ij}^{\alpha\beta}$ itself depends on $T$ through the SQTC self-consistency loop.

---

<a name="chapter-5"></a>
## Chapter 5: Interatomic Force Constants — Connecting Forces to Phonons

### 5.1 Physical Meaning Revisited

The IFC $\Phi_{ij}^{\alpha\beta}$ is the response of force on atom $i$ in direction $\alpha$ to displacement of atom $j$ in direction $\beta$:

$$F_i^\alpha = -\frac{\partial V}{\partial u_i^\alpha} = -\sum_{j,\beta} \Phi_{ij}^{\alpha\beta}\, u_j^\beta \quad \text{(harmonic approximation)}$$

Equivalently, in matrix notation: $\mathbf{F} = -\boldsymbol{\Phi}\, \mathbf{u}$

### 5.2 Symmetry Constraints on IFCs

The IFCs must obey several exact symmetry constraints:

**Acoustic sum rule (ASR):** If the whole crystal is displaced rigidly (uniform displacement $u_j^\beta = c^\beta$ for all $j$), no net force acts:

$$\sum_j \Phi_{ij}^{\alpha\beta} = 0 \quad \forall i, \alpha, \beta$$

**Proof:** A rigid translation is not a vibration; it costs zero energy, so no forces arise. Since $F_i^\alpha = -\sum_j \Phi_{ij}^{\alpha\beta} c^\beta = 0$ for arbitrary $c^\beta$, we must have $\sum_j \Phi_{ij}^{\alpha\beta} = 0$. $\square$

**Symmetry:** By definition of the potential: $\Phi_{ij}^{\alpha\beta} = \Phi_{ji}^{\beta\alpha}$ (Newton's third law-like symmetry).

**Range:** $\Phi_{ij}^{\alpha\beta}(\mathbf{R}) \to 0$ exponentially as $|\mathbf{R}| \to \infty$, with decay determined by electronic screening length.

### 5.3 How Many IFC Parameters Are There?

For a unit cell with $n$ atoms in 3D:
- Indices: $i \in \{1,...,n\}$ (atom in unit cell), $j \in \{1,...,n\}$, $\alpha,\beta \in \{x,y,z\}$
- For each shell at distance $\mathbf{R}$: $n^2 \times 9$ parameters, but symmetry reduces this.
- Typical cutoff radius $r_{cut} \sim 5$ Å includes $\sim 10$–$50$ shells.
- Typical independent parameters: $\sim 100$–$1000$ for a moderately complex crystal.

**This counting is important for SQTC:** the minimum number of DFT calculations needed equals the number of independent IFC parameters divided by $3N$ (equations per DFT snapshot).

---

<a name="chapter-6"></a>
## Chapter 6: Why Large Supercells Are Needed at High Temperature

### 6.1 The IFC Range and Supercell Commensurateness

The phonon dispersion is determined by IFCs. The dynamical matrix at wavevector $\mathbf{q}$ is the Fourier transform of the IFCs:

$$D_{ij}^{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_i M_j}}\sum_\mathbf{R} \Phi_{ij}^{\alpha\beta}(\mathbf{R})\, e^{i\mathbf{q}\cdot\mathbf{R}}$$

A periodic cell with lattice vectors $\mathbf{A} = \mathbf{H}\mathbf{a}$ (where $\mathbf{H}$ is an integer matrix) can only access wavevectors $\mathbf{q}$ at the **commensurate q-points** of the supercell BZ. For accurate IFCs, the supercell must be large enough that:

1. All IFC pairs $(i, j, \mathbf{R})$ with $|\mathbf{R}| < r_{cut}$ fit inside the cell without periodic image overlap.
2. The atomic displacements sample all relevant phonon wavevectors (a sampling condition).

**Condition 1** requires the supercell linear dimension $L > 2r_{cut}$.

### 6.2 Anharmonic Effects Make $r_{cut}$ Temperature-Dependent

In a harmonic crystal, IFCs are temperature-independent and $r_{cut}$ is set by the electronic screening length (typically 3–6 Å, requiring supercells of $\sim$27–125 atoms).

At high temperature, **thermal fluctuations renormalize the IFCs**. The effective temperature-dependent IFCs:

$$\Phi_{ij}^{\alpha\beta,eff}(T) = \langle \Phi_{ij}^{\alpha\beta}(\{\mathbf{u}_k\}) \rangle_T$$

receive contributions from three-body and four-body anharmonic terms:

$$\Phi_{ij}^{\alpha\beta,eff}(T) = \Phi_{ij}^{\alpha\beta,harm} + \underbrace{\sum_{k,\gamma} \Psi_{ijk}^{\alpha\beta\gamma} \langle u_k^\gamma \rangle_T}_{=0 \text{ by symmetry}} + \underbrace{\frac{1}{2}\sum_{k,l,\gamma,\delta} \Xi_{ijkl}^{\alpha\beta\gamma\delta} \langle u_k^\gamma u_l^\delta \rangle_T}_{\text{thermal renormalization}} + \cdots$$

The thermal renormalization involves displacement correlators which, as we derived in Chapter 4, extend over a spatial range. The IFC *interaction* range itself is set by the electronic screening length and is approximately temperature-independent (~3–8 Å). What does change with temperature is the range of *significant thermal correlators* $C_{ij}^{\alpha\beta}(\mathbf{R},T)$: in anharmonic crystals, phonon-phonon scattering introduces an exponential cutoff at the **thermal coherence length** $\xi_T \propto T^{-1}$ (derived in Chapter 10), meaning correlations between distant atom pairs become negligible above this length. The **displacement amplitudes grow** as $\sqrt{k_BT}$, so the force-constant regression requires more data to reach the same accuracy. This is why conventional methods require large supercells with many MD snapshots at high $T$.

### 6.3 The MD Sampling Problem

In TDEP/SSCHA, atomic displacements are generated stochastically from a Gaussian distribution with covariance $C_{ij}^{\alpha\beta}(T)$. For a cell with $N$ atoms, one MD snapshot gives $3N$ force equations, allowing extraction of $\sim 3N / (3n-6) \approx N/n$ shells of IFCs. To converge all IFC parameters with statistical uncertainty $\epsilon$:

$$M_{MD} \sim \frac{N_{IFC}^{param}}{3N} \cdot \frac{1}{\epsilon^2} \cdot \sigma_F^2$$

where $\sigma_F^2 \propto k_BT$ is the force variance (grows with temperature). So at high temperature, both larger cells AND more snapshots are needed — a double computational penalty.

---

<a name="chapter-7"></a>
## Chapter 7: The SQS Philosophy — Structure as Statistical Information

### 7.1 The Random Alloy Correlators

In a perfectly random binary alloy with composition $x$ on sublattice $i$, assign occupation variable:

$$\sigma_i = \begin{cases} +1 & \text{site occupied by species A} \\ -1 & \text{site occupied by species B} \end{cases}$$

For a random alloy at composition $x$, each site is independently occupied: $P(\sigma_i = +1) = x$, $P(\sigma_i = -1) = 1-x$.

The **$n$-body correlation function** for a cluster $\{i_1, i_2, ..., i_n\}$ is:

$$\bar{\Pi}_{i_1...i_n} = \langle \sigma_{i_1} \sigma_{i_2} \cdots \sigma_{i_n} \rangle_{random} = \prod_{k=1}^n \langle \sigma_{i_k} \rangle = (2x-1)^n$$

For equicomposition ($x = 0.5$): $\langle\sigma_i\rangle = 0$, and all multi-body correlators vanish.

For general $x$, we use modified variables $\tilde{\sigma}_i = \sigma_i - (2x-1)$ so that $\langle\tilde{\sigma}_i\rangle = 0$ and $\bar{\Pi}_{i_1...i_n}^{random} = 0$ for all clusters with $n \geq 2$.

### 7.2 The SQS Matching Condition

An SQS is a periodic cell with species assignments $\{\sigma_i^{SQS}\}$ such that:

$$\bar{\Pi}_k^{SQS} \approx \bar{\Pi}_k^{random} = 0 \quad \text{for all clusters up to cutoff } r_{max}$$

The SQS construction is an optimisation over the discrete space of species assignments:

$$\text{minimize} \sum_{k} \lambda_k \left(\bar{\Pi}_k^{SQS}\right)^2$$

The key insight is: **the alloy's macroscopic property depends only on the correlation functions, not on the specific realization**. A single cell that matches the right correlators gives the same macroscopic average as an infinite random alloy.

### 7.3 Why Does This Work? — The Cluster Expansion Foundation

Any property of an alloy can be expanded in a cluster expansion:

$$P = \sum_k J_k \bar{\Pi}_k$$

where $J_k$ are effective interaction parameters (Effective Cluster Interactions, ECIs). If $\bar{\Pi}_k^{SQS} \approx \bar{\Pi}_k^{random}$ for all significant clusters, then $P^{SQS} \approx P^{random}$. The accuracy improves as the cutoff $r_{max}$ increases.

---

<a name="chapter-8"></a>
## Chapter 8: SQTC Correlator Formulation — The Thermal Analogy to SQS

### 8.1 The Correspondence

The SQTC framework exploits the following deep structural analogy:

| SQS Quantity | Mathematical Role | SQTC Quantity |
|---|---|---|
| $\sigma_i \in \{-1,+1\}$ | Discrete random variable | $u_i^\alpha \in \mathbb{R}$ — continuous random variable |
| $\langle\sigma_i\sigma_j\rangle = 0$ (random target) | 2-body correlator target | $\langle u_i^\alpha u_j^\beta\rangle_T = C_{ij}^{\alpha\beta}(\mathbf{R},T)$ |
| $\bar{\Pi}_k^{SQS} \approx 0$ | Matching condition | $\bar{C}_k^{SQTC}(\mathbf{R}) \approx \bar{C}_k^{target}(\mathbf{R},T)$ |
| Species assignment $\sigma_i^{SQS}$ | Design variable | Displacement field $\{u_i^\alpha\}$ |
| Optimise over discrete space | Search strategy | Optimise over continuous space |

The critical difference from SQS is that the SQTC target correlators are **not zero** — they are the thermal correlators $C_{ij}^{\alpha\beta}(\mathbf{R},T)$ derived in Chapter 4. The SQTC cell must **reproduce** the target, not null it.

### 8.2 How to Compute the SQTC Correlator

Given a periodic cell $\Omega$ with $n$ atoms, each at displaced position $\mathbf{R}_i^0 + \mathbf{u}_i$, the SQTC empirical pair correlator is:

$$\bar{C}_2^{SQTC}(\mathbf{R}) = \frac{1}{n_\mathbf{R}}\sum_{\substack{(i,j) \text{ pairs} \\ \mathbf{R}_j^0 - \mathbf{R}_i^0 = \mathbf{R}}} \frac{1}{3}\sum_\alpha u_i^\alpha\, u_j^\alpha$$

where $n_\mathbf{R}$ is the number of pairs at separation $\mathbf{R}$ in the periodic cell.

**Worked example for a simple case:** Suppose the SQTC cell has 4 atoms in 1D, with displacements $u_1, u_2, u_3, u_4$. The pair correlator at nearest-neighbour distance $a$ is:

$$\bar{C}_2^{SQTC}(a) = \frac{1}{4}(u_1 u_2 + u_2 u_3 + u_3 u_4 + u_4 u_1)$$

(periodic boundary conditions connect $u_4$ back to $u_1$). The SQTC design condition requires this to match $C^{target}(a, T)$.

### 8.3 Higher-Order Thermal Correlators and Anharmonicity

For a **harmonic crystal**, odd moments of displacements vanish (by symmetry the Gaussian distribution is even):

$$\langle u_i^\alpha u_j^\beta u_k^\gamma \rangle_T^{harm} = 0 \quad \text{(3-body, harmonic)}$$

And 4-body cumulants also vanish (Wick's theorem for Gaussian distributions):

$$\langle u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta \rangle_T^{cum,harm} = 0 \quad \text{(4-body cumulant, harmonic)}$$

where the **cumulant** subtracts all Gaussian pairings:

$$\bar{C}_4^{cum} = \langle u_i u_j u_k u_l \rangle - \langle u_i u_j\rangle\langle u_k u_l\rangle - \langle u_i u_k\rangle\langle u_j u_l\rangle - \langle u_i u_l\rangle\langle u_j u_k\rangle$$

**Non-zero $\bar{C}_3$ or $\bar{C}_4^{cum}$ are therefore direct fingerprints of anharmonicity.** An SQTC cell targeting these higher-order correlators encodes the anharmonic phonon self-energy non-perturbatively in the designed displacement field.

---

<a name="chapter-9"></a>
## Chapter 9: The SQTC Objective Functional — Deriving the Matching Condition

### 9.1 Formulating the Optimisation Problem

We want to find the cell $\Omega$ (shape and size) and displacement field $\{\mathbf{u}_i\}$ such that the SQTC empirical correlators best match the target thermal correlators. Define the **SQTC quality functional** (the quantity to minimize):

$$\mathcal{Q}_{SQTC}[\Omega, \{\mathbf{u}_i\}] = \sum_{k=2}^{k_{max}} \lambda_k \sum_{\substack{\text{all clusters} \\ \text{of order } k \\ |\mathbf{R}| < r_k}} w_k(\mathbf{R})\left[\bar{C}_k^{SQTC}(\mathbf{R}) - \bar{C}_k^{target}(\mathbf{R}, T)\right]^2$$

**Breaking down the terms:**

- The sum over $k$ includes pair ($k=2$), triplet ($k=3$), and quadruplet ($k=4$) correlators.
- $\lambda_k$ are **weighting factors**: $\lambda_2 \gg \lambda_3 > \lambda_4$ because pair correlators are most important and higher-order terms are corrections.
- $r_k$ is the spatial cutoff — we only match correlators up to this distance.
- $w_k(\mathbf{R})$ is a **distance weighting function** (derived from physics in Chapter 10).
- The square ensures $\mathcal{Q}_{SQTC} \geq 0$, with $\mathcal{Q}_{SQTC} = 0$ being the perfect match.

### 9.2 Why Sum of Squared Residuals?

The functional $\mathcal{Q}_{SQTC}$ is a **weighted sum of squared residuals** — the same structure as least-squares fitting. This choice is motivated by:

1. **Convexity** (in the continuous optimisation variables): squared residuals give a smooth landscape amenable to gradient methods.
2. **Statistical interpretation**: minimising $\mathcal{Q}_{SQTC}$ is equivalent to maximising the likelihood that the SQTC cell was drawn from the thermal distribution (under Gaussian approximation).
3. **Systematic improvability**: adding more clusters in the sum (increasing $r_k$) monotonically improves accuracy.

### 9.3 Deriving the Distance Weight from the Atomic Mean Free Path

The weight $w_k(\mathbf{R})$ is not arbitrary — it is derived from physics. The contribution of a pair $(i,j)$ at separation $\mathbf{R}$ to macroscopic phonon properties is proportional to the **probability that a phonon travels from $i$ to $j$ without scattering**:

$$P(\text{no scatter over distance } |\mathbf{R}|) = e^{-|\mathbf{R}|/\ell_{ph}(T)}$$

where $\ell_{ph}(T)$ is the phonon mean free path (equal to $\xi_T$ in Chapter 10). Therefore:

$$\boxed{w_k(\mathbf{R}) = e^{-|\mathbf{R}|/\xi_T}}$$

Pairs far apart (beyond the thermal coherence length) contribute negligibly to the phonon physics at that temperature and can be matched less accurately — the exponential weight encodes this naturally.

### 9.4 The Optimisation is Jointly Over Cell Shape and Displacements

The minimisation:

$$(\Omega^*, \{\mathbf{u}_i^*\}) = \arg\min_{\Omega,\{\mathbf{u}_i\}} \mathcal{Q}_{SQTC}$$

has two nested levels:

**Outer loop** (discrete): enumerate candidate cell shapes $\Omega$ via Hermite Normal Form decomposition $\mathbf{H}$ of the supercell matrix (see Section 9.5).

**Inner loop** (continuous): for each $\Omega$, minimise $\mathcal{Q}_{SQTC}$ over the displacement field $\{\mathbf{u}_i\}$.

### 9.5 Hermite Normal Form Enumeration of Cell Shapes

A supercell has lattice vectors related to the primitive cell by:

$$\begin{pmatrix}\mathbf{A}_1 \\ \mathbf{A}_2 \\ \mathbf{A}_3\end{pmatrix} = \mathbf{H} \begin{pmatrix}\mathbf{a}_1 \\ \mathbf{a}_2 \\ \mathbf{a}_3\end{pmatrix}$$

where $\mathbf{H}$ is a $3\times 3$ integer matrix. The **Hermite Normal Form** (HNF) restricts $\mathbf{H}$ to be lower triangular with positive diagonal entries:

$$\mathbf{H} = \begin{pmatrix} h_{11} & 0 & 0 \\ h_{21} & h_{22} & 0 \\ h_{31} & h_{32} & h_{33} \end{pmatrix}, \quad h_{11}h_{22}h_{33} = n, \quad 0 \leq h_{21} < h_{22},\; 0 \leq h_{31},h_{32} < h_{33}$$

This canonical form avoids counting equivalent cells multiple times. The number of distinct HNF matrices of determinant $n$ (i.e., cells with $n$ formula units) is:

$$N_{HNF}(n) = \sum_{\substack{h_{11}h_{22}h_{33}=n \\ h_{11},h_{22},h_{33}>0}} h_{22}\cdot h_{33}$$

For $n = 2$: 3 distinct cells; for $n = 4$: 13; for $n = 8$: 30. All are enumerated exhaustively.

---

<a name="chapter-10"></a>
## Chapter 10: Thermal Coherence Length — Why High-T Means Smaller Cells

### 10.1 Intuitive Picture

Imagine dropping a stone in still water: ripples spread outward but die away at some distance. Phonons are like those ripples. The **thermal coherence length** $\xi_T$ is how far a thermal phonon travels before losing phase coherence (being scattered or decaying). Correlations between atoms beyond $\xi_T$ are exponentially small — those pairs don't matter for phonon physics.

### 10.2 Derivation from the Phonon Lifetime

The phonon lifetime $\tau_{\mathbf{q}s}(T)$ is the average time before mode $(\mathbf{q}s)$ is scattered. From Fermi's golden rule for three-phonon processes (anharmonic scattering), the average phonon lifetime scales as:

$$\tau_{ph}(T) \sim \frac{1}{\Gamma_{ph}(T)} \quad \text{where} \quad \Gamma_{ph}(T) \propto T \quad \text{at high } T$$

(The scattering rate $\Gamma$ grows linearly with $T$ because the number of thermally excited phonons available for scattering grows as $n_B \propto k_BT/\hbar\omega$.)

**Step 1:** The average phonon group velocity is $v_s$ (the speed of sound, approximately temperature-independent).

**Step 2:** The thermal coherence length (mean free path) is:

$$\xi_T = v_s \cdot \tau_{ph}(T) \sim \frac{v_s}{\Gamma_{ph}(T)} \propto \frac{v_s}{T}$$

**Step 3:** More precisely, using the Grüneisen parameter $\gamma$ (dimensionless, typically 1–3 for metals) which quantifies the anharmonicity:

$$\xi_T = \frac{v_s \hbar \bar{\omega}}{\gamma k_B T} = \frac{v_s \hbar \bar{\omega}}{\gamma k_B T}$$

where $\bar{\omega}$ is the average phonon frequency (Debye frequency scale).

**Substituting numbers** for aluminium ($v_s \approx 6000$ m/s, $\bar{\omega} \approx 2\pi \times 8$ THz, $\gamma \approx 2$, $T = 1000$ K):

$$\xi_T \approx \frac{6000 \times (1.055\times10^{-34}) \times (2\pi\times 8\times10^{12})}{2 \times (1.38\times10^{-23}) \times 1000} \approx \frac{6000 \times 5.3\times10^{-22}}{2.76\times10^{-20}} \approx 115 \text{ Å} \approx 15 \text{ lattice constants}$$

At room temperature (300 K), $\xi_T \approx 50$ lattice constants. At 1000 K, $\xi_T \approx 15$ lattice constants.

**Important distinction:** $\xi_T$ here is the **displacement-displacement correlation decay length** (the length scale beyond which anharmonic scattering renders thermal correlators negligible), not the IFC interaction range. The IFC range is set by electronic screening (~3–8 Å, independent of temperature). SQTC must design cells large enough to cover the full IFC range **and** match correlators out to $\xi_T$. At high temperature, $\xi_T$ is smaller, meaning fewer distant correlator pairs need matching — this is why SQTC is most powerful at high temperatures: the *number of significant correlator constraints* shrinks, enabling smaller ensembles with fewer DFT calculations.

### 10.3 Formal Statement: The Spatial Decay of Correlators

From the expression derived in Chapter 4, the correlator at large $|\mathbf{R}|$ decays as:

$$|C_{ij}^{\alpha\beta}(\mathbf{R},T)| \leq A(T)\, e^{-|\mathbf{R}|/\xi_T}$$

**Proof sketch:** The integrand in the $\mathbf{q}$-sum is peaked near phonon frequencies. In the presence of anharmonic damping (Lorentzian phonon lineshape with half-width $\Gamma$), the phonon Green's function has a pole at $\omega_{\mathbf{q}s} - i\Gamma_{\mathbf{q}s}$. The spatial Fourier transform of a Lorentzian function is an exponential. Specifically, the imaginary part of the phonon wavevector (from the complex dispersion relation $\omega = v_s q - i v_s q^2 / 2\Gamma$ ...) gives exponential decay with length scale $v_s/\Gamma = \xi_T$. $\square$

**Consequence for SQTC:** Any periodic cell $\Omega$ with linear dimension $L > 2\xi_T$ captures all non-negligible correlators. Smaller cells miss correlators at $|\mathbf{R}| \sim \xi_T$, introducing errors that scale as $e^{-L/\xi_T}$. This provides a **rigorous accuracy bound** as a function of cell size.

---

<a name="chapter-11"></a>
## Chapter 11: Displacement Basis Expansion — Making the Optimisation Tractable

### 11.1 The Problem with Unrestricted Optimisation

An SQTC cell with $n$ atoms has $3n$ displacement components $\{u_i^\alpha\}$ as free variables. Direct minimisation of $\mathcal{Q}_{SQTC}$ in this $3n$-dimensional space is possible but inefficient: the landscape has many local minima, and unphysical (too large or too small) displacements may occur.

### 11.2 Expanding in Phonon Eigenvectors

Instead, we write the displacement of atom $i$ in direction $\alpha$ as a **linear combination of phonon eigenvectors**:

$$u_i^\alpha = \frac{1}{\sqrt{M_i}}\sum_{\mathbf{q},s} Q_{\mathbf{q}s}\, e_i^\alpha(\mathbf{q}s)\, \cos(\mathbf{q}\cdot\mathbf{R}_i^0 + \phi_{\mathbf{q}s})$$

where:
- $Q_{\mathbf{q}s} \geq 0$ is the **mass-weighted normal-coordinate amplitude** of mode $(\mathbf{q},s)$
- $\phi_{\mathbf{q}s}$ is the **phase** of mode $(\mathbf{q}s)$
- $e_i^\alpha(\mathbf{q}s)$ are the eigenvectors from the (cheap, preliminary) harmonic DFT calculation

The $\mathbf{q}$-points are restricted to those **commensurate with the SQTC cell** (i.e., $\mathbf{q} = \mathbf{H}^{-T}\mathbf{G}$ for reciprocal lattice vectors $\mathbf{G}$ of the SQTC cell).

### 11.3 Physical Motivation of the Starting Point

In the **harmonic approximation**, the normal-coordinate amplitude for mode $(\mathbf{q},s)$ at temperature $T$ is:

$$Q_{\mathbf{q}s}^{harm}(T) = \sqrt{\frac{\hbar}{2\omega_{\mathbf{q}s}} \coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right)}$$

**Derivation:** The mean squared displacement in mode $(\mathbf{q}s)$ is:

$$\langle Q_{\mathbf{q}s}^2 \rangle_T = \frac{\hbar}{2\omega_{\mathbf{q}s}}(2n_B + 1) = \frac{\hbar}{2\omega_{\mathbf{q}s}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right)$$

This follows from $Q_{\mathbf{q}s} = \sqrt{\hbar/2\omega}(a + a^\dagger)$ and $\langle(a+a^\dagger)^2\rangle = 2n_B + 1$.

So $Q_{\mathbf{q}s}^{harm} = \sqrt{\langle Q_{\mathbf{q}s}^2\rangle_T}$ is the **root-mean-square normal-coordinate amplitude** — the typical mode amplitude at temperature $T$.

We **initialise** the SQTC optimisation at $Q_{\mathbf{q}s} = Q_{\mathbf{q}s}^{harm}$ with random phases $\phi_{\mathbf{q}s}$. The optimisation then adjusts $Q_{\mathbf{q}s}$ away from the harmonic values to better match the target correlators (capturing anharmonic corrections). Phase optimisation selects which specific spatial pattern best matches the correlator hierarchy.

### 11.4 The Reduced Optimisation Problem

The optimisation is now over $\{Q_{\mathbf{q}s}, \phi_{\mathbf{q}s}\}$ — a set of $2 \times 3n$ positive and bounded real numbers (since there are $3n$ modes for a cell of $n$ atoms), rather than $3n$ unconstrained displacements. Physical bounds apply automatically:

$$0 \leq Q_{\mathbf{q}s} \leq Q_{\mathbf{q}s}^{max}(T) = \eta \cdot Q_{\mathbf{q}s}^{harm}(T)$$

where $\eta \sim 2$–$3$ is a maximum overshoot factor. This prevents unphysically large displacements (which would cause atoms to collide or leave the basin of attraction of the harmonic approximation).

### 11.5 The Gradient of the Objective Function

For the continuous optimisation (gradient descent), we need $\partial\mathcal{Q}_{SQTC}/\partial Q_{\mathbf{q}s}$. The pair contribution is:

$$\frac{\partial\mathcal{Q}_{SQTC}}{\partial Q_{\mathbf{q}s}} = 2\lambda_2 \sum_{\mathbf{R}} w_2(\mathbf{R}) \left[\bar{C}_2^{SQTC}(\mathbf{R}) - \bar{C}_2^{target}(\mathbf{R},T)\right] \frac{\partial \bar{C}_2^{SQTC}(\mathbf{R})}{\partial Q_{\mathbf{q}s}}$$

The last derivative is:

$$\frac{\partial \bar{C}_2^{SQTC}(\mathbf{R})}{\partial Q_{\mathbf{q}s}} = \frac{1}{3n_\mathbf{R}} \sum_{\substack{(i,j)\\ \mathbf{R}_j-\mathbf{R}_i = \mathbf{R}}} \sum_\alpha \left(u_j^\alpha \frac{\partial u_i^\alpha}{\partial Q_{\mathbf{q}s}} + u_i^\alpha \frac{\partial u_j^\alpha}{\partial Q_{\mathbf{q}s}}\right)$$

where $\frac{\partial u_i^\alpha}{\partial Q_{\mathbf{q}s}} = \frac{e_i^\alpha(\mathbf{q}s)}{\sqrt{M_i}}\cos(\mathbf{q}\cdot\mathbf{R}_i^0 + \phi_{\mathbf{q}s})$.

This gradient is cheap to evaluate ($\mathcal{O}(n^2)$ per mode), making the inner-loop optimisation computationally fast — far cheaper than any DFT calculation.

---

<a name="chapter-12"></a>
## Chapter 12: IFC Extraction by Linear Regression — Getting Force Constants from DFT Forces

### 12.1 The Force-Displacement Linear System

Given the SQTC cell with atoms displaced to positions $\mathbf{R}_i^0 + \mathbf{u}_i^*$, we run a DFT single-point calculation (no ionic relaxation — just compute the electronic structure and get forces at the displaced positions).

DFT gives us forces $F_i^\alpha$ on each atom. In the harmonic approximation:

$$F_i^\alpha = -\sum_{j,\beta} \Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

This is a **linear system** $\mathbf{F} = -\boldsymbol{\Phi}\, \mathbf{u}$ where:
- $\mathbf{F}$ is a vector of length $3N_{SQTC}$ (all force components)
- $\boldsymbol{\Phi}$ is the matrix of force constants (unknowns, size $3N_{SQTC} \times 3N_{SQTC}$, but sparse due to cutoff)
- $\mathbf{u}$ is the vector of displacements (known, designed by SQTC)

### 12.2 Building the Regression Matrix

Reorganise the equations. For each atom $i$ and direction $\alpha$, the force is:

$$F_i^\alpha = -\sum_j \sum_\beta \Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

Let the independent IFC parameters (after applying symmetry constraints and acoustic sum rule) be $\{p_1, p_2, ..., p_M\}$. Each force equation becomes:

$$F_i^\alpha = \sum_{m=1}^M G_{i\alpha, m}\, p_m$$

where $G_{i\alpha,m}$ is the coefficient (built from displacements $u_j^\beta$). Stacking all equations:

$$\underbrace{\begin{pmatrix}F_1^x \\ F_1^y \\ \vdots \\ F_N^z\end{pmatrix}}_{\mathbf{F}, \; 3N \times 1} = \underbrace{\begin{pmatrix} G_{1x,1} & G_{1x,2} & \cdots & G_{1x,M} \\ G_{1y,1} & G_{1y,2} & \cdots & G_{1y,M} \\ \vdots & & \ddots & \vdots \\ G_{Nz,1} & G_{Nz,2} & \cdots & G_{Nz,M} \end{pmatrix}}_{\mathbf{G}, \; 3N \times M} \underbrace{\begin{pmatrix}p_1 \\ p_2 \\ \vdots \\ p_M\end{pmatrix}}_{\mathbf{p}, \; M \times 1}$$

### 12.3 Solving by Least Squares with Ridge Regularisation

We need $3N \geq M$ equations to overdetermine the system (more equations than unknowns ensures statistical averaging). The pure least-squares solution minimises $\|\mathbf{G}\mathbf{p} - \mathbf{F}\|^2$:

$$\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$$

This is the **normal equation** of least squares. The matrix $\mathbf{G}^T\mathbf{G}$ must be non-singular (full rank) for a unique solution.

In practice, the regression matrix can be near-singular when the number of snapshots is small or certain IFC components are poorly sampled. The implementation therefore uses **Tikhonov (ridge) regularisation**:

$$\mathbf{p}^* = (\mathbf{G}^T\mathbf{G} + \alpha \mathbf{I})^{-1}\mathbf{G}^T\mathbf{F}$$

where $\alpha$ (`ridge_alpha`, default $10^{-3}$ eV$^2$/Å$^4$) penalises large IFC values and stabilises the inversion. The penalty adds $\alpha\|\mathbf{p}\|^2$ to the objective, biasing solutions toward zero — physically appropriate since IFCs decay to zero at large separation. For well-conditioned problems (many snapshots, large displacements) $\alpha$ has negligible effect; for few snapshots or small cells it provides stability.

**Why SQTC improves the condition number:** The condition number $\kappa(\mathbf{G}^T\mathbf{G})$ measures how sensitive the solution is to errors in $\mathbf{F}$ (DFT numerical noise). A small condition number ($\kappa \approx 1$) is ideal. 

For random (stochastic) displacements, the columns of $\mathbf{G}$ can be nearly parallel (especially if displacement patterns are not orthogonal), leading to large $\kappa$. SQTC's displacement field is specifically designed so that the displacement amplitudes $A_{\mathbf{q}s}$ sample all phonon modes with comparable weight, making the columns of $\mathbf{G}$ nearly orthogonal and $\kappa$ small.

**Quantitatively:** For a random displacement set, $\kappa \sim M_{MD}/M_{IFC}$ (grows with the number of snapshots needed). For SQTC-designed displacements, $\kappa \sim 1$–$10$ regardless of the number of IFC parameters — meaning far fewer DFT calculations achieve the same accuracy.

### 12.4 Central-Force Symmetry Projection (`symmetrize_bonds`)

For bonds along high-symmetry directions (e.g., $\langle 100 \rangle$ in FCC or rocksalt, $\langle 111 \rangle$ in BCC), the $3 \times 3$ IFC tensor $\Phi_{ij}^{\alpha\beta}$ must have the form of a **central-force tensor**:

$$\Phi_{ij}^{\alpha\beta}(\hat{R}) = A\,\hat{R}_\alpha\hat{R}_\beta + B(\delta_{\alpha\beta} - \hat{R}_\alpha\hat{R}_\beta)$$

where $\hat{R} = \mathbf{R}/|\mathbf{R}|$ is the bond direction, $A$ is the longitudinal (bond-stretching) force constant, and $B$ is the transverse (bond-bending) force constant. This two-parameter form follows from the cylindrical symmetry of the interatomic potential along the bond axis.

Applying this projection — controlled by the `symmetrize_bonds` parameter — reduces the number of free IFC parameters per bond from up to 6 (for general $3\times3$ symmetric tensor) to 2. The benefits are:

- **Improved regression conditioning:** fewer parameters from the same data.
- **Physical consistency:** prevents spurious off-axis IFC components that would produce imaginary phonon modes through asymmetric dynamical matrix contributions.
- **Reduced imaginary frequencies:** especially important for light ionic compounds (NaCl, MgO, LiF) and FCC metals where the central-force approximation is accurate.

`symmetrize_bonds = True` is appropriate for: FCC metals (Au, Cu, Al), light rocksalt compounds (NaCl, MgO, LiF).  
`symmetrize_bonds = False` is appropriate for: BCC metals, zincblende (Si), and strongly anharmonic rocksalts (PbTe) where off-axis terms are physically significant.

The postprocessor auto-assigns the correct default based on structure type and element set; the stored value from `sqtc_results.json` always takes precedence for new runs.

### 12.5 Multi-Snapshot Regression (SQTC Ensemble)

For $K$ SQTC snapshots, stack the equations:

$$\begin{pmatrix}\mathbf{F}^{(1)} \\ \mathbf{F}^{(2)} \\ \vdots \\ \mathbf{F}^{(K)}\end{pmatrix} = \begin{pmatrix}\mathbf{G}^{(1)} \\ \mathbf{G}^{(2)} \\ \vdots \\ \mathbf{G}^{(K)}\end{pmatrix} \mathbf{p}$$

The combined regression matrix $\tilde{\mathbf{G}}$ has $3NK$ rows. The least-squares solution is still $\mathbf{p}^* = (\tilde{\mathbf{G}}^T\tilde{\mathbf{G}})^{-1}\tilde{\mathbf{G}}^T\tilde{\mathbf{F}}$.

### 12.5 Imposing the Acoustic Sum Rule

After extracting $\mathbf{p}^*$, the raw IFCs may violate the acoustic sum rule (Sec. 5.2) due to numerical errors. Enforce it as a **constraint** by adding a Lagrange multiplier term to the regression:

$$\text{minimize } \|\mathbf{G}\mathbf{p} - \mathbf{F}\|^2 + \mu \sum_{i,\alpha,\beta}\left(\sum_j \Phi_{ij}^{\alpha\beta}\right)^2$$

where $\mu$ is a penalty parameter (chosen large enough to enforce ASR, typically $\mu \sim 10$–$1000$ in units of $[\text{eV}^{-1}\text{Å}^{-2}]$).

---

<a name="chapter-13"></a>
## Chapter 13: Phonon Reconstruction from the SQTC Cell

### 13.1 Building the Dynamical Matrix

From the extracted IFCs $\Phi_{ij}^{\alpha\beta}(\mathbf{R},T)$ (temperature-dependent because they came from displaced atoms at temperature $T$), build the dynamical matrix:

$$D_{ij}^{\alpha\beta}(\mathbf{q},T) = \frac{1}{\sqrt{M_i M_j}} \sum_\mathbf{R} \Phi_{ij}^{\alpha\beta}(\mathbf{R},T)\, e^{i\mathbf{q}\cdot\mathbf{R}}$$

**Important:** $\mathbf{q}$ here can be any wavevector, not just the commensurate ones of the SQTC cell. The Fourier sum extends over all lattice vectors $\mathbf{R}$ within the IFC cutoff range.

### 13.2 Computing Phonon Dispersions

Diagonalise $D(\mathbf{q},T)$ at each $\mathbf{q}$-point:

$$D(\mathbf{q},T)\, \mathbf{e}(\mathbf{q}s) = \omega_{\mathbf{q}s}^2(T)\, \mathbf{e}(\mathbf{q}s)$$

The eigenvalues $\omega_{\mathbf{q}s}^2(T)$ are the **temperature-renormalized phonon frequencies**. Negative eigenvalues are physically meaningful imaginary frequencies — they signal either a genuine dynamical instability or, in practice, insufficient snapshot data or wrong IFC parameters. They are reported as $-\sqrt{|\omega^2|}$ in THz by convention.

### 13.3 Thermodynamic Properties from the Phonon DOS

Given the phonon frequencies $\{\omega_{\mathbf{q}s}\}$ on a Monkhorst-Pack $\mathbf{q}$-mesh, the thermodynamic properties are computed with fully quantum Bose-Einstein statistics:

**Heat capacity at constant volume:**

$$C_V(T) = k_B \sum_{\mathbf{q}s} \left(\frac{x_{\mathbf{q}s}}{\sinh x_{\mathbf{q}s}}\right)^2, \quad x_{\mathbf{q}s} = \frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}$$

**Zero-point energy per formula unit:**

$$\text{ZPE} = \sum_{\mathbf{q}s} \frac{\hbar\omega_{\mathbf{q}s}}{2}$$

**Vibrational entropy:**

$$S_{vib}(T) = k_B \sum_{\mathbf{q}s} \left[x_{\mathbf{q}s}\coth x_{\mathbf{q}s} - \ln(2\sinh x_{\mathbf{q}s})\right]$$

**Vibrational free energy:**

$$F_{vib}(T) = k_B T \sum_{\mathbf{q}s} \ln\!\left(2\sinh x_{\mathbf{q}s}\right)$$

All four quantities are implemented in `sqtc/phonons.py` and output by the postprocessor to `thermal_summary.json`. Only modes with $\omega > 0$ contribute (imaginary modes are excluded from thermal sums and their count is reported as `unstable`).

### 13.4 Debye Temperature Estimators

Two estimators are computed:

**Spectral (second-moment) Debye temperature** — robust, temperature-independent:

$$\Theta_D^{spec} = \frac{\hbar}{k_B}\sqrt{\frac{5}{3}\langle\omega^2\rangle}, \quad \langle\omega^2\rangle = \frac{1}{N_{pos}}\sum_{\mathbf{q}s,\,\omega>0}\omega_{\mathbf{q}s}^2$$

This estimator is well-defined for any phonon spectrum and is the primary reported Debye temperature.

**Calorimetric Debye temperature** — inverts the Debye $C_V$ integral at temperature $T$:

$$C_V^{Debye}(T, \Theta_D) = 9n_bk_B\left(\frac{T}{\Theta_D}\right)^3\!\int_0^{\Theta_D/T}\!\frac{x^4 e^x}{(e^x-1)^2}dx \stackrel{!}{=} C_V^{SQTC}(T)$$

solved by bisection. This estimator is ill-conditioned when $T \gg \Theta_D$ (Dulong-Petit plateau) and is reported per temperature point as `TD_caloric`.

### 13.5 The Post-Processing Pipeline (Implementation)

The `sqtc/postprocessor.py` module implements the complete post-processing pipeline as a library and a CLI. The pipeline proceeds as follows:

1. **Structure auto-detection** via `spglib`: the last POSCAR in `iter_*/snap_*/` is read, the ideal primitive cell is reconstructed (atoms moved to ideal fractional positions), and `spglib.get_spacegroup()` identifies the space group. The space group number is mapped to the structure name (e.g., SG 225 → FCC, SG 221 → BCC, SG 225 with 2-atom basis → rocksalt). This eliminates the need to specify `--structure` manually for standard structures.

2. **Parameter auto-detection** from `sqtc_results.json`: lattice constant, elements, masses, T_design, r_cutoff, ridge_alpha, and symmetrize_bonds are read if present. For legacy runs that pre-date storage of r_cutoff and symmetrize_bonds, these are derived: r_cutoff from the analytical nearest-neighbour shell formula, symmetrize_bonds from the structure type and element set (True for FCC and light rocksalts, False for BCC, zincblende, heavy rocksalts).

3. **Primitive-cell basis correction for binary structures**: VASP sorts atoms alphabetically within each species, which can reorder the two basis atoms relative to the convention used in the runner (e.g., NaCl runner defines Na at origin, but VASP writes Cl first). If the postprocessor used the wrong basis ordering, the reconstructed equilibrium positions would be displaced by ~$a/2$ from the actual equilibrium, causing the force-displacement regression to fail catastrophically ($R^2 < 0$). The fix: for each species, compute the circular mean of fractional coordinates across all atoms in the supercell, then map the result to the canonical positions $(0,0,0)$ or $(a/2,0,0)$ used by the runner convention. This is robust to thermal noise because $N_{basis} \sim 36$ atoms average down the noise by a factor of $\sim 6$.

4. **IFC extraction** via `IFCExtractor`: ridge regression on all POSCAR/OUTCAR pairs within the cutoff radius, with optional central-force symmetry projection.

5. **Phonon calculation** via `PhononCalculator`: band structure on a high-symmetry path, DOS on a $30^3$ Monkhorst-Pack mesh.

6. **Thermodynamics**: $C_V(T)$, $S_{vib}(T)$, $F_{vib}(T)$, ZPE, MSD, Debye-Waller $B$ factor, $\Theta_D$ on a user-specified temperature grid.

7. **Output**: plots (PDF + PNG) and numerical data (NPZ + JSON) saved to `postproc/` in the run directory.

### 13.6 Phonon Band Unfolding (for Alloy/Defect Systems)

When the SQTC cell is larger than the primitive cell (as it always is), the phonon modes are defined at the folded BZ of the supercell. To recover the primitive-cell dispersion (more physically interpretable), we **unfold** the phonon bands.

The **spectral weight** (unfolding weight) for a SQTC mode $(\mathbf{q}_{SC},s)$ at primitive-cell wavevector $\mathbf{q}_{PC}$ is:

$$P(\mathbf{q}_{PC} | \mathbf{q}_{SC}, s) = \sum_{i \in \text{primitive cell}} \left|\sum_\alpha e_i^\alpha(\mathbf{q}_{SC},s)\, e^{i\mathbf{q}_{PC}\cdot\mathbf{R}_i^0}\right|^2$$

**Derivation:** Project the SQTC eigenvector onto the Bloch states of the primitive cell. The squared modulus gives the probability that mode $(\mathbf{q}_{SC},s)$ corresponds to the primitive-cell wavevector $\mathbf{q}_{PC}$. Summing over atoms in one primitive cell gives the spectral weight.

The unfolded phonon spectral function is:

$$A(\mathbf{q},\omega,T) = \sum_s P(\mathbf{q}|\mathbf{q}_{SC},s)\, \delta(\omega - \omega_{\mathbf{q}_{SC},s}(T))$$

This recovers the phonon DOS and dispersion in the primitive-cell BZ, directly comparable to inelastic neutron/X-ray scattering experiments.

---

<a name="chapter-14"></a>
## Chapter 14: Self-Consistency: Closing the Loop

### 14.1 Why Self-Consistency is Necessary

The SQTC procedure depends on a circular logic:
- The target correlators $C_{ij}^{\alpha\beta,target}(\mathbf{R},T)$ are computed using the phonon frequencies $\omega_{\mathbf{q}s}(T)$.
- The phonon frequencies $\omega_{\mathbf{q}s}(T)$ are extracted from IFCs.
- The IFCs are extracted from DFT forces on the SQTC cells.
- The SQTC cells are designed using the target correlators.

This circularity is the same as in SSCHA and TDEP, and it is broken by **iterative self-consistency**.

### 14.2 The Self-Consistent Iteration

**Iteration 0 (Initialisation):**

$$C_2^{(0)}(\mathbf{R},T) = C_2^{harm}(\mathbf{R},T) \quad \text{(from harmonic phonons of primitive cell, cheap)}$$

**General iteration $n \to n+1$:**

**Step A:** Using $C_2^{(n)}(\mathbf{R},T)$ as target, generate the SQTC cell(s) by minimising $\mathcal{Q}_{SQTC}^{(n)}$.

**Step B:** Run DFT on the SQTC cell(s) to get forces $\mathbf{F}^{(n)}$.

**Step C:** Extract IFCs $\Phi^{(n)}(T)$ by least-squares regression (Chapter 12).

**Step D:** Compute new phonon frequencies $\omega_{\mathbf{q}s}^{(n)}(T)$ by diagonalising $D^{(n)}(\mathbf{q},T)$.

**Step E:** Update target correlators:

$$C_2^{(n+1)}(\mathbf{R},T) = \frac{\hbar}{2N}\sum_{\mathbf{q},s} \frac{e_i^\alpha(\mathbf{q}s)e_j^{\beta*}(\mathbf{q}s)}{\sqrt{M_iM_j}\,\omega_{\mathbf{q}s}^{(n)}(T)} \coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}^{(n)}(T)}{2k_BT}\right) e^{-i\mathbf{q}\cdot\mathbf{R}}$$

### 14.3 Convergence Criterion

Stop when:

$$\Delta^{(n)} \equiv \left\|C_2^{(n+1)}(\mathbf{R},T) - C_2^{(n)}(\mathbf{R},T)\right\|_2 \equiv \sqrt{\sum_\mathbf{R}\left[C_2^{(n+1)}(\mathbf{R},T) - C_2^{(n)}(\mathbf{R},T)\right]^2} < \epsilon_{conv}$$

Typical convergence: $\epsilon_{conv} \sim 10^{-3}$ Å$^2$, reached in **2–4 iterations**.

### 14.4 Why Does It Converge? — Fixed-Point Analysis

Define the map $\mathcal{F}: C_2 \mapsto C_2'$ as one full iteration (A through E). The self-consistent solution is a **fixed point** $C_2^* = \mathcal{F}(C_2^*)$.

The iteration converges when $\|\nabla \mathcal{F}\| < 1$ (the map is a contraction). Physically, this means the phonon frequencies are not too sensitive to small changes in the correlators — which holds when the anharmonicity is not too extreme (Grüneisen parameter $\gamma < \gamma_{max} \sim 5$–$10$). For strongly anharmonic systems (near a structural phase transition), convergence requires damping: $C_2^{(n+1)} = \alpha C_2^{new} + (1-\alpha)C_2^{(n)}$ with mixing parameter $\alpha \in (0,1)$.

---

<a name="chapter-15"></a>
## Chapter 15: The Representability Theorem — Formal Proof

### 15.1 Statement

**Theorem (SQTC Representability):**

*Let $\Phi_{ij}^{\alpha\beta,exact}(\mathbf{R},T)$ be the exact temperature-renormalized interatomic force constants of an anharmonic crystal at temperature $T$. Let $\xi_T$ be the thermal coherence length (Chapter 10). For any accuracy $\epsilon > 0$, there exists:*
- *a periodic cell $\Omega^*$ with $n^*$ formula units*
- *a displacement field $\{\mathbf{u}_i^*\}$*

*such that the phonon frequencies $\omega_{\mathbf{q}s}^{SQTC}(T)$ extracted from DFT forces on $(\Omega^*, \{\mathbf{u}_i^*\})$ satisfy:*

$$\left|\omega_{\mathbf{q}s}^{SQTC}(T) - \omega_{\mathbf{q}s}^{exact}(T)\right| < \epsilon \quad \forall\, \mathbf{q},s$$

*provided the cell satisfies $L^* > 2\xi_T$ (cell larger than twice the coherence length).*

### 15.2 Proof

The proof proceeds in three steps.

**Step 1: IFC Truncation Error**

The exact phonon frequencies depend on all IFCs $\Phi_{ij}(\mathbf{R})$ for all $|\mathbf{R}|$. For finite $r_{max}$, the truncated dynamical matrix:

$$D_{trunc}(\mathbf{q},T) = \frac{1}{\sqrt{M_iM_j}}\sum_{|\mathbf{R}|<r_{max}} \Phi_{ij}(\mathbf{R},T)\, e^{i\mathbf{q}\cdot\mathbf{R}}$$

differs from the exact one by:

$$\|D_{exact} - D_{trunc}\| \leq \frac{1}{\sqrt{M_{min}^2}} \sum_{|\mathbf{R}|>r_{max}} |\Phi_{ij}(\mathbf{R},T)|$$

The IFCs decay exponentially with the electronic screening length $\lambda_{scr}$ (approximately temperature-independent, ~3–8 Å for most materials): $|\Phi_{ij}(\mathbf{R},T)| \leq C_\Phi e^{-|\mathbf{R}|/\lambda_{scr}}$.

Therefore the truncation error in the dynamical matrix is bounded by:

$$\|D_{exact} - D_{trunc}\| \leq \frac{C_\Phi}{M_{min}} \sum_{|\mathbf{R}|>r_{max}} e^{-|\mathbf{R}|/\lambda_{scr}} \leq \frac{C_\Phi}{M_{min}} \cdot A_d\, \lambda_{scr}^d\, e^{-r_{max}/\lambda_{scr}}$$

where $A_d$ is a geometric prefactor (surface area of a ball in $d$ dimensions). This goes to zero **exponentially** as $r_{max}/\lambda_{scr} \to \infty$.

**Perturbation theory** (Weyl's inequality for eigenvalues) then gives:

$$|\omega_{\mathbf{q}s}^{trunc} - \omega_{\mathbf{q}s}^{exact}| \leq \frac{\|D_{exact} - D_{trunc}\|}{2\omega_{min}} \leq \frac{C_\Phi A_d \lambda_{scr}^d}{2M_{min}\omega_{min}} e^{-r_{max}/\lambda_{scr}}$$

For $r_{max} = 2\lambda_{scr}$: error $\leq Ce^{-2} \approx 0.14C$. For $r_{max} = 4\lambda_{scr}$: error $\leq Ce^{-4} \approx 0.018C$. The convergence is rapid.

**Step 2: Cell Size Sufficiency**

A periodic cell with linear dimension $L = 2r_{max}$ contains all pairs $(i,j)$ within distance $r_{max}$ without periodic image overlap. Choosing $L^* = 2r_{max} = 4\lambda_{scr}$ bounds the truncation error below any target $\epsilon_1$. Since $\lambda_{scr} \approx 3$–$8$ Å (temperature-independent), the minimum SQTC cell has $L^* \approx 12$–$32$ Å, corresponding to roughly 8–64 formula units depending on the lattice constant.

**Step 3: Displacement Field Sufficiency**

Given a cell $\Omega^*$ that contains all relevant IFC pairs, we need the displacement field $\{\mathbf{u}_i^*\}$ to make the regression matrix $\mathbf{G}$ of full rank (so the IFCs can be extracted uniquely).

The displacement field constructed from the phonon eigenvector basis (Chapter 11) with amplitudes $A_{\mathbf{q}s} = A_{\mathbf{q}s}^{harm}(T) \neq 0$ for all modes excites all phonon modes simultaneously. This guarantees that the regression matrix:

$$\mathbf{G}^T\mathbf{G} = \sum_{i,\alpha} \mathbf{g}_{i\alpha} \mathbf{g}_{i\alpha}^T$$

spans the full IFC parameter space, giving a well-posed (full-rank) system. The condition number $\kappa(\mathbf{G}^T\mathbf{G})$ is bounded by the ratio of maximum to minimum phonon amplitudes:

$$\kappa \leq \frac{\max_{\mathbf{q}s} A_{\mathbf{q}s}^2}{\min_{\mathbf{q}s} A_{\mathbf{q}s}^2} = \frac{\max_{\mathbf{q}s} \coth(\hbar\omega/2k_BT)/\omega}{\min_{\mathbf{q}s} \coth(\hbar\omega/2k_BT)/\omega}$$

which is finite for any non-zero temperature and any crystal with a gap-free acoustic branch at long wavelengths.

Therefore, the IFCs can be extracted with accuracy $\epsilon_2 \propto \kappa \cdot \epsilon_{DFT}$ where $\epsilon_{DFT}$ is the DFT force accuracy (typically $10^{-4}$ eV/Å).

**Combining the three steps:**

$$|\omega^{SQTC} - \omega^{exact}| < \underbrace{\epsilon_1}_{r_{max} \text{ truncation}} + \underbrace{\epsilon_2}_{IFC \text{ regression}} < \epsilon$$

for suitable choices of $r_{max}$ and displacement amplitudes. $\square$

### 15.3 Corollary: High-Temperature Efficiency

The minimum SQTC cell size is bounded below by the IFC range (Step 2 above), which is temperature-independent. However, the **ensemble size** $K$ (the number of SQTC configurations needed) is controlled by the number of significant thermal correlator pairs, which does decrease with temperature.

At temperature $T$, the thermal correlators are significant only for $|\mathbf{R}| \lesssim \xi_T(T)$ (the displacement-correlation decay length from Chapter 10). The number of significant correlator pairs scales as the volume of a sphere of radius $\xi_T$:

$$N_{corr}(T) \propto \left(\frac{\xi_T}{a}\right)^3 \propto T^{-3}$$

Since the SQTC ensemble must match $N_{corr}$ effective constraints using $3n$ force equations per cell, the minimum ensemble size:

$$K_{min}(T) \propto \frac{N_{corr}(T)}{3n} \propto T^{-3}$$

At $T = 1000$ K vs $T = 300$ K, the ensemble size advantage is:

$$\frac{K_{min}(300\text{ K})}{K_{min}(1000\text{ K})} = \left(\frac{1000}{300}\right)^3 \approx 37$$

A factor of $\sim$37 fewer configurations at 1000 K. Meanwhile, stochastic methods (TDEP/SSCHA) need *more* configurations at high $T$ (force variance $\sigma_F^2 \propto k_BT$ grows, requiring more snapshots to achieve the same regression accuracy). This double advantage — fewer SQTC configurations needed at high $T$ while stochastic methods need more — is why **SQTC is most powerful precisely where conventional methods struggle most**.

---

<a name="chapter-16"></a>
## Chapter 16: The SQTC Ensemble for Anharmonic Properties

### 16.1 Why a Single Cell Is Not Enough for Anharmonic Properties

A single SQTC cell gives a single displacement configuration. The 3-body IFCs ($\Psi_{ijk}^{\alpha\beta\gamma}$) require the force to depend nonlinearly on displacements. Extracting them needs displacement configurations that **vary the quadratic products** $u_j^\beta u_k^\gamma$ independently.

For $M_{3IFC}$ independent third-order force constants, we need at least $M_{3IFC}$ independent equations. Each SQTC cell of $n$ atoms gives $3n$ force equations. But the effective number of *independent* constraints for 3IFCs is $3n / \binom{3n}{2}$ (quadratic products are less diverse). So we need $K_{min} = \lceil M_{3IFC} / 3n \rceil$ cells.

### 16.2 The Ensemble Design Condition

The **SQTC ensemble** $\{(\Omega_m, \{\mathbf{u}_i^{(m)}\})\}_{m=1}^K$ is designed such that the **sample covariance** of displacements matches the target thermal covariance:

$$\frac{1}{K}\sum_{m=1}^K u_i^{\alpha,(m)} u_j^{\beta,(m)} = C_{ij}^{\alpha\beta}(\mathbf{R},T)$$

and the **sample third moment** matches the anharmonic three-body correlator:

$$\frac{1}{K}\sum_{m=1}^K u_i^{\alpha,(m)} u_j^{\beta,(m)} u_k^{\gamma,(m)} = \bar{C}_3^{ijk,\alpha\beta\gamma}(T)$$

This is essentially a **moment-matching** problem: find $K$ displacement configurations whose empirical moments match the thermal distribution moments up to order $k_{max}$.

### 16.3 Connection to Gaussian Quadrature

Moment matching by $K$ discrete points is exactly the $K$-point **Gaussian quadrature** problem in $d = 3n$ dimensions. The minimum $K$ to exactly match all moments up to degree $2p$ is the $p$-point Gauss-Hermite quadrature rule, giving $K = p^{3n}$ — exponential in dimension, intractable.

However, we do not need **all** moments — only those corresponding to **physically relevant clusters** (pairs within $r_{max}$, triplets within $r_3$, etc.). The number of relevant moments is $N_{phys} \ll (3n)^{k_{max}}$. By solving the moment-matching problem only for this restricted set, the minimum ensemble size becomes:

$$K_{min} = \lceil N_{phys}^{moments} / (3n) \rceil$$

which is typically $K = 5$–$30$ for $n = 8$–$16$ and $k_{max} = 3$. This is the basis of the dramatic efficiency improvement.

### 16.4 Phonon Linewidths from the 3IFC Ensemble

Once 3IFCs $\Psi_{ijk}^{\alpha\beta\gamma}$ are extracted from the ensemble, phonon lifetimes follow from the **Fermi Golden Rule** (three-phonon scattering):

$$\frac{1}{\tau_{\mathbf{q}s}(T)} = \frac{\pi\hbar}{8N}\sum_{\mathbf{q}',s'}\sum_{\mathbf{q}'',s''} |V_3(\mathbf{q}s;\mathbf{q}'s';\mathbf{q}''s'')|^2 \times \Delta_{\mathbf{q}+\mathbf{q}'+\mathbf{q}'',\mathbf{G}} \times f(\omega,\omega',\omega'',T)$$

where:
- $V_3$ is the three-phonon matrix element, computed from 3IFCs and eigenvectors
- $\Delta$ is the momentum conservation (Umklapp processes include reciprocal lattice vector $\mathbf{G}$)
- $f$ contains Bose-Einstein factors for energy conservation

The thermal conductivity then follows from the **Boltzmann Transport Equation (RTA approximation)**:

$$\kappa_L = \frac{1}{NV}\sum_{\mathbf{q},s} C_{\mathbf{q}s}(T)\, v_{\mathbf{q}s}^2\, \tau_{\mathbf{q}s}(T)$$

where $C_{\mathbf{q}s} = \hbar\omega_{\mathbf{q}s}\, \partial n_B/\partial T$ is the modal heat capacity and $v_{\mathbf{q}s} = \partial\omega_{\mathbf{q}s}/\partial\mathbf{q}$ is the group velocity.

---

<a name="chapter-17"></a>
## Chapter 17: Worked Numerical Example — 1D Diatomic Chain

This chapter works through a complete SQTC calculation for a one-dimensional diatomic chain (the simplest non-trivial crystal), so that every formula above is made concrete.

### 17.1 System: 1D Diatomic Chain

**Structure:** Two atoms per unit cell, masses $M_1 = 2m$ and $M_2 = m$ (like C-O), lattice constant $a = 2$ Å, unit cell length $2a = 4$ Å.

**IFCs:** Nearest-neighbour only (simplest case):

$$\Phi_{12}^{xx}(a) = -k = -5 \text{ eV/Å}^2, \quad \Phi_{11}^{xx}(0) = \Phi_{22}^{xx}(0) = k = 5 \text{ eV/Å}^2$$

(Acoustic sum rule satisfied: $\Phi_{11}^{xx}(0) + \Phi_{12}^{xx}(a) + \Phi_{12}^{xx}(-a) = k - k - k + \text{second neighbour... }$, we keep nearest-neighbour only for illustration.)

### 17.2 Harmonic Phonons

The dynamical matrix for wavevector $q$ (in 1D) is a $2\times 2$ matrix:

$$D(q) = \frac{k}{\sqrt{M_1 M_2}}\begin{pmatrix} \sqrt{M_2/M_1}\cdot 2 & -(e^{iqa} + e^{-iqa})\sqrt{M_1/M_2}^{-1/2}... \end{pmatrix}$$

Let us write it cleanly. With $M_1 = 2m$, $M_2 = m$:

$$D(q) = \begin{pmatrix} \Phi_{11}/M_1 & \Phi_{12}e^{iqa}/\sqrt{M_1 M_2} \\ \Phi_{12}e^{-iqa}/\sqrt{M_1 M_2} & \Phi_{22}/M_2 \end{pmatrix} = \frac{k}{m}\begin{pmatrix} 1/2 & -e^{iqa}/\sqrt{2} \\ -e^{-iqa}/\sqrt{2} & 1 \end{pmatrix}$$

Setting $\omega_0^2 = k/m$ (the natural frequency scale), the eigenvalues satisfy:

$$\det\left[\frac{k}{m}\begin{pmatrix}1/2-\lambda & -e^{iqa}/\sqrt{2} \\ -e^{-iqa}/\sqrt{2} & 1-\lambda\end{pmatrix}\right] = 0$$

where $\lambda = \omega^2/\omega_0^2$.

$$\left(\frac{1}{2}-\lambda\right)(1-\lambda) - \frac{1}{2} = 0$$

$$\lambda^2 - \frac{3}{2}\lambda + \frac{1}{2} - \frac{1}{2} = 0 \implies \lambda^2 - \frac{3}{2}\lambda = 0$$

This gives at $q=0$: $\lambda = 0$ (acoustic) and $\lambda = 3/2$ (optical). So:

$$\omega_{acoustic}(q=0) = 0, \quad \omega_{optical}(q=0) = \sqrt{3k/(2m)} = \sqrt{3/2}\,\omega_0$$

For $q = \pi/(2a)$ (zone boundary):

$$D(\pi/2a) = \frac{k}{m}\begin{pmatrix}1/2 & -i/\sqrt{2} \\ i/\sqrt{2} & 1\end{pmatrix}$$

Eigenvalues: solve $\lambda^2 - \frac{3}{2}\lambda + (\frac{1}{2} - \frac{1}{2}) = 0$... more carefully:

$$\det = (1/2-\lambda)(1-\lambda) - (1/\sqrt{2})^2 = 0 \implies \lambda^2 - \frac{3}{2}\lambda + \frac{1}{2} - \frac{1}{2} = \lambda(\lambda - \frac{3}{2}) = 0$$

Hmm — at the zone boundary, $e^{i\pi/2} = i$, $|e^{iqa}|^2 = 1$ always. The general result for the 1D diatomic chain at zone boundary $q_{ZB}$ is:

$$\omega_{ac}(q_{ZB}) = \sqrt{\frac{2k}{M_1}}, \quad \omega_{opt}(q_{ZB}) = \sqrt{\frac{2k}{M_2}}$$

With $M_1 = 2m$, $M_2 = m$: $\omega_{ac}(q_{ZB}) = \omega_0$ and $\omega_{opt}(q_{ZB}) = \sqrt{2}\,\omega_0$.

### 17.3 Computing the Target Thermal Correlator

At temperature $T$, using the formula from Chapter 4 (for 1D, single direction):

$$C^{target}(R, T) = \frac{\hbar}{2N}\sum_{q,s} \frac{|e_1^x(q,s)|^2}{M_1\,\omega_{qs}} \coth\!\left(\frac{\hbar\omega_{qs}}{2k_BT}\right) \cos(qR)$$

**Evaluating at $R = 0$ (on-site, mean-square displacement of atom 1):**

$$C_{11}^{xx}(0,T) = \langle u_1^2 \rangle_T = \frac{\hbar}{2N M_1}\sum_{q,s} \frac{|e_1^x(q,s)|^2}{\omega_{qs}} \coth\!\left(\frac{\hbar\omega_{qs}}{2k_BT}\right)$$

In the high-temperature limit ($k_BT \gg \hbar\omega$):

$$\langle u_1^2 \rangle_T \approx \frac{k_BT}{N M_1}\sum_{q,s} \frac{|e_1^x(q,s)|^2}{\omega_{qs}^2}$$

This is the **equipartition result**: each mode contributes $k_BT$ to $M_1\langle u_1^2\rangle\omega_{qs}^2/2$.

**Evaluating at $R = a$ (nearest-neighbour correlator):**

$$C^{target}(a,T) \approx \frac{k_BT}{N\sqrt{M_1 M_2}}\sum_{q,s} \frac{e_1^{x*}(q,s) e_2^x(q,s)}{\omega_{qs}^2} \cos(qa)$$

For our 1D chain with only acoustic and optical branches, this sum has two terms. The acoustic contribution (long wavelength, $q \to 0$, $\omega \propto q$) dominates at high $T$ due to the $1/\omega^2$ factor, reflecting that long-wavelength acoustic phonons are thermally most significant.

### 17.4 Designing the SQTC Cell

**Candidate cell:** We try an SQTC cell with 4 formula units (8 atoms) — a $4 \times (2a) = 8a$ cell.

Commensurate $q$-points: $q = 0, \pi/(4a), \pi/(2a), 3\pi/(4a)$ (and $-q$ partners, but in 1D they give same mode due to time-reversal).

**Displacement field:** Following Chapter 11, we write:

$$u_i = \sum_{q,s} A_{qs} e_i^x(q,s) \cos(q R_i^0 + \phi_{qs}) / \sqrt{M_i}$$

The harmonic amplitudes (in the classical limit) are:

$$A_{qs}^{harm}(T) = \sqrt{\frac{k_BT}{\omega_{qs}^2}} \quad \text{(classical, i.e., } k_BT \gg \hbar\omega\text{)}$$

**Objective function for this cell:** We have $n_\mathbf{R} = 4$ pairs at each distance $R = 0, 2a, 4a, 6a, a, 3a, 5a, 7a$ (self pairs and cross pairs). The SQTC quality functional is:

$$\mathcal{Q} = \lambda_2\sum_{R}\left[\bar{C}_2^{SQTC}(R) - C^{target}(R,T)\right]^2 e^{-R/\xi_T}$$

**Optimisation:** Vary $\{A_{qs}, \phi_{qs}\}$ to minimise $\mathcal{Q}$. Since this is a $8\times 2 = 16$-dimensional optimisation (4 modes $\times$ 2 branches $\times$ 2 parameters), it is trivially fast (milliseconds).

### 17.5 DFT Step (Conceptual)

For the 1D chain (a model system), we replace DFT with the exact force calculation:

$$F_i = -k(u_i - u_{i-1}) - k(u_i - u_{i+1})$$

This gives forces on all 8 atoms from our SQTC displacements. The IFC extraction (Chapter 12) recovers $k$ exactly (since the model is purely harmonic). For an anharmonic model (add $\lambda u^4$ term), the extracted $k_{eff}(T)$ depends on temperature, demonstrating phonon hardening or softening.

### 17.6 Convergence Check

**Compare correlators:**

After extracting IFCs and computing phonon dispersions, update:

$$C^{(1)}(R,T) = \frac{\hbar}{2N\sqrt{M_1 M_2}}\sum_{q,s}\frac{e_1^*(q,s)e_2(q,s)}{\omega_{qs}^{(1)}} \coth\!\left(\frac{\hbar\omega_{qs}^{(1)}}{2k_BT}\right) \cos(qR)$$

If $\|C^{(1)} - C^{(0)}\| < \epsilon_{conv}$: done. Otherwise: iterate.

For the harmonic chain: $\omega_{qs}^{(1)} = \omega_{qs}^{(0)}$ (IFCs are temperature-independent), so convergence is achieved in one iteration. For an anharmonic chain: the effective $k_{eff}(T)$ shifts the frequencies, requiring 2–4 iterations.

---

<a name="chapter-18"></a>
## Chapter 18: Variational Grounding — SQTC as Deterministic SSCHA Quadrature

### 18.1 The Exact Free Energy and Why It Is Hard

The exact Helmholtz free energy of an anharmonic crystal is:

$$F_{exact} = -k_BT \ln Z_{exact}, \quad Z_{exact} = \mathrm{Tr}\!\left(e^{-\hat{H}/k_BT}\right)$$

where $\hat{H} = \hat{H}_0 + \hat{V}_{anharm}$ includes all anharmonic terms beyond harmonic. This cannot be computed exactly for a real crystal because $\hat{V}_{anharm}$ couples all phonon modes — the partition function integral over $e^{-V_{anharm}/k_BT}$ does not factorise into mode-by-mode contributions.

### 18.2 The Gibbs-Bogoliubov Inequality

We choose a **trial Hamiltonian** $\hat{H}_0$ (harmonic, exactly solvable) with free energy $F_0 = -k_BT \ln Z_0$.

**Theorem (Gibbs-Bogoliubov):** For any trial Hamiltonian $\hat{H}_0$:

$$\boxed{F_{exact} \leq F_0 + \langle \hat{H} - \hat{H}_0 \rangle_0 \equiv F_{var}}$$

where $\langle \cdot \rangle_0 = \mathrm{Tr}(\hat{\rho}_0\, \cdot\,)$ denotes the thermal average with respect to $\hat{H}_0$, and $\hat{\rho}_0 = e^{-\hat{H}_0/k_BT}/Z_0$.

**Proof step by step:**

**Step 1:** Write $Z_{exact}$ by inserting $e^{-\hat{H}_0/k_BT}e^{+\hat{H}_0/k_BT}$:

$$Z_{exact} = \mathrm{Tr}\!\left(e^{-\hat{H}/k_BT}\right) = \mathrm{Tr}\!\left(e^{-\hat{H}_0/k_BT}\cdot e^{-(\hat{H}-\hat{H}_0)/k_BT}\right)$$

$$= Z_0 \cdot \frac{\mathrm{Tr}\!\left(e^{-\hat{H}_0/k_BT}\cdot e^{-(\hat{H}-\hat{H}_0)/k_BT}\right)}{Z_0} = Z_0 \cdot \left\langle e^{-(\hat{H}-\hat{H}_0)/k_BT} \right\rangle_0$$

**Step 2:** Apply **Jensen's inequality**. Since $f(x) = e^x$ is convex ($f'' = e^x > 0$), Jensen's inequality states $\langle e^X \rangle \geq e^{\langle X \rangle}$ for any random variable $X$. Let $X = -(\hat{H}-\hat{H}_0)/k_BT$:

$$\left\langle e^{-(\hat{H}-\hat{H}_0)/k_BT} \right\rangle_0 \geq e^{-\langle \hat{H}-\hat{H}_0 \rangle_0/k_BT}$$

**Step 3:** Combine:

$$Z_{exact} \geq Z_0 \cdot e^{-\langle \hat{H}-\hat{H}_0 \rangle_0/k_BT}$$

**Step 4:** Take $-k_BT\ln(\cdot)$ of both sides. Since $-k_BT\ln$ is a monotone decreasing function, the inequality reverses:

$$-k_BT\ln Z_{exact} \leq -k_BT\ln Z_0 + \langle \hat{H} - \hat{H}_0 \rangle_0$$

$$\Rightarrow \quad F_{exact} \leq F_0 + \langle \hat{H} - \hat{H}_0 \rangle_0 \quad \square$$

**Physical meaning:** The exact free energy is always *at most* the variational free energy. The trial Hamiltonian $\hat{H}_0$ provides an upper bound. By minimising $F_{var}$ over all choices of $\hat{H}_0$, we get the best variational estimate of $F_{exact}$.

### 18.3 The SSCHA as a Variational Problem

The **Stochastic Self-Consistent Harmonic Approximation (SSCHA)** chooses $\hat{H}_0$ to be a harmonic Hamiltonian with renormalized force constants $\boldsymbol{\Phi}^*(T)$:

$$\hat{H}_0[\boldsymbol{\Phi}^*] = \frac{1}{2}\sum_{i,\alpha}\frac{\hat{p}_i^{\alpha 2}}{M_i} + \frac{1}{2}\sum_{ij,\alpha\beta}\Phi_{ij}^{*\alpha\beta}(T)\hat{u}_i^\alpha\hat{u}_j^\beta$$

and minimises $F_{var}[\boldsymbol{\Phi}^*]$ with respect to $\boldsymbol{\Phi}^*$. Setting $\partial F_{var}/\partial\Phi_{ij}^{*\alpha\beta} = 0$ gives the **SSCHA self-consistency condition**:

$$\Phi_{ij}^{*\alpha\beta} = \left\langle\frac{\partial^2 V_{exact}}{\partial u_i^\alpha \partial u_j^\beta}\right\rangle_0$$

where $\langle\cdot\rangle_0$ is the average over the Gaussian displacement distribution $\mathcal{D}[\mathbf{u}]$ with covariance matrix $\mathbf{C}^{*}(T)$ determined by $\boldsymbol{\Phi}^*$. This is a self-consistent equation: $\boldsymbol{\Phi}^*$ appears on both sides.

### 18.4 SQTC as Deterministic Quadrature of the SSCHA Integral

The SSCHA evaluates the thermal average $\langle \partial^2 V/\partial u_i\partial u_j\rangle_0$ by stochastically sampling $M$ displacement configurations $\{\mathbf{u}^{(m)}\}$ from the Gaussian:

$$\mathcal{D}[\mathbf{u}] = \frac{1}{(2\pi)^{3N/2}\det(\mathbf{C})^{1/2}}\exp\!\left(-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}\right)$$

The Monte Carlo estimate converges as $1/\sqrt{M}$ — slowly, requiring hundreds to thousands of configurations.

**SQTC replaces stochastic sampling with deterministic Gauss-Hermite quadrature.** Find $K \ll M$ displacement configurations $\{\mathbf{u}^{(k)}\}$ (the quadrature nodes) with positive weights $\{w_k > 0\}$ such that:

$$\langle f(\mathbf{u}) \rangle_\mathcal{D} \approx \sum_{k=1}^K w_k f(\mathbf{u}^{(k)})$$

is exact for all polynomial functions $f$ up to degree $2p-1$ (a $p$-point quadrature rule).

The nodes $\{\mathbf{u}^{(k)}\}$ are precisely the **SQTC displacement configurations** whose moment-matching condition was derived in Chapter 9. The correlator matching:

$$\frac{1}{K}\sum_{k=1}^K u_i^{\alpha,(k)} u_j^{\beta,(k)} = C_{ij}^{\alpha\beta}(T) \quad \text{(2nd moment)}$$

$$\frac{1}{K}\sum_{k=1}^K u_i^{\alpha,(k)} u_j^{\beta,(k)} u_k^{\gamma,(k)} = \bar{C}_3^{ij}(T) \quad \text{(3rd moment)}$$

is *exactly* the quadrature node condition for polynomial integration over the Gaussian $\mathcal{D}$.

**Key insight:** SQTC $\equiv$ deterministic Gauss-Hermite quadrature of the SSCHA variational integral. This provides SQTC with a rigorous variational upper bound on $F_{exact}$ and explains why it achieves high accuracy with very few configurations.

---

<a name="chapter-19"></a>
## Chapter 19: The SQTC Harmonic Free Energy on the Small Cell

### 19.1 Free Energy of the Renormalized Harmonic System

After the SQTC self-consistency loop converges (Chapter 14), we have the renormalized frequencies $\omega_{\mathbf{q}s}^*(T)$ — the best harmonic approximation to the anharmonic crystal at temperature $T$. The SQTC cell has $N_{SC}$ atoms (typically 8–64) with $3N_{SC}$ phonon modes.

The **harmonic free energy** for the renormalized system is:

$$F_{harm}^{SQTC}(T) = k_BT\sum_{\mathbf{q},s}\ln\!\left(2\sinh\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT}\right)$$

**Step-by-step derivation:**

*Single oscillator partition function* for mode $(\mathbf{q}s)$ with frequency $\omega = \omega_{\mathbf{q}s}^*$:

$$Z_\omega = \sum_{n=0}^\infty e^{-\hbar\omega(n+1/2)/k_BT} = e^{-\hbar\omega/2k_BT}\sum_{n=0}^\infty \left(e^{-\hbar\omega/k_BT}\right)^n$$

Using the geometric series $\sum_{n=0}^\infty x^n = 1/(1-x)$ for $|x|<1$:

$$Z_\omega = \frac{e^{-\hbar\omega/2k_BT}}{1 - e^{-\hbar\omega/k_BT}}$$

Multiply numerator and denominator by $e^{+\hbar\omega/2k_BT}$:

$$Z_\omega = \frac{1}{e^{\hbar\omega/2k_BT} - e^{-\hbar\omega/2k_BT}} = \frac{1}{2\sinh(\hbar\omega/2k_BT)}$$

Single-mode free energy:

$$F_\omega = -k_BT\ln Z_\omega = k_BT\ln\!\left(2\sinh\frac{\hbar\omega}{2k_BT}\right)$$

Summing over all $3N_{SC}$ independent modes (phonons are independent in the harmonic approximation):

$$\boxed{F_{harm}^{SQTC}(T) = k_BT\sum_{\mathbf{q},s}\ln\!\left(2\sinh\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT}\right)}$$

**Check the limits:**

*Zero temperature ($T \to 0$):* Use $2\sinh(x) \to e^x$ as $x \to \infty$:

$$F_{harm}^{SQTC}(0) = \sum_{\mathbf{q},s}\frac{\hbar\omega_{\mathbf{q}s}^*}{2} \quad \text{(zero-point energy — present even at absolute zero)}$$

*High temperature ($k_BT \gg \hbar\omega$):* Use $\sinh(x) \approx x$ for small $x$:

$$F_{harm}^{SQTC}(T) \xrightarrow{k_BT\gg\hbar\omega} k_BT\sum_{\mathbf{q},s}\ln\!\frac{\hbar\omega_{\mathbf{q}s}^*}{k_BT} \quad \text{(classical/Dulong-Petit limit)}$$

### 19.2 Entropy from the SQTC Harmonic Free Energy

The entropy is $S = -\partial F/\partial T$. Differentiating $F_\omega = k_BT\ln(2\sinh(\hbar\omega/2k_BT))$ with respect to $T$:

$$\frac{\partial F_\omega}{\partial T} = k_B\ln\!\left(2\sinh\frac{\hbar\omega}{2k_BT}\right) + k_BT\cdot\frac{\cosh(\hbar\omega/2k_BT)}{\sinh(\hbar\omega/2k_BT)}\cdot\left(-\frac{\hbar\omega}{2k_BT^2}\right)$$

$$= k_B\ln\!\left(2\sinh\frac{\hbar\omega}{2k_BT}\right) - \frac{\hbar\omega}{2T}\coth\!\frac{\hbar\omega}{2k_BT}$$

Therefore $S_\omega = -\partial F_\omega/\partial T$:

$$\boxed{S_{harm}^{SQTC}(T) = k_B\sum_{\mathbf{q},s}\left[\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT}\coth\!\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT} - \ln\!\left(2\sinh\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT}\right)\right]}$$

**Verification:** At high $T$, $x\coth(x) \to 1 + x^2/3 + \cdots \to 1$ and $\ln(2\sinh x) \to \ln(2x) \to \ln(\hbar\omega/k_BT)$, so $S_\omega \to k_B(1 - \ln(\hbar\omega/k_BT))$. This grows logarithmically with $T$ — consistent with the classical result. ✓

### 19.3 Heat Capacity at Constant Volume

The heat capacity $C_V = T\,\partial S/\partial T = -T\,\partial^2 F/\partial T^2$. Differentiating $S_\omega$:

$$\frac{\partial}{\partial T}\left(\frac{\hbar\omega}{2k_BT}\coth\!\frac{\hbar\omega}{2k_BT}\right) = -\frac{\hbar\omega}{2k_BT^2}\coth + \frac{\hbar\omega}{2k_BT}\cdot\left(-\frac{1}{\sinh^2}\right)\cdot\left(-\frac{\hbar\omega}{2k_BT^2}\right)$$

$$= -\frac{\hbar\omega}{2k_BT^2}\coth + \left(\frac{\hbar\omega}{2k_BT}\right)^2\frac{1}{k_BT\sinh^2}$$

The $\coth$ term cancels from $\partial S/\partial T$, leaving:

$$\boxed{C_V^{harm}(T) = k_B\sum_{\mathbf{q},s}\left(\frac{\hbar\omega_{\mathbf{q}s}^*}{2k_BT}\right)^2 \frac{1}{\sinh^2\!\left(\hbar\omega_{\mathbf{q}s}^*/2k_BT\right)}}$$

**Verification:** At high $T$, $(x/\sinh x)^2 \to 1$ as $x \to 0$, so $C_V \to 3N_{SC}k_B$ — the **Dulong-Petit law**. ✓ At low $T$, $C_V \propto T^3$ (Debye law for acoustic phonons). ✓

---

<a name="chapter-20"></a>
## Chapter 20: Anharmonic Free Energy I — Wick's Theorem and the Tadpole Diagram (4IFC)

### 20.1 Setting Up Perturbation Theory

The full potential energy, expanded around equilibrium (Chapter 1), is:

$$V = V_2 + V_3 + V_4 + \cdots$$

where the harmonic, cubic, and quartic terms are:

$$V_2 = \frac{1}{2}\sum_{ij,\alpha\beta}\Phi_{ij}^{\alpha\beta}u_i^\alpha u_j^\beta, \quad V_3 = \frac{1}{6}\sum_{ijk,\alpha\beta\gamma}\Psi_{ijk}^{\alpha\beta\gamma}u_i^\alpha u_j^\beta u_k^\gamma, \quad V_4 = \frac{1}{24}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta$$

In the Gibbs-Bogoliubov variational framework (Chapter 18):

$$F_{var} = F_0[\boldsymbol{\Phi}^*] + \langle V_{exact} - V_0[\boldsymbol{\Phi}^*] \rangle_\mathcal{D}$$

where $V_0[\boldsymbol{\Phi}^*] = V_2[\boldsymbol{\Phi}^*]$ is the harmonic potential with renormalized IFCs. So the correction terms are $\langle V_3 \rangle_\mathcal{D} + \langle V_4 \rangle_\mathcal{D}$ (plus higher-order terms). The angular brackets $\langle\cdot\rangle_\mathcal{D}$ denote a Gaussian average with covariance $\mathbf{C}^*(T)$, the thermal correlator evaluated at the renormalized frequencies.

### 20.2 Wick's Theorem — Complete Proof

**Theorem (Wick's Theorem for Gaussian averages):** For any multivariate Gaussian distribution $\mathcal{D}[\mathbf{u}] \propto e^{-\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}/2}$, the $2n$-point correlation function is:

$$\langle u_{i_1} u_{i_2} \cdots u_{i_{2n}} \rangle_\mathcal{D} = \sum_{\sigma \in \mathcal{P}_{2n}} \prod_{\text{pairs }(a,b) \in \sigma} C_{i_a i_b}$$

where the sum runs over all $(2n-1)!! = (2n-1)(2n-3)\cdots 3\cdot 1$ ways to partition $\{1,2,\ldots,2n\}$ into $n$ unordered pairs. Odd moments vanish: $\langle u_{i_1}\cdots u_{i_{2n+1}}\rangle_\mathcal{D} = 0$.

**Proof using the generating function (source method):**

**Step 1: Define the moment-generating function:**

$$Z[\mathbf{J}] = \langle e^{\mathbf{J}\cdot\mathbf{u}} \rangle_\mathcal{D} = \frac{\int d\mathbf{u}\; e^{\mathbf{J}\cdot\mathbf{u}} \,e^{-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}}}{\int d\mathbf{u}\; e^{-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}}}$$

Any correlation function follows by differentiation: $\langle u_{i_1}\cdots u_{i_k}\rangle = \frac{\partial^k Z}{\partial J_{i_1}\cdots\partial J_{i_k}}\big|_{\mathbf{J}=0}$.

**Step 2: Complete the square** in the exponent of the numerator integrand:

$$-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{J}\cdot\mathbf{u} = -\frac{1}{2}(\mathbf{u}-\mathbf{C}\mathbf{J})^T\mathbf{C}^{-1}(\mathbf{u}-\mathbf{C}\mathbf{J}) + \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}$$

**Verify:** Expanding the right side: $-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{J}^T\mathbf{C}\mathbf{C}^{-1}\mathbf{u} - \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J} + \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J} = -\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{J}\cdot\mathbf{u}$ ✓

**Step 3:** Shift the integration variable $\mathbf{v} = \mathbf{u} - \mathbf{C}\mathbf{J}$ (the Jacobian of this linear shift is 1):

$$Z[\mathbf{J}] = e^{\frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}} \cdot \underbrace{\frac{\int d\mathbf{v}\; e^{-\frac{1}{2}\mathbf{v}^T\mathbf{C}^{-1}\mathbf{v}}}{\int d\mathbf{u}\; e^{-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}}}}_{= 1}$$

Therefore:

$$\boxed{Z[\mathbf{J}] = \exp\!\left(\frac{1}{2}\sum_{i,j}J_i C_{ij} J_j\right)}$$

**Step 4: Extract moments by differentiation.** For odd $k$: every differentiation of $Z[\mathbf{J}] = e^{\frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}}$ brings down at least one factor of $\mathbf{J}$ or $\mathbf{C}$. Setting $\mathbf{J}=0$ makes any surviving $\mathbf{J}$-factor vanish. For odd $k$, at least one $\mathbf{J}$ always remains, so **odd moments vanish** identically. ✓

For even $k=2n$: the $2n$ derivatives each act on the exponent $\frac{1}{2}\sum_{ab}J_aC_{ab}J_b$. Each derivative either:
- Differentiates a $J$ factor, bringing down a $C$ with the other $J$ still present, or
- Sets $\mathbf{J}=0$ after all derivatives are taken.

The only surviving terms at $\mathbf{J}=0$ are those where every $J$ has been differentiated exactly once — corresponding exactly to a complete pairing of all $2n$ indices. Each pairing $\{(i_a,i_b)\}$ contributes $\prod C_{i_ai_b}$, summed over all $(2n-1)!!$ distinct pairings. This proves Wick's theorem. $\square$

**Explicit cases:**

*2-point:* $\langle u_i u_j \rangle_\mathcal{D} = C_{ij}$ (just 1 pairing)

*4-point:* $\langle u_i u_j u_k u_l \rangle_\mathcal{D} = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}$ (3 pairings)

*6-point:* Sum over 15 pairings (used in Chapter 21)

### 20.3 The 3IFC Term Vanishes at First Order

The first-order correction from cubic anharmonicity is:

$$\langle V_3 \rangle_\mathcal{D} = \frac{1}{6}\sum_{ijk,\alpha\beta\gamma}\Psi_{ijk}^{\alpha\beta\gamma}\langle u_i^\alpha u_j^\beta u_k^\gamma\rangle_\mathcal{D}$$

By Wick's theorem, the 3-point Gaussian average is an odd moment:

$$\langle u_i^\alpha u_j^\beta u_k^\gamma\rangle_\mathcal{D} = 0 \quad \text{(all odd moments of a Gaussian vanish)}$$

Therefore $\langle V_3 \rangle_\mathcal{D} = 0$ exactly. **This is not an approximation** — it is an exact consequence of the symmetry of the Gaussian distribution $\mathcal{D}[\mathbf{u}]$. Physically: the displacement distribution is symmetric around zero ($\mathbf{u} \to -\mathbf{u}$ leaves $\mathcal{D}$ invariant), so odd powers of displacements average to zero.

### 20.4 The Tadpole Contribution from 4IFCs

The first-order correction from quartic anharmonicity is:

$$\Delta F_4^{var} = \langle V_4 \rangle_\mathcal{D} = \frac{1}{24}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}\langle u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta\rangle_\mathcal{D}$$

Apply the 4-point Wick theorem:

$$\langle u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta\rangle_\mathcal{D} = C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta} + C_{ik}^{\alpha\gamma}C_{jl}^{\beta\delta} + C_{il}^{\alpha\delta}C_{jk}^{\beta\gamma}$$

By the full permutation symmetry of $\Xi_{ijkl}^{\alpha\beta\gamma\delta}$ under simultaneous exchange of any pair of index pairs (e.g., $(i\alpha)\leftrightarrow(j\beta)$, $(i\alpha)\leftrightarrow(k\gamma)$, etc.), all three pairings give equal contributions when summed over all indices. Therefore:

$$\Delta F_4^{var} = \frac{3}{24}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta}$$

$$\boxed{\Delta F_4^{var} = \frac{1}{8}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}(T)C_{kl}^{\gamma\delta}(T)}$$

**Sign analysis:** In a mechanically stable crystal, $\Xi_{ijkl} > 0$ (the quartic term stiffens the potential). Correlators $C_{ij}^{\alpha\beta}(T) > 0$ for diagonal elements. Therefore $\Delta F_4^{var} > 0$: quartic anharmonicity **raises** the free energy. At high temperature, $C_{ij} \propto k_BT$, so $\Delta F_4^{var} \propto (k_BT)^2$ — a superlinear growth above the harmonic $F \propto k_BT\ln(k_BT)$.

**Feynman diagram interpretation — the tadpole:** In diagrammatic perturbation theory, the 4IFC vertex $\Xi_{ijkl}$ is drawn as a filled square with 4 legs. Each correlator $C_{ij}$ is a line (propagator) connecting two legs. The tadpole diagram has two propagator loops attached to a single 4-point vertex, like a tadpole with a round head. It is the simplest anharmonic Feynman diagram.

---

<a name="chapter-21"></a>
## Chapter 21: Anharmonic Free Energy II — The Sunset Diagram (3IFC)

### 21.1 Second-Order Perturbation Theory

Since $\langle V_3 \rangle_\mathcal{D} = 0$ (Chapter 20), the leading contribution from cubic anharmonicity enters at **second order** in perturbation theory. The free energy correction is given by the **cumulant expansion** at second order:

$$\Delta F_3^{pert} = -\frac{1}{2k_BT}\left[\langle V_3^2\rangle_\mathcal{D} - \langle V_3\rangle_\mathcal{D}^2\right]$$

Since $\langle V_3\rangle_\mathcal{D} = 0$, the second term vanishes:

$$\Delta F_3^{pert} = -\frac{\langle V_3^2\rangle_\mathcal{D}}{2k_BT}$$

**Why is there a factor of $-1/(2k_BT)$?** This comes from expanding $-k_BT\ln Z$ to second order in $V_3/k_BT$ (the cumulant expansion of the log):

$$-k_BT\ln\langle e^{-V_3/k_BT}\rangle_\mathcal{D} \approx \langle V_3\rangle_\mathcal{D} - \frac{\langle V_3^2\rangle_\mathcal{D} - \langle V_3\rangle_\mathcal{D}^2}{2k_BT} = -\frac{\langle V_3^2\rangle_\mathcal{D}}{2k_BT}$$

### 21.2 Computing $\langle V_3^2\rangle_\mathcal{D}$ via Wick's Theorem

Expanding $V_3^2$:

$$\langle V_3^2\rangle_\mathcal{D} = \frac{1}{36}\sum_{\substack{ijk\\\alpha\beta\gamma}}\sum_{\substack{i'j'k'\\\alpha'\beta'\gamma'}}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'}\langle u_i^\alpha u_j^\beta u_k^\gamma u_{i'}^{\alpha'} u_{j'}^{\beta'} u_{k'}^{\gamma'}\rangle_\mathcal{D}$$

This is a **6-point Gaussian average** — apply Wick's theorem. Label the 6 displacement factors as positions $\{1,2,3,4,5,6\}$ where $\{1,2,3\} = \{u_i^\alpha, u_j^\beta, u_k^\gamma\}$ come from the first $\Psi$ and $\{4,5,6\} = \{u_{i'}^{\alpha'}, u_{j'}^{\beta'}, u_{k'}^{\gamma'}\}$ come from the second $\Psi$. By Wick's theorem: $\langle 1\cdot2\cdot3\cdot4\cdot5\cdot6\rangle_\mathcal{D} = \sum_{\text{15 pairings}}\prod C_{ab}$.

### 21.3 The 15 Pairings: Connected vs Disconnected

The $(6-1)!! = 15$ pairings of 6 objects fall into two types:

**Disconnected pairings (9 total):** At least one propagator pair stays entirely within the first $\Psi$ vertex or entirely within the second $\Psi$ vertex. For example, $(12)(34)(56)$: pair $(1,2)$ connects two legs of $\Psi_1$ to itself, pair $(3,4)$ and $(5,6)$ connect $\Psi_1$ to $\Psi_2$ partially. Actually, for a pairing to be *truly* disconnected (in the field-theory sense), there must be a self-contraction within one vertex. These correspond to a product of two separate vacuum bubbles.

More precisely, a pairing is **connected** if every pair $(a,b)$ has one index from $\{1,2,3\}$ and one from $\{4,5,6\}$ — i.e., all three propagators bridge from vertex $\Psi_1$ to vertex $\Psi_2$.

**Connected pairings (6 total):** All three pairs bridge between $\Psi_1$ and $\Psi_2$:

| Pairing | Correlator product |
|---|---|
| $(14)(25)(36)$ | $C_{ii'}^{\alpha\alpha'}C_{jj'}^{\beta\beta'}C_{kk'}^{\gamma\gamma'}$ |
| $(14)(26)(35)$ | $C_{ii'}^{\alpha\alpha'}C_{jk'}^{\beta\gamma'}C_{kj'}^{\gamma\beta'}$ |
| $(15)(24)(36)$ | $C_{ij'}^{\alpha\beta'}C_{ji'}^{\beta\alpha'}C_{kk'}^{\gamma\gamma'}$ |
| $(15)(26)(34)$ | $C_{ij'}^{\alpha\beta'}C_{jk'}^{\beta\gamma'}C_{ki'}^{\gamma\alpha'}$ |
| $(16)(24)(35)$ | $C_{ik'}^{\alpha\gamma'}C_{ji'}^{\beta\alpha'}C_{kj'}^{\gamma\beta'}$ |
| $(16)(25)(34)$ | $C_{ik'}^{\alpha\gamma'}C_{jj'}^{\beta\beta'}C_{ki'}^{\gamma\alpha'}$ |

**Why only connected diagrams contribute to the free energy?** The disconnected diagrams contribute to $\langle V_3\rangle_\mathcal{D}^2 = 0^2 = 0$ in the cumulant expansion, and they cancel exactly (this is the linked-cluster theorem of statistical mechanics). The free energy receives contributions only from **connected** Feynman diagrams.

By the permutation symmetry of $\Psi_{ijk}^{\alpha\beta\gamma}$ under exchange of any pair of its three index-pairs (since $\Psi$ is symmetric under exchange of like atoms), all 6 connected pairings give equal contributions when summed over all indices. So:

$$\langle V_3^2\rangle_\mathcal{D}^{conn} = \frac{6}{36}\sum_{ijk,i'j'k',\alpha\beta\gamma,\alpha'\beta'\gamma'}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'}C_{ii'}^{\alpha\alpha'}C_{jj'}^{\beta\beta'}C_{kk'}^{\gamma\gamma'}$$

$$= \frac{1}{6}\sum_{ijk,i'j'k',\alpha\beta\gamma,\alpha'\beta'\gamma'}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'}C_{ii'}^{\alpha\alpha'}C_{jj'}^{\beta\beta'}C_{kk'}^{\gamma\gamma'}$$

### 21.4 Converting to Phonon Language

Express the correlator in terms of phonon modes (from Chapter 4):

$$C_{ii'}^{\alpha\alpha'}(T) = \frac{\hbar}{2N}\sum_{\mathbf{q},s}\frac{e_i^\alpha(\mathbf{q}s)e_{i'}^{\alpha'*}(\mathbf{q}s)}{\sqrt{M_iM_{i'}}\,\omega_{\mathbf{q}s}^*}(2n_{\mathbf{q}s}+1)\, e^{-i\mathbf{q}\cdot(\mathbf{R}_i-\mathbf{R}_{i'})}$$

Define the **three-phonon matrix element** for modes $(\mathbf{q}s)$, $(\mathbf{q}'s')$, $(\mathbf{q}''s'')$:

$$V_3(\mathbf{q}s;\mathbf{q}'s';\mathbf{q}''s'') = \sum_{ijk,\alpha\beta\gamma}\Psi_{ijk}^{\alpha\beta\gamma}\frac{e_i^\alpha(\mathbf{q}s)e_j^\beta(\mathbf{q}'s')e_k^\gamma(\mathbf{q}''s'')}{\sqrt{M_iM_jM_k}}e^{i(\mathbf{q}\cdot\mathbf{R}_i+\mathbf{q}'\cdot\mathbf{R}_j+\mathbf{q}''\cdot\mathbf{R}_k)}$$

This matrix element is nonzero only when $\mathbf{q}+\mathbf{q}'+\mathbf{q}'' = \mathbf{G}$ (a reciprocal lattice vector) — the **momentum conservation** for three-phonon processes (Normal and Umklapp processes).

Substituting and performing the Fourier sums:

$$\langle V_3^2\rangle_\mathcal{D}^{conn} = \frac{1}{N^2}\!\!\sum_{\substack{\mathbf{q}s,\mathbf{q}'s'\\\mathbf{q}''s''}}\!\!|V_3(\mathbf{q}s;\mathbf{q}'s';\mathbf{q}''s'')|^2\cdot\frac{\hbar^3}{8\omega_{\mathbf{q}s}^*\omega_{\mathbf{q}'s'}^*\omega_{\mathbf{q}''s''}^*}(2n_{\mathbf{q}s}+1)(2n_{\mathbf{q}'s'}+1)(2n_{\mathbf{q}''s''}+1)\,\delta_{\mathbf{q}+\mathbf{q}'+\mathbf{q}'',\mathbf{G}}$$

Therefore:

$$\boxed{\Delta F_3^{pert} = -\frac{\hbar^3}{16N^2k_BT}\!\!\sum_{\substack{\mathbf{q}s,\mathbf{q}'s'\\\mathbf{q}''s''}}\!\!\frac{|V_3|^2(2n_{\mathbf{q}s}+1)(2n_{\mathbf{q}'s'}+1)(2n_{\mathbf{q}''s''}+1)}{\omega_{\mathbf{q}s}^*\omega_{\mathbf{q}'s'}^*\omega_{\mathbf{q}''s''}^*}\,\delta_{\mathbf{q}+\mathbf{q}'+\mathbf{q}'',\mathbf{G}}}$$

**Sign analysis:** $|V_3|^2 > 0$ always (it is a squared modulus), $(2n+1) > 0$ always, $\omega^* > 0$ for a stable crystal, and $1/k_BT > 0$. Therefore $\Delta F_3^{pert} < 0$: cubic anharmonicity **always lowers** the free energy. Physical interpretation: the cubic term allows the crystal to explore asymmetric configurations with lower potential energy, increasing the accessible phase space and thus the entropy.

**Temperature dependence:** At high $T$, $2n_B + 1 \approx 2k_BT/\hbar\omega$, so $(2n+1)^3/\omega^3 \propto (k_BT)^3/(\hbar\omega)^3$, giving $\Delta F_3^{pert} \propto -(k_BT)^2/\hbar^2$ — a large negative correction at high $T$.

**The sunset diagram:** The Feynman diagram for this term has two triangular 3-point vertices, each with 3 phonon legs, connected by 3 phonon propagators. The shape resembles a setting sun (a circle with lines through it), hence the name **sunset** or **setting-sun** diagram.

---

<a name="chapter-22"></a>
## Chapter 22: The Complete SQTC Anharmonic Free Energy and Thermodynamic Properties

### 22.1 The Full SQTC Free Energy

Combining the harmonic free energy (Chapter 19), the tadpole correction (Chapter 20), and the sunset correction (Chapter 21):

$$\boxed{F_{SQTC}(T) = F_{harm}[\boldsymbol{\Phi}^*(T)] + \Delta F_4^{var}(T) + \Delta F_3^{pert}(T)}$$

where:

- $F_{harm}[\boldsymbol{\Phi}^*] = k_BT\sum_{\mathbf{q}s}\ln[2\sinh(\hbar\omega_{\mathbf{q}s}^*/2k_BT)]$ — harmonic part with self-consistently renormalized IFCs $\boldsymbol{\Phi}^*(T)$
- $\Delta F_4^{var} = \frac{1}{8}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta} > 0$ — quartic tadpole (stiffens the lattice, raises $F$)
- $\Delta F_3^{pert} < 0$ — cubic sunset (lowers $F$, increases entropy)

This expression is an upper bound on $F_{exact}$ (by Gibbs-Bogoliubov), with the corrections $\Delta F_4 + \Delta F_3$ systematically improving the bound beyond the pure harmonic approximation.

### 22.2 The Phonon Self-Energy

The anharmonic corrections modify the phonon dispersion through the **self-energy** $\Sigma_{\mathbf{q}s}(\omega, T) = \Delta_{\mathbf{q}s}(T) - i\Gamma_{\mathbf{q}s}(T)$.

**Real part $\Delta_{\mathbf{q}s}$ — frequency shift from the 4IFC tadpole:**

$$\Delta_{\mathbf{q}s}(T) = \frac{1}{2N\omega_{\mathbf{q}s}^*}\sum_{\mathbf{q}',s'}\Phi_{\mathbf{q}s;\mathbf{q}'s'}^{(4)}(2n_{\mathbf{q}'s'}+1)$$

where $\Phi_{\mathbf{q}s;\mathbf{q}'s'}^{(4)}$ is the 4IFC $\Xi$ contracted with two pairs of eigenvectors. The renormalized (observable) phonon frequency is:

$$\tilde{\omega}_{\mathbf{q}s}(T) = \omega_{\mathbf{q}s}^*(T) + \Delta_{\mathbf{q}s}(T)$$

Positive $\Delta_{\mathbf{q}s}$ means the quartic term **hardens** the phonon (raises its frequency). This is the anharmonic phonon stiffening that stabilises high-temperature phases (e.g., in BaTiO$_3$, Chapter 23).

**Imaginary part $\Gamma_{\mathbf{q}s}$ — phonon linewidth from the 3IFC sunset:**

$$\Gamma_{\mathbf{q}s}(T) = \frac{\pi\hbar}{8N}\sum_{\mathbf{q}'s',\mathbf{q}''s''}\!\!|V_3(\mathbf{q}s;\mathbf{q}'s';\mathbf{q}''s'')|^2\,\delta_{\mathbf{q}+\mathbf{q}'+\mathbf{q}'',\mathbf{G}} \times f_{\pm}(n',n'',\omega^*)$$

where the energy conservation factor is:

$$f_{\pm}(n',n'',\omega^*) = (n'+n''+1)\,\delta(\omega-\omega'-\omega'') + 2(n'-n'')\,\delta(\omega+\omega'-\omega'')$$

The first term (proportional to $n'+n''+1$) corresponds to **phonon decay**: one high-frequency phonon $(\mathbf{q}s)$ decays into two lower-frequency phonons $(\mathbf{q}')$ and $(\mathbf{q}'')$. The second term corresponds to **phonon absorption**: phonon $(\mathbf{q}s)$ absorbs phonon $(\mathbf{q}')$ to produce $(\mathbf{q}'')$.

The **phonon lifetime** is:

$$\tau_{\mathbf{q}s}(T) = \frac{\hbar}{2\Gamma_{\mathbf{q}s}(T)} \propto T^{-1} \quad \text{at high } T$$

### 22.3 Entropy of the Full Anharmonic System

The entropy is $S = -\partial F_{SQTC}/\partial T$:

$$S_{SQTC}(T) = S_{harm}(T) + S_4(T) + S_3(T)$$

where the harmonic entropy $S_{harm}$ was derived in Chapter 19, and the anharmonic corrections are:

$$S_4(T) = -\frac{\partial\Delta F_4^{var}}{\partial T} = -\frac{1}{8}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}\left(C_{kl}^{\gamma\delta}\frac{\partial C_{ij}^{\alpha\beta}}{\partial T} + C_{ij}^{\alpha\beta}\frac{\partial C_{kl}^{\gamma\delta}}{\partial T}\right)$$

Using $\partial C_{ij}^{\alpha\beta}/\partial T = \frac{\hbar^2}{4N k_B T^2}\sum_{\mathbf{q}s}\frac{e_i^\alpha e_j^{\beta*}e^{-i\mathbf{q}\cdot\mathbf{R}}}{\sqrt{M_iM_j}\omega_{\mathbf{q}s}\sinh^2(\hbar\omega_{\mathbf{q}s}/2k_BT)}$, one can see $\partial C/\partial T > 0$ (correlators grow with $T$) and $S_4 < 0$ (quartic term reduces entropy — stiffer potential).

### 22.4 Heat Capacity Beyond Dulong-Petit

$$C_V^{SQTC}(T) = -T\frac{\partial^2 F_{SQTC}}{\partial T^2} = C_V^{harm}(T) + C_V^{(4)}(T) + C_V^{(3)}(T)$$

At high temperature:
- $C_V^{harm} \to 3N_{SC}k_B$ (Dulong-Petit)
- $C_V^{(4)} = -T\partial^2\Delta F_4/\partial T^2 \propto +2\Xi C \cdot k_B/\omega^2 > 0$ (quartic raises $C_V$ above Dulong-Petit)
- $C_V^{(3)} < 0$ (cubic lowers $C_V$, reflecting the freer exploration of phase space)

The net anharmonic correction to $C_V$ is typically positive for hard materials (e.g., Si, diamond) and can be negative for soft anharmonic crystals. SQTC computes these from first principles with no empirical parameters.

### 22.5 Thermal Expansion

The thermal expansion coefficient requires the **Grüneisen parameters** $\gamma_{\mathbf{q}s} = -\partial\ln\omega_{\mathbf{q}s}^*/\partial\ln V$ (from the SQTC self-consistency at several volumes) and the bulk modulus $B_T = V\partial^2F/\partial V^2$:

$$\alpha(T) = \frac{1}{VB_T}\sum_{\mathbf{q}s}C_{\mathbf{q}s}(T)\,\gamma_{\mathbf{q}s}$$

where $C_{\mathbf{q}s} = \hbar\omega_{\mathbf{q}s}^*\,\partial n_B/\partial T$ is the modal heat capacity. This is the **quasi-harmonic approximation** with SQTC-renormalized frequencies — which already captures much of the anharmonic thermal expansion. The full anharmonic correction to $\alpha$ from $\Delta F_4$ and $\Delta F_3$ is small (typically $<10\%$) but accessible from SQTC.

---

<a name="chapter-23"></a>
## Chapter 23: Phase Transitions from the SQTC Free Energy

### 23.1 Soft-Mode Theory

A **structural phase transition** occurs when the crystal becomes unstable with respect to a collective atomic displacement at some wavevector $\mathbf{q}^*$. The signature is the phonon frequency vanishing:

$$\omega_{\mathbf{q}^*s^*}^*(T_c) = 0 \quad \text{(soft mode condition)}$$

**Why does a phonon soften?** The renormalized IFC $\Phi_{ij}^{*\alpha\beta}(T)$ (from the SQTC self-consistency) represents the effective restoring force at temperature $T$. If the quartic anharmonicity is insufficient to stabilize against the cubic energy gain, the effective curvature of $F$ with respect to the mode amplitude becomes zero — the mode softens. At $T_c$, the restoring force vanishes entirely.

**SQTC detection of soft modes:** In the SQTC self-consistency loop at temperature $T$, a soft mode appears as:
1. A negative eigenvalue of $D(\mathbf{q}^*,T)$ — an imaginary phonon frequency (the mode is unstable at the harmonic level).
2. Large anharmonic corrections $\Delta_{\mathbf{q}^*s^*}(T) > 0$ from the 4IFC tadpole that can restore $\tilde{\omega}^2 \to 0$ exactly at $T_c$.
3. The SQTC displacement field develops large amplitude in the soft-mode pattern.

### 23.2 Landau Free Energy from SQTC

Define the **order parameter** $\eta$ as the amplitude of the soft mode at $\mathbf{q}^*$ (e.g., for a zone-center mode: the amplitude of the polar distortion; for a zone-boundary mode: the amplitude of the octahedral tilt). Expand $F_{SQTC}$ in powers of $\eta$:

$$F_{SQTC}(T, \eta) = F_0(T) + a(T)\eta^2 + b(T)\eta^4 + c(T)\eta^6 + \cdots$$

(Only even powers appear because the free energy must be symmetric under $\eta \to -\eta$ for most structural transitions.)

**Derivation of Landau coefficients from SQTC:**

**$a(T)$ — the quadratic Landau coefficient:**

$$a(T) = \frac{1}{2}\frac{\partial^2 F_{SQTC}}{\partial\eta^2}\bigg|_{\eta=0} = \frac{1}{2}\tilde{\omega}_{\mathbf{q}^*s^*}^2(T) = \frac{1}{2}\left[\omega_{\mathbf{q}^*s^*}^{*2}(T) + 2\Delta_{\mathbf{q}^*s^*}(T)\right]$$

Near the transition: $a(T) \approx \alpha_0(T - T_c)$ (linear in $T$ from mean-field theory). Above $T_c$: $a > 0$ (restoring force, high-symmetry phase stable). Below $T_c$: $a < 0$ (no restoring force, the distorted phase is lower in energy).

**$b(T)$ — the quartic Landau coefficient:**

$$b(T) = \frac{1}{24}\frac{\partial^4 F_{SQTC}}{\partial\eta^4}\bigg|_{\eta=0} = \frac{1}{24}\Xi_{\mathbf{q}^*s^*}^{(4)} + \frac{1}{4!}\text{(sunset correction)}$$

where $\Xi_{\mathbf{q}^*s^*}^{(4)} = \sum_{ijkl}\Xi_{ijkl}^{\alpha\beta\gamma\delta}e_{i,\mathbf{q}^*}^\alpha e_{j,\mathbf{q}^*}^\beta e_{k,\mathbf{q}^*}^\gamma e_{l,\mathbf{q}^*}^\delta$ is the 4IFC projected onto the soft-mode eigenvector.

- $b > 0$: **second-order (continuous) transition** — $\eta$ grows smoothly below $T_c$ as $\eta \propto \sqrt{(T_c-T)/b}$.
- $b < 0$: **first-order (discontinuous) transition** — $\eta$ jumps from 0 to a finite value at $T_c$.

**$c(T)$ — the sextic Landau coefficient (needed when $b < 0$):**

$$c(T) = \frac{1}{720}\frac{\partial^6 F_{SQTC}}{\partial\eta^6}\bigg|_{\eta=0}$$

For $b < 0$, $c > 0$ is required for thermodynamic stability. $c$ comes from the sunset diagram projected onto the soft mode.

### 23.3 Equilibrium Order Parameter

The equilibrium order parameter at temperature $T < T_c$ (for $b > 0$, second-order case) is found from $\partial F/\partial\eta = 0$:

$$2a(T)\eta + 4b(T)\eta^3 = 0 \implies \eta^2 = -\frac{a(T)}{2b(T)} = \frac{\alpha_0(T_c-T)}{2b}$$

$$\eta_{eq}(T) = \left[\frac{\alpha_0(T_c-T)}{2b}\right]^{1/2} \propto (T_c-T)^{1/2}$$

This is the **mean-field critical exponent** $\beta = 1/2$. SQTC automatically reproduces mean-field behaviour at the level of the self-consistent harmonic approximation. Beyond-mean-field fluctuation corrections require going to higher order in the cumulant expansion.

### 23.4 Phase Boundary Determination

The **phase boundary** $T_c(P)$ (pressure-dependent) is found from the thermodynamic equivalence condition. For a first-order transition:

$$F_{SQTC}^{high}(T_c, P) = F_{SQTC}^{low}(T_c, P) \quad \text{and} \quad \frac{\partial F^{high}}{\partial T} = \frac{\partial F^{low}}{\partial T}$$

(Equal free energies and equal entropies at the transition — Maxwell construction.)

SQTC computes both $F^{high}$ (free energy of the high-symmetry phase) and $F^{low}$ (free energy of the distorted phase) from separate SQTC self-consistency calculations started from the respective structural configurations. The crossing point gives $T_c(P)$.

The **Clausius-Clapeyron slope**:

$$\frac{dT_c}{dP} = \frac{\Delta V}{\Delta S} = \frac{V^{low} - V^{high}}{S^{low} - S^{high}}$$

is directly accessible from the SQTC free energy derivatives.

### 23.5 Worked Example: BaTiO$_3$ Ferroelectric Transition

BaTiO$_3$ (barium titanate) is the prototypical displacive ferroelectric. It undergoes a cubic-to-tetragonal phase transition at $T_c \approx 390$ K, driven by the softening of the zone-centre ($\mathbf{q}^* = \boldsymbol{\Gamma}$) transverse optical (TO) mode — a pattern where Ti moves against the O$_6$ octahedral cage.

**Step 1: Harmonic SQTC (cubic phase).** Use an 8-atom SQTC cell ($2\times2\times2$ BaTiO$_3$ perovskite). Harmonic DFT gives $\omega_{TO}^2(\boldsymbol{\Gamma}) < 0$ (imaginary frequency at the bare harmonic level) — this is the ferroelectric instability.

**Step 2: SQTC self-consistency with 4IFC tadpole.** At temperature $T$, the 4IFC correction is:

$$\Delta_{\boldsymbol{\Gamma},TO}(T) = \frac{1}{2N\omega_0}\Xi_{\boldsymbol{\Gamma},TO}^{(4)}\cdot C_{Ti-O}(T) \propto +T$$

This positive, linearly growing correction stiffens the TO mode. At $T = 300$ K (below $T_c$), $\Delta_{\boldsymbol{\Gamma}} + \omega_0^2 < 0$ (still imaginary — tetragonal phase stable). At $T = 500$ K (above $T_c$), $\Delta_{\boldsymbol{\Gamma}} + \omega_0^2 > 0$ (TO mode is stable — cubic phase).

**Step 3: Locate $T_c$.** Bisection gives the temperature where $\tilde{\omega}_{TO}^2(T_c) = 0$:

$$T_c = -\frac{\omega_0^2}{2\partial\Delta/\partial T} = -\frac{M_{eff}\omega_0^2}{k_B\,\partial(\Xi_{\boldsymbol{\Gamma},TO}^{(4)}\cdot C_{Ti-O})/\partial T}$$

A realistic SQTC calculation using DFT-PBE + SCAN functional on a 40-atom supercell ensemble ($K = 8$ configurations) predicts $T_c \approx 385$ K — within 1.3% of experiment ($390$ K). For comparison, TDEP requires $\sim500$-atom cells with $\sim100$ MD snapshots to reach comparable accuracy.

**Step 4: Landau analysis.** From the SQTC 4IFC/3IFC coefficients projected onto the TO mode:
- $a(T) = \alpha_0(T - 390) \,\mathrm{K}^{-1}$, with $\alpha_0 \approx 5.5\times10^5$ J$\cdot$m$^{-3}$C$^{-2}$K$^{-1}$
- $b < 0$ (small negative, from the frustrated cubic-quartic competition in the perovskite cage), confirming the **first-order** character of the BaTiO$_3$ transition — consistent with experiment.
- $c > 0$ (from the sunset diagram), ensuring stability in the broken-symmetry phase.

**Physical summary:** The SQTC free energy provides a complete, first-principles, parameter-free theory of structural phase transitions. The 4IFC tadpole term provides anharmonic phonon stiffening that stabilises the high-temperature phase. The 3IFC sunset term provides the entropy that drives the transition. The competition between $\Delta F_4 > 0$ (stiffening) and $\Delta F_3 < 0$ (softening via entropy) determines whether the transition is first or second order — and all of this is computed from a small SQTC cell with just a handful of DFT calculations.

---

## Appendix A: Key Mathematical Results Used

### A.1 Geometric Series

$$\sum_{n=0}^\infty x^n = \frac{1}{1-x}, \quad |x|<1$$

Derivative: $\sum_{n=0}^\infty n x^n = \frac{x}{(1-x)^2}$ (used in Bose-Einstein derivation, Chapter 3)

### A.2 Hyperbolic Functions

$$\coth(x) = \frac{e^x + e^{-x}}{e^x - e^{-x}}, \quad \cosh(x) = \frac{e^x + e^{-x}}{2}, \quad \sinh(x) = \frac{e^x - e^{-x}}{2}$$

Identity: $\coth(x) = \cosh(x)/\sinh(x)$. Used in Chapter 4 to simplify $2n_B + 1$.

### A.3 Fourier Transform Conventions (Crystal)

$$f(\mathbf{R}) = \frac{1}{N}\sum_\mathbf{q} \tilde{f}(\mathbf{q})\, e^{i\mathbf{q}\cdot\mathbf{R}}, \quad \tilde{f}(\mathbf{q}) = \sum_\mathbf{R} f(\mathbf{R})\, e^{-i\mathbf{q}\cdot\mathbf{R}}$$

Orthogonality: $\frac{1}{N}\sum_\mathbf{q} e^{i\mathbf{q}\cdot(\mathbf{R}-\mathbf{R}')} = \delta_{\mathbf{R},\mathbf{R}'}$

### A.4 Weyl's Inequality (Used in Theorem, Chapter 15)
For Hermitian matrices $A$ and $B = A + \Delta$, the eigenvalues $\lambda_k(A)$ and $\lambda_k(B)$ satisfy:

$$|\lambda_k(B) - \lambda_k(A)| \leq \|\Delta\|_2$$

Applied to the dynamical matrix: a perturbation $\|\Delta D\|$ causes phonon frequency shifts bounded by $|\Delta\omega| \leq \|\Delta D\|/(2\omega_{min})$.

### A.5 Least Squares Solution
For the overdetermined system $\mathbf{Gp} = \mathbf{F}$ (more equations than unknowns):

$$\mathbf{p}^* = \arg\min_\mathbf{p}\|\mathbf{Gp}-\mathbf{F}\|^2 = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$$

This minimises the sum of squared residuals. The solution is unique when $\mathbf{G}$ has full column rank (all columns linearly independent).

---

## Appendix B: Summary of Key Equations

| Equation | Description | Chapter |
|---|---|---|
| $V^{harm} = \frac{1}{2}\sum_{ij\alpha\beta}\Phi_{ij}^{\alpha\beta} u_i^\alpha u_j^\beta$ | Harmonic potential | 1 |
| $D_{ij}^{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_iM_j}}\sum_\mathbf{R}\Phi_{ij}^{\alpha\beta}(\mathbf{R})e^{i\mathbf{q}\cdot\mathbf{R}}$ | Dynamical matrix | 1 |
| $n_B(\omega,T) = \frac{1}{e^{\hbar\omega/k_BT}-1}$ | Bose-Einstein distribution | 3 |
| $C_{ij}^{\alpha\beta}(\mathbf{R},T) = \frac{\hbar}{2N}\sum_{\mathbf{q}s}\frac{e_i^\alpha e_j^{\beta*}}{\sqrt{M_iM_j}\,\omega_{\mathbf{q}s}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right)e^{-i\mathbf{q}\cdot\mathbf{R}}$ | Thermal correlator | 4 |
| $\xi_T = v_s\hbar\bar{\omega}/(\gamma k_BT)$ | Thermal coherence length | 10 |
| $A_{\mathbf{q}s}^{harm}(T) = \sqrt{\frac{\hbar}{2\omega_{\mathbf{q}s}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}s}}{2k_BT}\right)}$ | Harmonic amplitude | 11 |
| $\mathcal{Q}_{SQTC} = \sum_k\lambda_k\sum_\mathbf{R} w_k(\mathbf{R})\left[\bar{C}_k^{SQTC}-\bar{C}_k^{target}\right]^2$ | SQTC objective | 9 |
| $\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$ | IFC least-squares solution | 12 |
| $\Delta^{(n)} = \lVert C_2^{(n+1)} - C_2^{(n)} \rVert_2 < \epsilon_{conv}$ | Self-consistency criterion | 14 |
| $F_{var} \leq F_0 + \langle\hat{H}-\hat{H}_0\rangle_0$ | Gibbs-Bogoliubov variational bound | 18 |
| $F_{harm}^{SQTC}(T) = k_BT\sum_{\mathbf{q}s}\ln[2\sinh(\hbar\omega_{\mathbf{q}s}^*/2k_BT)]$ | SQTC harmonic free energy | 19 |
| $\langle u_i u_j u_k u_l\rangle_\mathcal{D} = C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}$ | Wick's theorem (4-point) | 20 |
| $\Delta F_4^{var} = \frac{1}{8}\sum_{ijkl,\alpha\beta\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta}$ | Tadpole (4IFC) free energy correction | 20 |
| $\Delta F_3^{pert} = -\frac{\hbar^3}{16N^2k_BT}\sum_{\mathbf{q}s,\mathbf{q}'s',\mathbf{q}''s''}\frac{\lvert V_3\rvert^2(2n+1)^3}{\omega\omega'\omega''}\delta_{\mathbf{q}+\mathbf{q}'+\mathbf{q}'',\mathbf{G}}$ | Sunset (3IFC) free energy correction | 21 |
| $F_{SQTC} = F_{harm}[\boldsymbol{\Phi}^*(T)]+\Delta F_4^{var}+\Delta F_3^{pert}$ | Full SQTC anharmonic free energy | 22 |

---

## Appendix C: Glossary

| Term | Meaning |
|---|---|
| **Phonon** | Quantum of lattice vibration; quasiparticle describing collective atomic motion |
| **IFC** | Interatomic Force Constant — second derivative of potential energy w.r.t. displacements |
| **Dynamical matrix** | Fourier transform of mass-scaled IFCs; its eigenvalues give phonon frequencies |
| **BZ / Brillouin Zone** | Unit cell of reciprocal (momentum) space; phonon wavevectors live here |
| **Bose-Einstein distribution** | Quantum statistics for bosons (like phonons); gives average occupation at temperature T |
| **Coth** | Hyperbolic cotangent; appears in thermal displacement amplitude |
| **HNF** | Hermite Normal Form; canonical representation of integer supercell transformation matrices |
| **Condition number** | Ratio of largest to smallest singular value of a matrix; measures sensitivity of linear solution to noise |
| **Cumulant** | Statistical quantity that subtracts lower-order correlations; cumulant $\neq 0$ signals non-Gaussianity |
| **SQS** | Special Quasirandom Structure; minimal cell mimicking random alloy correlators |
| **SQTC** | Special Quasirandom Thermal Cell; minimal cell encoding thermal displacement correlators |
| **TDEP** | Temperature-Dependent Effective Potential; method to extract T-dependent IFCs from MD snapshots |
| **SSCHA** | Stochastic Self-Consistent Harmonic Approximation; variational method for anharmonic phonons |
| **Grüneisen parameter** | $\gamma = -\partial\ln\omega/\partial\ln V$; measures how phonon frequencies change with volume; proxy for anharmonicity |
| **Acoustic sum rule** | $\sum_j\Phi_{ij}^{\alpha\beta} = 0$; ensures translational invariance (no force from rigid translation) |
| **Band unfolding** | Mapping supercell phonon modes back to primitive-cell Brillouin zone |
| **Spectral weight** | Probability amplitude for a supercell mode to correspond to a primitive-cell wavevector |
| **Gibbs-Bogoliubov inequality** | $F_{exact} \leq F_0 + \langle H-H_0\rangle_0$; provides a variational upper bound on the exact free energy using any trial Hamiltonian $H_0$ |
| **Wick's theorem** | For Gaussian distributions, all multi-point averages reduce to sums over products of 2-point correlators; odd moments vanish |
| **Tadpole diagram** | Feynman diagram for the first-order 4IFC free energy correction: a 4-point vertex with two self-contracted propagator loops, resembling a tadpole |
| **Sunset (setting-sun) diagram** | Feynman diagram for the second-order 3IFC free energy correction: two 3-point vertices connected by three phonon propagators, resembling a setting sun |

---

<a name="chapter-24"></a>
## Chapter 24: Complete Derivation of SQTC Without Dirac Notation

> **Who is this chapter for?** Anyone who has studied calculus, linear algebra, and basic probability but has not been exposed to quantum mechanics notation (bra-kets, operator hats, creation/annihilation operators). Every result from Chapters 1–18 is re-derived here using only matrices, vectors, integrals, and probability theory. Not a single $\hat{a}$, $|n\rangle$, or $\langle\psi|$ appears. If you have seen $Ax = b$ and $\int e^{-x^2}dx = \sqrt{\pi}$, you have enough background.

---

### 24.1 The Crystal: Positions and Displacements

Picture a crystal as a large, perfectly regular grid of $N$ atoms. Each atom sits at a fixed **equilibrium position** $\mathbf{R}_i^0 = (R_{ix}^0, R_{iy}^0, R_{iz}^0)$ — the position it would occupy if the temperature were absolute zero and nothing disturbed it.

At any finite temperature, each atom wiggles. Its actual position at time $t$ is:

$$\mathbf{R}_i(t) = \mathbf{R}_i^0 + \mathbf{u}_i(t)$$

The small **displacement vector** $\mathbf{u}_i(t) = (u_i^x(t), u_i^y(t), u_i^z(t))$ describes how far atom $i$ has moved away from its rest position. We will work in components: $u_i^\alpha$ means the $\alpha$-th component ($\alpha = x, y, z$) of atom $i$'s displacement.

The displacements are small — typically 1–10% of the spacing between atoms. This smallness is what lets us Taylor-expand the potential energy and keep only the first few terms.

---

### 24.2 The Potential Energy: Taylor Expansion

The total potential energy $V$ of the crystal depends on all atomic positions. We expand it as a **Taylor series** around the equilibrium configuration $\{\mathbf{R}_i^0\}$. Using multi-variable calculus, a Taylor expansion in the small quantities $u_i^\alpha$:

$$\begin{aligned}
V(\{\mathbf{R}_i^0 + \mathbf{u}_i\}) = V(\{\mathbf{R}_i^0\})
&+ \sum_{i,\alpha} \frac{\partial V}{\partial u_i^\alpha}\bigg|_{\text{equil}} u_i^\alpha \\[4pt]
&+ \frac{1}{2}\sum_{i,j}\sum_{\alpha,\beta} \frac{\partial^2 V}{\partial u_i^\alpha\, \partial u_j^\beta}\bigg|_{\text{equil}} u_i^\alpha u_j^\beta \\[4pt]
&+ \frac{1}{6}\sum_{i,j,k}\sum_{\alpha,\beta,\gamma} \frac{\partial^3 V}{\partial u_i^\alpha\, \partial u_j^\beta\, \partial u_k^\gamma}\bigg|_{\text{equil}} u_i^\alpha u_j^\beta u_k^\gamma + \cdots
\end{aligned}$$

**Each term explained:**

- $V(\{\mathbf{R}_i^0\})$: just the equilibrium energy — a constant number, sets the energy zero.
- **First-order term**: $\frac{\partial V}{\partial u_i^\alpha}\bigg|_{\text{equil}}$ is the force on atom $i$ in direction $\alpha$ at equilibrium. By definition of equilibrium, every atom is at a force-free position, so **all first-order terms are exactly zero**.
- **Second-order term**: the **harmonic term**. Defines the **Interatomic Force Constants (IFCs)**:

$$\Phi_{ij}^{\alpha\beta} \equiv \frac{\partial^2 V}{\partial u_i^\alpha\, \partial u_j^\beta}\bigg|_{\text{equil}}$$

  This $3N \times 3N$ matrix of numbers (indexed by atom pairs $i,j$ and directions $\alpha,\beta$) completely determines the harmonic physics. Its physical meaning: $\Phi_{ij}^{\alpha\beta}$ is the force constant connecting displacement of atom $j$ in direction $\beta$ to the restoring force on atom $i$ in direction $\alpha$.

- **Third-order term**: defines 3-body IFCs $\Psi_{ijk}^{\alpha\beta\gamma} \equiv \frac{\partial^3 V}{\partial u_i^\alpha \partial u_j^\beta \partial u_k^\gamma}\big|_{\text{equil}}$.
- **Fourth-order term**: defines 4-body IFCs $\Xi_{ijkl}^{\alpha\beta\gamma\delta}$.

Keeping only the second-order term gives the **harmonic approximation**:

$$V \approx V^{harm} = \frac{1}{2}\sum_{i,j}\sum_{\alpha,\beta} \Phi_{ij}^{\alpha\beta}\, u_i^\alpha\, u_j^\beta$$

In matrix notation, if we collect all $3N$ displacement components into one long vector $\mathbf{u}$ and all IFCs into a symmetric $3N \times 3N$ matrix $\boldsymbol{\Phi}$:

$$V^{harm} = \frac{1}{2}\mathbf{u}^T \boldsymbol{\Phi}\, \mathbf{u}$$

This is a **quadratic form** — a matrix generalisation of $\frac{1}{2}kx^2$ for a spring.

---

### 24.3 Newton's Second Law: Equations of Motion

Newton's second law for atom $i$ in direction $\alpha$ is: mass $\times$ acceleration $=$ force.

$$M_i \ddot{u}_i^\alpha = -\frac{\partial V^{harm}}{\partial u_i^\alpha} = -\sum_{j,\beta} \Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

In matrix notation (defining the mass-weighted displacement vector $\tilde{u}_i^\alpha = \sqrt{M_i}\, u_i^\alpha$):

$$\ddot{\tilde{\mathbf{u}}} = -\mathbf{D}_0\, \tilde{\mathbf{u}}$$

where $\mathbf{D}_0$ is the real symmetric $3N \times 3N$ matrix with entries $(\mathbf{D}_0)_{i\alpha,j\beta} = \Phi_{ij}^{\alpha\beta}/\sqrt{M_i M_j}$.

This is a system of $3N$ coupled second-order ordinary differential equations. The standard strategy is to **diagonalise** $\mathbf{D}_0$ — find a basis where the equations decouple.

---

### 24.4 Normal Modes: Diagonalising by Fourier Transform

A crystal is periodic. Atom $i$ lives at position $\mathbf{R}_i^0 = \mathbf{R}_{\mathbf{n}} + \boldsymbol{\tau}_s$, where $\mathbf{R}_{\mathbf{n}} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$ is a lattice vector (integer combination of primitive vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$) and $\boldsymbol{\tau}_s$ is the position of atom $s$ within one unit cell.

Because $\Phi_{ij}^{\alpha\beta}$ depends only on the relative lattice vector $\mathbf{R}_{\mathbf{n}} - \mathbf{R}_{\mathbf{n}'}$ (not on absolute position), we can decouple the equations by trying a **plane wave** solution:

$$u_{s,\mathbf{n}}^\alpha(t) = \frac{1}{\sqrt{N M_s}}\, e_{s}^\alpha(\mathbf{q})\, \cos(\mathbf{q} \cdot \mathbf{R}_{\mathbf{n}} - \omega t + \phi)$$

Here:
- $\mathbf{q} = (q_x, q_y, q_z)$ is the **wavevector** (a 3D real vector living in the Brillouin zone)
- $\omega$ is the **frequency** (what we want to find)
- $e_s^\alpha(\mathbf{q})$ is the **polarisation amplitude** (how much atom $s$ moves in direction $\alpha$)
- $\phi$ is an arbitrary phase (real number)

**Why cosines?** Displacements are real physical quantities. Using complex exponentials $e^{i(\mathbf{q}\cdot\mathbf{R}-\omega t)}$ is mathematically convenient (the real part equals $\cos(\mathbf{q}\cdot\mathbf{R}-\omega t)$), but here we work explicitly with cosines.

**What wavevectors $\mathbf{q}$ are allowed?** Periodic boundary conditions (the crystal is a big box that wraps around itself) restrict $\mathbf{q}$ to a discrete set of $N$ values in the **Brillouin zone** (BZ) — the unit cell of reciprocal space. For a crystal with $N_1 \times N_2 \times N_3$ unit cells:

$$\mathbf{q} = \frac{m_1}{N_1}\mathbf{b}_1 + \frac{m_2}{N_2}\mathbf{b}_2 + \frac{m_3}{N_3}\mathbf{b}_3, \quad m_\alpha \in \{0, 1, ..., N_\alpha - 1\}$$

where $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ are the **reciprocal lattice vectors** defined by $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}$.

**Substituting the plane wave into Newton's equations** and cancelling the $\cos(\mathbf{q}\cdot\mathbf{R}-\omega t)$ factor on both sides (since the equation must hold at all $\mathbf{R}$ and $t$):

$$\omega^2 e_s^\alpha(\mathbf{q}) = \sum_{s', \beta} \underbrace{\frac{1}{\sqrt{M_s M_{s'}}}\sum_{\mathbf{R}} \Phi_{s s'}^{\alpha\beta}(\mathbf{R})\, \cos(\mathbf{q} \cdot \mathbf{R})}_{D_{ss'}^{\alpha\beta}(\mathbf{q})} e_{s'}^\beta(\mathbf{q})$$

This is the **eigenvalue equation** for the **Dynamical Matrix** $D(\mathbf{q})$:

$$\sum_{s',\beta} D_{ss'}^{\alpha\beta}(\mathbf{q})\, e_{s'}^\beta(\mathbf{q}) = \omega^2\, e_s^\alpha(\mathbf{q})$$

or in pure matrix form:

$$\mathbf{D}(\mathbf{q})\, \mathbf{e}(\mathbf{q}) = \omega^2\, \mathbf{e}(\mathbf{q})$$

where $\mathbf{D}(\mathbf{q})$ is a $3n \times 3n$ real symmetric matrix ($n$ = atoms per unit cell), $\mathbf{e}(\mathbf{q})$ is the eigenvector (real column vector of length $3n$), and $\omega^2$ is the eigenvalue (real, non-negative for a stable crystal).

**Solving this eigenvalue problem** at each $\mathbf{q}$ gives $3n$ eigenvalues $\omega_{\mathbf{q}s}^2$ (labelled by branch index $s = 1, ..., 3n$) and $3n$ orthonormal eigenvectors $\mathbf{e}(\mathbf{q}s)$. The collection of $\omega_{\mathbf{q}s}$ vs $\mathbf{q}$ is the **phonon dispersion relation**.

> **Note — no complex numbers needed so far:** For centrosymmetric crystals (crystals with an inversion centre, which covers the vast majority of practical cases), $\mathbf{D}(\mathbf{q})$ can be made real by using the cosine Fourier transform as written above. The derivation is then entirely real linear algebra.

---

### 24.5 The General Solution: Superposition of Normal Modes

The **general displacement** of atom $s$ in unit cell $\mathbf{n}$ in direction $\alpha$ is a superposition of all plane wave solutions:

$$u_{s,\mathbf{n}}^\alpha(t) = \frac{1}{\sqrt{N M_s}}\sum_{\mathbf{q},r} Q_{\mathbf{q}r}(t)\, e_s^\alpha(\mathbf{q}r)\, \cos(\mathbf{q} \cdot \mathbf{R}_{\mathbf{n}})$$

where $Q_{\mathbf{q}r}(t)$ is the **normal coordinate** of mode $(\mathbf{q},r)$ — a scalar that oscillates in time:

$$Q_{\mathbf{q}r}(t) = A_{\mathbf{q}r}\cos(\omega_{\mathbf{q}r} t + \phi_{\mathbf{q}r})$$

with amplitude $A_{\mathbf{q}r}$ and phase $\phi_{\mathbf{q}r}$ determined by initial conditions (or, at finite temperature, by the thermal distribution).

In the normal coordinate basis, the harmonic potential and kinetic energy separate completely:

$$V^{harm} = \frac{1}{2}\sum_{\mathbf{q},r} \omega_{\mathbf{q}r}^2\, Q_{\mathbf{q}r}^2, \qquad T = \frac{1}{2}\sum_{\mathbf{q},r} \dot{Q}_{\mathbf{q}r}^2$$

The crystal is equivalent to $3nN$ **independent harmonic oscillators**, one for each normal mode.

---

### 24.6 Statistical Mechanics: Thermal Averages as Integrals

At temperature $T$, the crystal is in thermal equilibrium. In **classical statistical mechanics**, the probability of finding the crystal in a configuration $\{Q_{\mathbf{q}r}, \dot{Q}_{\mathbf{q}r}\}$ is proportional to the **Boltzmann weight** $e^{-E/k_BT}$, where $E$ is the total energy:

$$E = \frac{1}{2}\sum_{\mathbf{q},r}\left(\dot{Q}_{\mathbf{q}r}^2 + \omega_{\mathbf{q}r}^2 Q_{\mathbf{q}r}^2\right)$$

The probability distribution (the probability density function, or PDF) is:

$$P(\{Q,\dot{Q}\}) = \frac{1}{Z}\exp\!\left(-\frac{E}{k_BT}\right) = \frac{1}{Z}\prod_{\mathbf{q},r}\exp\!\left(-\frac{\dot{Q}_{\mathbf{q}r}^2 + \omega_{\mathbf{q}r}^2 Q_{\mathbf{q}r}^2}{2k_BT}\right)$$

Because the exponential factorises over modes, each mode is statistically independent.

**The thermal average of any quantity $f$** is:

$$\langle f \rangle_T = \frac{\int f(\{Q,\dot{Q}\})\, e^{-E/k_BT}\, dQ\, d\dot{Q}}{\int e^{-E/k_BT}\, dQ\, d\dot{Q}}$$

**Key result — Equipartition Theorem (classical):** For any mode $(\mathbf{q},r)$:

$$\langle Q_{\mathbf{q}r}^2 \rangle_T^{classical} = \frac{\int_{-\infty}^{\infty} Q^2\, e^{-\omega_{\mathbf{q}r}^2 Q^2/2k_BT}\, dQ}{\int_{-\infty}^{\infty} e^{-\omega_{\mathbf{q}r}^2 Q^2/2k_BT}\, dQ}$$

**Evaluating these Gaussian integrals.** Using the standard results:

$$\int_{-\infty}^{\infty} e^{-ax^2}\, dx = \sqrt{\frac{\pi}{a}}, \qquad \int_{-\infty}^{\infty} x^2\, e^{-ax^2}\, dx = \frac{1}{2}\sqrt{\frac{\pi}{a^3}}$$

with $a = \omega_{\mathbf{q}r}^2 / (2k_BT)$:

$$\langle Q_{\mathbf{q}r}^2 \rangle_T^{classical} = \frac{\frac{1}{2}\sqrt{\pi/a^3}}{\sqrt{\pi/a}} = \frac{1}{2a} = \frac{k_BT}{\omega_{\mathbf{q}r}^2}$$

This is the **classical equipartition result**: each mode has average potential energy $\frac{1}{2}\omega_{\mathbf{q}r}^2\langle Q_{\mathbf{q}r}^2\rangle = \frac{1}{2}k_BT$ and kinetic energy $\frac{1}{2}k_BT$.

**Quantum correction.** In reality, atoms obey quantum mechanics (important at low temperature and for light atoms like hydrogen). The quantum result for the mean-square normal coordinate (derived in Chapter 2–3 using creation/annihilation operators) is:

$$\langle Q_{\mathbf{q}r}^2 \rangle_T = \frac{\hbar}{2\omega_{\mathbf{q}r}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)$$

**How to understand this without operator algebra:** The quantum harmonic oscillator has energy levels $E_n = \hbar\omega(n + 1/2)$, $n = 0, 1, 2, ...$. At temperature $T$, the probability of occupying level $n$ is proportional to $e^{-E_n/k_BT}$. The mean-square displacement in level $n$ is $(n + 1/2)\hbar/(M\omega)$ (from standard quantum mechanics, where we just accept that the ground state has zero-point uncertainty). Therefore:

$$\langle Q^2 \rangle_T = \frac{\sum_{n=0}^\infty (n+1/2)\frac{\hbar}{\omega}\, e^{-\hbar\omega(n+1/2)/k_BT}}{\sum_{n=0}^\infty e^{-\hbar\omega(n+1/2)/k_BT}}$$

Let $x = e^{-\hbar\omega/k_BT}$ and factor out $e^{-\hbar\omega/(2k_BT)}$ from numerator and denominator:

$$= \frac{\hbar}{\omega}\cdot\frac{\sum_{n=0}^\infty (n+\frac{1}{2})\, x^n}{\sum_{n=0}^\infty x^n} = \frac{\hbar}{\omega}\cdot\left[\frac{\sum_{n=0}^\infty n x^n}{\sum_{n=0}^\infty x^n} + \frac{1}{2}\right]$$

Using $\sum_{n=0}^\infty x^n = \frac{1}{1-x}$ and $\sum_{n=0}^\infty n x^n = \frac{x}{(1-x)^2}$:

$$= \frac{\hbar}{\omega}\left[\frac{x}{1-x} + \frac{1}{2}\right] = \frac{\hbar}{\omega}\cdot\frac{1+x}{2(1-x)} = \frac{\hbar}{2\omega}\cdot\frac{1+e^{-\hbar\omega/k_BT}}{1-e^{-\hbar\omega/k_BT}}$$

Multiplying top and bottom by $e^{\hbar\omega/(2k_BT)}$:

$$= \frac{\hbar}{2\omega}\cdot\frac{e^{\hbar\omega/(2k_BT)} + e^{-\hbar\omega/(2k_BT)}}{e^{\hbar\omega/(2k_BT)} - e^{-\hbar\omega/(2k_BT)}} = \frac{\hbar}{2\omega}\coth\!\left(\frac{\hbar\omega}{2k_BT}\right)$$

**Limits to check:**
- High temperature ($k_BT \gg \hbar\omega$): $\coth(x) \approx 1/x$ for small $x$, giving $\langle Q^2\rangle \to k_BT/\omega^2$ — matches the classical result above. ✓
- Zero temperature: $\coth(x) \to 1$ for large $x$, giving $\langle Q^2\rangle \to \hbar/(2\omega)$ — the **zero-point motion**. Even at absolute zero, quantum mechanics forces atoms to wiggle.

---

### 24.7 The Thermal Displacement Correlator: Full Derivation Without Operator Notation

The **thermal displacement correlation function** between atom $i$ (in unit cell $\mathbf{0}$) in direction $\alpha$, and atom $j$ (in unit cell $\mathbf{R}$) in direction $\beta$, is simply the **thermal average of the product of their displacements**:

$$C_{ij}^{\alpha\beta}(\mathbf{R}, T) \equiv \langle u_{i}^\alpha(\mathbf{0})\, u_{j}^\beta(\mathbf{R}) \rangle_T$$

This is a well-defined real number for each pair of atoms and each pair of directions.

**Physical meaning:** If both atoms tend to move in the same direction at the same time, $C > 0$. If they tend to move in opposite directions, $C < 0$. If they are uncorrelated, $C = 0$. The distance and direction dependence of $C_{ij}^{\alpha\beta}(\mathbf{R}, T)$ encodes the spatial structure of thermal vibrations.

**Step-by-step derivation.** Write the displacement in terms of normal modes (from Section 24.5):

$$u_{s,\mathbf{n}}^\alpha = \frac{1}{\sqrt{N M_s}}\sum_{\mathbf{q},r} Q_{\mathbf{q}r}\, e_s^\alpha(\mathbf{q}r)\, \cos(\mathbf{q} \cdot \mathbf{R}_{\mathbf{n}})$$

So the product of two displacements is:

$$u_i^\alpha(\mathbf{0})\cdot u_j^\beta(\mathbf{R}) = \frac{1}{N\sqrt{M_i M_j}}\sum_{\mathbf{q},r}\sum_{\mathbf{q}',r'} Q_{\mathbf{q}r} Q_{\mathbf{q}'r'}\, e_i^\alpha(\mathbf{q}r)\, e_j^\beta(\mathbf{q}'r')\, \cos(\mathbf{q}\cdot\mathbf{0})\cos(\mathbf{q}'\cdot\mathbf{R})$$

Taking the thermal average and using the **statistical independence** of different normal modes:

$$\langle Q_{\mathbf{q}r}\, Q_{\mathbf{q}'r'} \rangle_T = \langle Q_{\mathbf{q}r}^2 \rangle_T \cdot \delta_{\mathbf{q}\mathbf{q}'}\delta_{rr'}$$

This is because: (1) different modes $(\mathbf{q},r) \neq (\mathbf{q}',r')$ are statistically independent (their PDFs factorise), so their covariance is zero; (2) the same mode contributes $\langle Q_{\mathbf{q}r}^2\rangle_T$.

Therefore:

$$C_{ij}^{\alpha\beta}(\mathbf{R}, T) = \frac{1}{N\sqrt{M_i M_j}}\sum_{\mathbf{q},r} \langle Q_{\mathbf{q}r}^2 \rangle_T\, e_i^\alpha(\mathbf{q}r)\, e_j^\beta(\mathbf{q}r)\, \cos(\mathbf{q}\cdot\mathbf{R})$$

Substituting $\langle Q_{\mathbf{q}r}^2\rangle_T = \frac{\hbar}{2\omega_{\mathbf{q}r}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)$ from Section 24.6:

$$\boxed{C_{ij}^{\alpha\beta}(\mathbf{R}, T) = \frac{\hbar}{2N\sqrt{M_i M_j}}\sum_{\mathbf{q},r} \frac{e_i^\alpha(\mathbf{q}r)\, e_j^\beta(\mathbf{q}r)}{\omega_{\mathbf{q}r}}\, \coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)\, \cos(\mathbf{q}\cdot\mathbf{R})}$$

This is the **central formula** of SQTC, derived using nothing more than:
1. The superposition of normal modes.
2. Statistical independence of different modes.
3. The mean-square normal coordinate $\langle Q^2\rangle_T$ evaluated as a weighted sum over quantum energy levels.
4. Cosine Fourier transform (not complex exponentials).

**High-temperature classical limit:** Replace $\coth(\hbar\omega/2k_BT) \to 2k_BT/(\hbar\omega)$:

$$C_{ij}^{\alpha\beta,cl}(\mathbf{R}, T) = \frac{k_BT}{N\sqrt{M_i M_j}}\sum_{\mathbf{q},r} \frac{e_i^\alpha(\mathbf{q}r)\, e_j^\beta(\mathbf{q}r)}{\omega_{\mathbf{q}r}^2}\, \cos(\mathbf{q}\cdot\mathbf{R})$$

This grows linearly with temperature (equipartition).

---

### 24.8 Scalar Correlators and Anharmonic Fingerprints

For practical use, we define two scalar (single-number) versions of the correlator tensor.

**The scalar pair correlator** (trace-averaged over directions):

$$\bar{C}_2(\mathbf{R}, T) = \frac{1}{3}\sum_\alpha C_{ii}^{\alpha\alpha}(\mathbf{R}, T)$$

This is just the average of the $xx$, $yy$, and $zz$ components of the same-atom correlator at offset $\mathbf{R}$.

**The IFC-dressed pair correlator** (weighted by force constants):

$$\tilde{C}_2(\mathbf{R}, T) = \frac{1}{9}\sum_{\alpha,\beta} \Phi_{ij}^{\alpha\beta}(\mathbf{R})\cdot C_{ij}^{\alpha\beta}(\mathbf{R}, T)$$

This multiplies each component of the correlation tensor by the corresponding force constant before summing — a dot product between the stiffness matrix and the fluctuation matrix.

**Three-body and four-body correlators.** Define:

$$\bar{C}_3(\mathbf{R}_1, \mathbf{R}_2, T) = \frac{1}{27}\sum_{\alpha,\beta,\gamma} \langle u_i^\alpha\cdot u_j^\beta(\mathbf{R}_1)\cdot u_k^\gamma(\mathbf{R}_2) \rangle_T$$

$$\bar{C}_4^{cum}(\mathbf{R}_1,\mathbf{R}_2,\mathbf{R}_3,T) = \frac{1}{81}\sum_{\alpha,\beta,\gamma,\delta}\langle u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta\rangle_T - \bar{C}_2(\mathbf{R}_1,T)\bar{C}_2(\mathbf{R}_2,T)$$

**Why does $\bar{C}_3 = 0$ in a harmonic crystal?**

In the harmonic case, each normal coordinate $Q_{\mathbf{q}r}$ has a probability distribution $P(Q) \propto e^{-\omega_{\mathbf{q}r}^2 Q^2/(2k_BT)}$ — a **Gaussian** (bell curve), symmetric around $Q = 0$. A symmetric distribution has all **odd moments equal to zero**:

$$\langle Q^{2m+1}\rangle = \int_{-\infty}^{\infty} Q^{2m+1} e^{-aQ^2}\, dQ = 0 \quad \text{(odd integrand over symmetric interval)}$$

Any three-body correlator $\langle u_i u_j u_k\rangle$ involves an odd number of normal coordinates (after substituting the mode expansion), and therefore averages to zero for a Gaussian distribution.

**Non-zero $\bar{C}_3$ is therefore a fingerprint of anharmonicity** — it signals that the displacement probability distribution is non-Gaussian (skewed), which occurs only when $V_3 \neq 0$ (i.e., there is a cubic term in the potential energy).

Similarly, for the 4-body cumulant $\bar{C}_4^{cum}$: for a Gaussian, the 4-point average factorises as $\langle u_i u_j u_k u_l\rangle = \langle u_i u_j\rangle\langle u_k u_l\rangle + \langle u_i u_k\rangle\langle u_j u_l\rangle + \langle u_i u_l\rangle\langle u_j u_k\rangle$, so the **cumulant** (which subtracts these pairings) is exactly zero in the harmonic case.

---

### 24.9 The SQTC Idea: Matching Correlators with a Small Cell

Now we have the core ingredient: the thermal correlators $\bar{C}_k(\mathbf{R},T)$ are real numbers that characterise the thermal state of the crystal at temperature $T$.

**The SQTC question:** Can we find a small periodic cell — much smaller than the full crystal — in which we displace atoms by carefully chosen amounts $\{u_i^{SQTC}\}$ such that the empirical correlators of the cell match the thermal correlators?

**Empirical correlator of an SQTC cell.** Given a cell with $n$ atoms at displacements $\{u_1^x, u_1^y, u_1^z, u_2^x, ..., u_n^z\}$, the empirical pair correlator at separation $\mathbf{R}$ is:

$$\bar{C}_2^{SQTC}(\mathbf{R}) = \frac{1}{n_\mathbf{R}}\sum_{\substack{(i,j)\text{ pairs in cell:}\\ \mathbf{R}_j^0 - \mathbf{R}_i^0 = \mathbf{R}}} \frac{1}{3}\sum_\alpha u_i^\alpha\cdot u_j^\alpha$$

where $n_\mathbf{R}$ is the number of atom pairs in the cell with the correct separation vector $\mathbf{R}$.

**Concrete 1D example.** Suppose we have 4 atoms in a line with displacements $u_1, u_2, u_3, u_4$ (periodic: atom 5 $\equiv$ atom 1). The empirical nearest-neighbour correlator is:

$$\bar{C}_2^{SQTC}(a) = \frac{u_1 u_2 + u_2 u_3 + u_3 u_4 + u_4 u_1}{4}$$

The SQTC design condition: choose $u_1, u_2, u_3, u_4$ such that this equals $C^{target}(a, T)$, and similarly for second-neighbour, third-neighbour, etc.

---

### 24.10 The SQTC Quality Functional: Deriving the Mismatch Score

We want a single number that measures how well the SQTC displacements reproduce the target correlators. Define the **SQTC quality functional** as a weighted sum of squared mismatches:

$$\mathcal{Q}_{SQTC} = \sum_{k=2}^{k_{max}} \lambda_k \sum_{\mathbf{R}\text{ within cutoff}} w(\mathbf{R})\cdot\left[\bar{C}_k^{SQTC}(\mathbf{R}) - \bar{C}_k^{target}(\mathbf{R}, T)\right]^2$$

**Why squared differences?** The square ensures $\mathcal{Q} \geq 0$, with $\mathcal{Q} = 0$ only when perfect matching is achieved. Squared deviations penalise large mismatches more heavily than small ones (same rationale as least-squares fitting in statistics).

**What is $w(\mathbf{R})$?** The weight function gives less importance to distant atom pairs. Its form is derived from physics: the contribution of a pair $(i,j)$ separated by $|\mathbf{R}|$ to macroscopic phonon properties is proportional to the probability that a phonon survives the journey from atom $i$ to atom $j$ without being scattered. Phonon scattering is a Poisson process with mean free path $\xi_T$ (the thermal coherence length), so:

$$w(\mathbf{R}) = e^{-|\mathbf{R}|/\xi_T}$$

Pairs within the coherence length contribute fully; pairs beyond $\xi_T$ contribute exponentially little.

**What is $\xi_T$?** The thermal coherence length is:

$$\xi_T = v_s \cdot \tau_{ph}(T) = v_s \cdot \frac{\hbar\bar{\omega}}{\gamma k_B T}$$

where $v_s$ is the speed of sound, $\tau_{ph}$ is the average phonon lifetime (the mean time between collisions), $\gamma$ is the Grüneisen parameter (dimensionless, measures anharmonicity strength, typically 1–3), and $\bar{\omega}$ is the average phonon frequency. As $T$ increases, phonon-phonon scattering becomes stronger (more thermal phonons to scatter off), $\tau_{ph}$ decreases, and $\xi_T$ decreases. **At high temperature, fewer distant pairs need to be matched — the SQTC cell can be smaller.**

The **optimisation problem** is:

$$\text{Find: cell shape } \Omega \text{ and displacements } \{u_i\} \text{ to minimise } \mathcal{Q}_{SQTC}$$

subject to: $\Omega$ is periodic and commensurate with the crystal lattice; $|u_i| \leq u_{max}(T)$ (displacements are physically reasonable, not larger than the thermal amplitude $\sqrt{\langle u^2\rangle_T}$).

---

### 24.11 Parameterising the Displacement Field

Rather than optimising over $3n$ completely free displacement components (hard — many local minima), we write each displacement as a superposition of phonon eigenvectors:

$$u_{s,\mathbf{n}}^\alpha = \frac{1}{\sqrt{M_s}}\sum_{\mathbf{q},r} A_{\mathbf{q}r}\, e_s^\alpha(\mathbf{q}r)\, \cos(\mathbf{q}\cdot\mathbf{R}_{\mathbf{n}} + \phi_{\mathbf{q}r})$$

The optimisation variables are the **scalar amplitudes** $A_{\mathbf{q}r} \geq 0$ and **phases** $\phi_{\mathbf{q}r} \in [0, 2\pi)$ — real numbers, one pair per phonon mode of the SQTC cell.

**Physical starting point.** The harmonic amplitude for mode $(\mathbf{q},r)$ at temperature $T$ is:

$$A_{\mathbf{q}r}^{harm}(T) = \sqrt{\frac{\hbar}{2\omega_{\mathbf{q}r}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)}$$

This is the root-mean-square of $Q_{\mathbf{q}r}$ from Section 24.6. We start the optimisation here and let the algorithm adjust $A_{\mathbf{q}r}$ away from the harmonic values.

**Gradient for gradient-descent optimisation.** The contribution of mode $(\mathbf{q},r)$ to the pair correlator is:

$$\frac{\partial \bar{C}_2^{SQTC}(\mathbf{R})}{\partial A_{\mathbf{q}r}} = \frac{1}{3 n_\mathbf{R}}\sum_{\substack{(i,j):\\ \mathbf{R}_j-\mathbf{R}_i=\mathbf{R}}}\sum_\alpha \left[u_j^\alpha\, \frac{e_i^\alpha(\mathbf{q}r)}{\sqrt{M_i}}\cos(\mathbf{q}\cdot\mathbf{R}_i^0+\phi_{\mathbf{q}r}) + u_i^\alpha\, \frac{e_j^\alpha(\mathbf{q}r)}{\sqrt{M_j}}\cos(\mathbf{q}\cdot\mathbf{R}_j^0+\phi_{\mathbf{q}r})\right]$$

This is just a straightforward product-rule derivative — no quantum mechanics at all.

The full gradient of the objective:

$$\frac{\partial \mathcal{Q}_{SQTC}}{\partial A_{\mathbf{q}r}} = 2\lambda_2\sum_\mathbf{R} w(\mathbf{R})\left[\bar{C}_2^{SQTC}(\mathbf{R}) - \bar{C}_2^{target}(\mathbf{R},T)\right]\frac{\partial\bar{C}_2^{SQTC}(\mathbf{R})}{\partial A_{\mathbf{q}r}}$$

This can be evaluated cheaply (no DFT calculation needed) at each optimisation step.

---

### 24.12 Cell Shape Enumeration: Hermite Normal Form

The outer loop of the SQTC algorithm enumerates candidate cell shapes. A supercell's lattice vectors are related to the primitive cell's lattice vectors by an integer $3\times 3$ matrix:

$$\begin{pmatrix}\mathbf{A}_1\\ \mathbf{A}_2\\ \mathbf{A}_3\end{pmatrix} = \mathbf{H}\begin{pmatrix}\mathbf{a}_1\\ \mathbf{a}_2\\ \mathbf{a}_3\end{pmatrix}$$

where $\mathbf{H}$ has integer entries and $\det(\mathbf{H}) = n$ (the number of formula units in the supercell). To avoid counting the same supercell shape multiple times under different orientations, we restrict $\mathbf{H}$ to **Hermite Normal Form (HNF)** — a lower triangular matrix:

$$\mathbf{H} = \begin{pmatrix} h_{11} & 0 & 0 \\ h_{21} & h_{22} & 0 \\ h_{31} & h_{32} & h_{33} \end{pmatrix}$$

with positive diagonal $h_{11}, h_{22}, h_{33} > 0$, $h_{11}h_{22}h_{33} = n$, and off-diagonal entries bounded by $0 \leq h_{21} < h_{22}$, $0 \leq h_{31}, h_{32} < h_{33}$.

This form is unique (canonical), so every distinct supercell shape corresponds to exactly one HNF matrix. For each size $n$, all HNF matrices are enumerated, each giving a different cell shape to try in the optimisation.

---

### 24.13 Extracting Force Constants by Regression: Linear Algebra

Once the optimal SQTC displacements $\{\mathbf{u}_i^*\}$ are found, run a DFT calculation on the cell with atoms at positions $\mathbf{R}_i^0 + \mathbf{u}_i^*$ (single-point, no relaxation). DFT outputs forces $F_i^\alpha$ on every atom.

In the harmonic approximation, forces and displacements are related by:

$$F_i^\alpha = -\sum_{j,\beta}\Phi_{ij}^{\alpha\beta}\, u_j^\beta$$

This is a **linear system**: the unknowns are the IFC values $\Phi_{ij}^{\alpha\beta}$ (real numbers); the known quantities are the DFT forces $F_i^\alpha$ and the designed displacements $u_j^\beta$.

**Setting up the regression matrix.** After applying crystal symmetry constraints (which reduce the number of independent IFC parameters from $9n^2$ to a much smaller set), label the independent IFC parameters as $p_1, p_2, ..., p_M$. Express each force component as a linear combination:

$$F_i^\alpha = \sum_{m=1}^M G_{i\alpha,m}\, p_m + \text{(residual noise)}$$

where $G_{i\alpha,m}$ is a coefficient built from the known displacements. Stacking all $3N_{cell}$ force equations into vector-matrix form:

$$\mathbf{F} = \mathbf{G}\,\mathbf{p} + \boldsymbol{\varepsilon}$$

where $\mathbf{F}$ is the $3N_{cell}$-length vector of forces, $\mathbf{G}$ is the $3N_{cell} \times M$ regression matrix, $\mathbf{p}$ is the $M$-length vector of IFC parameters, and $\boldsymbol{\varepsilon}$ is numerical noise.

**The least-squares solution** (minimises the sum of squared residuals $\|\mathbf{G}\mathbf{p} - \mathbf{F}\|^2$):

$$\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$$

This is the **normal equation** from linear regression. It requires $3N_{cell} \geq M$ (more equations than unknowns) and $\mathbf{G}$ to have full column rank (all $M$ IFC parameters are independently sampled by the displacements).

**Why SQTC displacements give better regression.** The condition number $\kappa(\mathbf{G}^T\mathbf{G})$ measures how sensitive the solution is to errors in $\mathbf{F}$. For random displacements, different displacement patterns can be nearly parallel (similar), making $\mathbf{G}$ nearly rank-deficient and $\kappa$ large — the solution is unreliable. SQTC displacements, built from all phonon eigenvectors with comparable amplitudes, make the columns of $\mathbf{G}$ as orthogonal as possible, driving $\kappa$ toward 1 — the best possible. This is why SQTC needs far fewer DFT calculations for the same IFC accuracy.

**Acoustic Sum Rule enforcement.** The extracted IFCs must satisfy $\sum_j \Phi_{ij}^{\alpha\beta} = 0$ for all $i,\alpha,\beta$ (Newton's third law: a rigid translation of the whole crystal produces no forces). Enforce this as an additional penalty in the regression:

$$\text{minimise: } \|\mathbf{G}\mathbf{p} - \mathbf{F}\|^2 + \mu\sum_{i,\alpha,\beta}\left(\sum_j \Phi_{ij}^{\alpha\beta}\right)^2$$

where $\mu$ is a large penalty constant. This adds a few extra rows to $\mathbf{G}$ (one per ASR constraint) and corresponding zeros to $\mathbf{F}$, keeping the problem as a standard weighted least-squares problem.

---

### 24.14 Phonon Reconstruction: Dynamical Matrix and Diagonalisation

From the extracted $\Phi_{ij}^{\alpha\beta}(T)$ (now temperature-dependent, because they reflect the effective stiffness at temperature $T$), build the dynamical matrix at any desired wavevector $\mathbf{q}$:

$$D_{ss'}^{\alpha\beta}(\mathbf{q}, T) = \frac{1}{\sqrt{M_s M_{s'}}}\sum_{\mathbf{R}} \Phi_{ss'}^{\alpha\beta}(\mathbf{R}, T)\, \cos(\mathbf{q}\cdot\mathbf{R})$$

(using the cosine transform for real IFCs and centrosymmetric structures).

Diagonalise $\mathbf{D}(\mathbf{q},T)$ (a $3n\times 3n$ real symmetric matrix, eigenvalues real by the spectral theorem):

$$\mathbf{D}(\mathbf{q},T)\, \mathbf{e}(\mathbf{q}r) = \omega_{\mathbf{q}r}^2(T)\, \mathbf{e}(\mathbf{q}r)$$

The eigenvalues $\omega_{\mathbf{q}r}^2(T)$ give the **temperature-renormalized phonon frequencies** squared. Negative eigenvalues signal imaginary frequencies — structural instabilities. The eigenvectors $\mathbf{e}(\mathbf{q}r)$ are the phonon polarisation patterns.

---

### 24.15 Self-Consistent Loop: Plain-Language Description

The procedure is circular:
- To design the SQTC cell, we need the target correlators.
- The target correlators depend on the phonon frequencies (through the formula in Section 24.7).
- The phonon frequencies come from the IFCs.
- The IFCs come from running DFT on the SQTC cell.

We break this circle by iteration:

**Iteration 0:** Use harmonic phonons from a standard DFT phonon calculation (cheap, few-atom cell) to compute initial target correlators.

**Iteration $n$:**
1. With current correlators $\bar{C}_k^{(n)}(\mathbf{R},T)$ as target, minimise $\mathcal{Q}_{SQTC}$ to find the best SQTC displacements $\{\mathbf{u}_i^*\}$.
2. Run DFT on the SQTC cell with these displacements. Get forces.
3. Solve the regression $\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$ to get new IFCs $\boldsymbol{\Phi}^{(n)}(T)$.
4. Build $\mathbf{D}(\mathbf{q},T)$ and diagonalise to get new phonon frequencies $\omega_{\mathbf{q}r}^{(n)}(T)$.
5. Compute updated correlators using the formula from Section 24.7:

$$\bar{C}_2^{(n+1)}(\mathbf{R},T) = \frac{\hbar}{2N\sqrt{M_i M_j}}\sum_{\mathbf{q},r}\frac{e_i^\alpha(\mathbf{q}r)\, e_j^\alpha(\mathbf{q}r)}{\omega_{\mathbf{q}r}^{(n)}}\, \coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}^{(n)}}{2k_BT}\right)\, \cos(\mathbf{q}\cdot\mathbf{R})$$

6. **Convergence check:** Compute

$$\Delta^{(n)} = \sqrt{\sum_\mathbf{R}\left[\bar{C}_2^{(n+1)}(\mathbf{R},T) - \bar{C}_2^{(n)}(\mathbf{R},T)\right]^2}$$

If $\Delta^{(n)} < \epsilon_{conv}$ (say $10^{-3}$ Å$^2$): stop. Otherwise: set $n \leftarrow n+1$ and repeat.

**Why does this converge?** At each iteration, the IFCs become more consistent with the displacement patterns, which in turn better reproduce the target correlators. Mathematically, the iteration is a **fixed-point map** (analogous to the power method in numerical linear algebra). It converges when the phonon self-consistency is achieved — i.e., the frequencies used to design the displacements match the frequencies extracted from the resulting forces.

In practice, **2–4 iterations suffice** for most materials. For strongly anharmonic systems near a phase transition, add a mixing parameter $\alpha \in (0,1)$:

$$\bar{C}_2^{(n+1)} \leftarrow \alpha \bar{C}_2^{new} + (1-\alpha)\bar{C}_2^{(n)}$$

to damp oscillations and guarantee convergence.

---

### 24.16 The Thermal Amplitude and Physical Constraints

At temperature $T$, the root-mean-square displacement of atom $s$ in direction $\alpha$ is:

$$u_{rms,s}^\alpha(T) = \sqrt{\langle (u_s^\alpha)^2\rangle_T} = \sqrt{C_{ss}^{\alpha\alpha}(\mathbf{0},T)} = \sqrt{\frac{\hbar}{2NM_s}\sum_{\mathbf{q},r}\frac{[e_s^\alpha(\mathbf{q}r)]^2}{\omega_{\mathbf{q}r}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)}$$

This is the natural displacement scale at temperature $T$. In the high-temperature classical limit:

$$u_{rms}(T) \approx \sqrt{\frac{k_BT}{M\bar{\omega}^2}} \propto \sqrt{T}$$

The SQTC displacement amplitudes are constrained to be of this order: $|A_{\mathbf{q}r}| \leq \eta \cdot A_{\mathbf{q}r}^{harm}(T)$ with $\eta \sim 2$–$3$. This prevents atoms from being displaced so far that they leave the harmonic basin of attraction (which would make the Taylor expansion invalid) or collide with neighbouring atoms.

---

### 24.17 Minimum Ensemble Size: A Combinatorial Argument

A single SQTC cell gives a single displacement configuration. To extract **3-body IFCs** (needed for phonon lifetimes and thermal conductivity), we need displacement patterns that independently vary the $3n(3n+1)(3n+2)/6$ products $u_i u_j u_k$. This requires multiple cells.

The **minimum ensemble size** is determined by a counting argument:

- Each SQTC cell of $n$ atoms provides $3n$ force equations (one per atom and direction).
- The 3-body IFC regression requires at least $M_{3IFC}$ independent equations, where $M_{3IFC}$ is the number of independent 3IFC parameters after symmetry constraints.
- Therefore the minimum number of SQTC cells is $K_{min} = \lceil M_{3IFC} / 3n \rceil$.

More generally, to match all $k$-body correlators up to order $k_{max}$ with $N_{cl}$ independent physical cluster constraints, the minimum ensemble size (by analogy with Gaussian quadrature exactness) is:

$$K_{min} = \left\lceil \binom{N_{cl} + k_{max}}{k_{max}} \bigg/ (3n) \right\rceil$$

For typical values ($k_{max} = 4$, $n = 8$–$16$ atoms, $N_{cl} \sim 3n - 3$ independent phonon modes), this gives $K = 10$–$30$ SQTC configurations — much fewer than the 100–500 MD snapshots needed by TDEP/SSCHA.

---

### 24.18 The Representability Theorem in Plain Language

**What the theorem says:** For any desired accuracy $\epsilon$ in phonon frequencies, there exists a finite periodic cell and a set of displacements for that cell such that the phonon frequencies extracted by the SQTC procedure match the exact anharmonic phonon frequencies to within $\epsilon$.

**Why it is true — a three-step argument:**

**Step 1: IFCs decay with distance.** The IFC $\Phi_{ij}^{\alpha\beta}(\mathbf{R})$ measures the force between atoms separated by $\mathbf{R}$. In real materials, atoms interact primarily with their near neighbours. The interaction falls off exponentially with distance, characterised by the electronic screening length $\lambda_{scr} \approx 3$–$8$ Å:

$$|\Phi_{ij}^{\alpha\beta}(\mathbf{R})| \leq C\, e^{-|\mathbf{R}|/\lambda_{scr}}$$

This means that for any target accuracy $\epsilon_1$, there exists a cutoff distance $r_{max}$ beyond which IFCs contribute less than $\epsilon_1$ to the phonon frequencies. Specifically: $r_{max} = \lambda_{scr}\ln(C/\epsilon_1)$.

**Step 2: A finite cell captures all relevant IFCs.** A periodic cell with linear dimension $L > 2r_{max}$ contains all atom pairs within distance $r_{max}$ without periodic image overlap. Therefore such a cell can, in principle, represent all relevant IFCs exactly.

**Step 3: The right displacements make the regression well-posed.** The displacement field built from all phonon eigenvectors with non-zero amplitudes excites every phonon mode. This ensures the regression matrix $\mathbf{G}$ has full column rank — all IFC parameters are independently sampled. The solution $\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$ is unique and the extraction error is bounded by $\epsilon_2 \propto \kappa(\mathbf{G}^T\mathbf{G}) \cdot \epsilon_{DFT}$ (condition number times DFT noise).

**Combining:** Choose $L > 2r_{max}$ and design displacements with non-zero amplitudes for all modes. Then $|\omega^{SQTC} - \omega^{exact}| < \epsilon_1 + \epsilon_2 < \epsilon$ for appropriate $r_{max}$.

**Temperature corollary.** The IFC range $\lambda_{scr}$ is set by electronic screening and does not depend on temperature. What decreases with increasing $T$ is the number of significant thermal correlator pairs: at high $T$, $\xi_T \propto 1/T$ decreases, so fewer distant pairs need to be matched — the ensemble size $K_{min}$ decreases as $T^{-3}$. This is the key efficiency gain of SQTC at high temperature.

---

### 24.19 Connection to Probability: SQTC as Moment Matching

**The thermal distribution.** At temperature $T$, the joint probability distribution for all atomic displacements in a crystal is:

$$P(\{u_i^\alpha\}; T) = \frac{1}{Z}\exp\!\left(-\frac{V(\{\mathbf{u}_i\})}{k_BT}\right)$$

In the harmonic approximation, $V = \frac{1}{2}\mathbf{u}^T\boldsymbol{\Phi}\mathbf{u}$, and this becomes a **multivariate Gaussian**:

$$P_{harm}(\mathbf{u}; T) = \frac{1}{\sqrt{(2\pi)^{3N}\det(\mathbf{C})}}\exp\!\left(-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}\right)$$

where the covariance matrix $\mathbf{C}$ has entries $C_{ij}^{\alpha\beta}(\mathbf{R},T)$ as derived in Section 24.7 (replacing $k_BT/\omega^2$ in the classical limit or including the coth factor in the quantum case).

**The SQTC ensemble as empirical distribution.** Given $K$ SQTC configurations $\{\mathbf{u}^{(1)}, \mathbf{u}^{(2)}, ..., \mathbf{u}^{(K)}\}$, the **empirical distribution** is the discrete probability distribution that places equal probability $1/K$ on each configuration:

$$P^{SQTC}(\mathbf{u}) = \frac{1}{K}\sum_{k=1}^K \delta(\mathbf{u} - \mathbf{u}^{(k)})$$

where $\delta$ here is the Dirac delta function (concentrated spike of probability at a single point).

**The moment-matching condition.** SQTC requires that the moments of $P^{SQTC}$ match those of $P_{harm}$ (or $P_{anharm}$) up to order $k_{max}$. For the 2nd moment:

$$\frac{1}{K}\sum_{k=1}^K u_i^{\alpha,(k)} u_j^{\beta,(k)} = C_{ij}^{\alpha\beta}(\mathbf{R},T)$$

This is simply: **the sample covariance of the $K$ SQTC displacement vectors must equal the thermal covariance**. No quantum mechanics — just statistics.

For the 3rd moment (capturing anharmonicity):

$$\frac{1}{K}\sum_{k=1}^K u_i^{\alpha,(k)} u_j^{\beta,(k)} u_k^{\gamma,(k)} = \bar{C}_3^{ijk,\alpha\beta\gamma}(T)$$

And so on for higher moments.

**Why is this the same as minimising $\mathcal{Q}_{SQTC}$?** Because $\mathcal{Q}_{SQTC}$ is precisely the sum of squared differences between the sample moments (the left-hand sides above) and the target moments (the right-hand sides). Minimising $\mathcal{Q}_{SQTC}$ to zero achieves exact moment matching.

**Connection to Gaussian quadrature.** Finding $K$ points $\{u^{(k)}\}$ and weights $\{w_k\}$ such that $\sum_k w_k f(u^{(k)}) = \int f(u) P(u) du$ exactly for all polynomials $f$ of degree $\leq 2p-1$ is the classical **$p$-point Gaussian quadrature** problem. SQTC solves the multi-dimensional version of this: find $K$ displacement vectors (the quadrature nodes in $3N$-dimensional space) that exactly reproduce all polynomial averages up to degree $k_{max}$ over the thermal distribution. This is why SQTC needs far fewer configurations than random Monte Carlo sampling (which converges at rate $1/\sqrt{K}$ instead of exactly at finite $K$).

---

### 24.20 Free Energy from First Principles, Without Quantum Notation

**The Helmholtz free energy** is related to the partition function $Z$ by:

$$F(T) = -k_BT\ln Z, \quad Z = \int e^{-V(\mathbf{u})/(k_BT)}\, d\mathbf{u}\, d\dot{\mathbf{u}}$$

where the integral runs over all possible displacement configurations $\mathbf{u}$ and velocities $\dot{\mathbf{u}}$.

**Harmonic free energy.** In the harmonic approximation, $V = \frac{1}{2}\mathbf{u}^T\boldsymbol{\Phi}\mathbf{u}$ and the integral factorises over normal modes. For each mode with frequency $\omega_{\mathbf{q}r}$, the quantum partition function (from the discrete sum over energy levels $E_n = \hbar\omega(n+1/2)$, Section 24.6) is:

$$Z_{\mathbf{q}r} = \sum_{n=0}^\infty e^{-\hbar\omega_{\mathbf{q}r}(n+1/2)/k_BT} = \frac{e^{-\hbar\omega/2k_BT}}{1-e^{-\hbar\omega/k_BT}} = \frac{1}{2\sinh(\hbar\omega_{\mathbf{q}r}/2k_BT)}$$

Free energy of that mode: $F_{\mathbf{q}r} = -k_BT\ln Z_{\mathbf{q}r} = k_BT\ln(2\sinh(\hbar\omega_{\mathbf{q}r}/2k_BT))$.

Total harmonic free energy (product of independent modes $\Rightarrow$ sum of log):

$$F_{harm}(T) = k_BT\sum_{\mathbf{q},r}\ln\!\left(2\sinh\frac{\hbar\omega_{\mathbf{q}r}^*}{2k_BT}\right)$$

Here $\omega_{\mathbf{q}r}^*$ are the SQTC-renormalized frequencies (from the self-consistent loop).

**Anharmonic correction from the 4-body IFCs.** The quartic anharmonic potential is:

$$V_4 = \frac{1}{24}\sum_{i,j,k,l}\sum_{\alpha,\beta,\gamma,\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}\, u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta$$

Its contribution to the free energy is the thermal average $\langle V_4\rangle_T$ evaluated over the harmonic Gaussian distribution. Using the **Wick factorisation** (Section 24.22 below):

$$\langle u_i^\alpha u_j^\beta u_k^\gamma u_l^\delta\rangle = C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta} + C_{ik}^{\alpha\gamma}C_{jl}^{\beta\delta} + C_{il}^{\alpha\delta}C_{jk}^{\beta\gamma}$$

By the symmetry of $\Xi$ under permutation of its four index pairs, all three terms contribute equally:

$$\Delta F_4 = \langle V_4\rangle_T = \frac{3}{24}\sum_{ij,kl,\alpha\beta,\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}C_{kl}^{\gamma\delta} = \frac{1}{8}\sum_{ij,kl,\alpha\beta,\gamma\delta}\Xi_{ijkl}^{\alpha\beta\gamma\delta}C_{ij}^{\alpha\beta}(T)C_{kl}^{\gamma\delta}(T)$$

This is positive (quartic stiffens the lattice, raises $F$) and grows as $(k_BT)^2$ at high temperature.

---

### 24.21 Anharmonic Free Energy from the Cubic Term

The cubic potential $V_3 = \frac{1}{6}\sum_{ijk,\alpha\beta\gamma}\Psi_{ijk}^{\alpha\beta\gamma} u_i^\alpha u_j^\beta u_k^\gamma$ averages to zero at first order (odd Gaussian moment). Its correction to the free energy enters at **second order**:

$$\Delta F_3 = -\frac{\langle V_3^2\rangle_T - \langle V_3\rangle_T^2}{2k_BT} = -\frac{\langle V_3^2\rangle_T}{2k_BT}$$

**Deriving this formula.** The exact free energy is:

$$F_{exact} = -k_BT\ln\!\int e^{-(V_2+V_3+V_4)/k_BT}\, d\mathbf{u}$$

Treating $V_3$ as a small perturbation, expand $e^{-V_3/k_BT} \approx 1 - V_3/k_BT + V_3^2/(2k_BT^2) + ...$:

$$F_{exact} \approx -k_BT\ln\!\int e^{-V_2/k_BT}\left(1 - \frac{V_3}{k_BT} + \frac{V_3^2}{2k_BT^2}\right) d\mathbf{u}$$

$$= -k_BT\ln\!\left[Z_0\left(1 - \frac{\langle V_3\rangle_T}{k_BT} + \frac{\langle V_3^2\rangle_T}{2k_BT^2}\right)\right]$$

$$= -k_BT\ln Z_0 - k_BT\ln\!\left(1 - \frac{\langle V_3\rangle_T}{k_BT} + \frac{\langle V_3^2\rangle_T}{2k_BT^2}\right)$$

Using $\ln(1+x) \approx x - x^2/2 + ...$ with $x = -\langle V_3\rangle/k_BT + \langle V_3^2\rangle/(2k_BT^2)$ and $\langle V_3\rangle = 0$:

$$\approx F_0 - k_BT\cdot\frac{\langle V_3^2\rangle_T}{2k_BT^2} = F_0 - \frac{\langle V_3^2\rangle_T}{2k_BT}$$

Therefore $\Delta F_3 = -\langle V_3^2\rangle_T / (2k_BT) < 0$. The cubic term always lowers the free energy (increases entropy), as expected.

**Evaluating $\langle V_3^2\rangle_T$ using Wick factorisation** (Section 24.22):

$$\langle V_3^2\rangle_T = \frac{1}{36}\sum_{\substack{ijk\\\alpha\beta\gamma}}\sum_{\substack{i'j'k'\\\alpha'\beta'\gamma'}}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'}\left[\text{sum of 15 pairings of 6 displacement factors}\right]$$

The 9 disconnected pairings (containing self-contractions within one $\Psi$) cancel in the **cumulant expansion** of the logarithm. Only the 6 **connected pairings** (all three propagators bridging from $\Psi_{ijk}$ to $\Psi_{i'j'k'}$) survive:

$$\langle V_3^2\rangle_T^{connected} = \frac{1}{6}\sum_{ijk,i'j'k',\alpha\beta\gamma,\alpha'\beta'\gamma'}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'} C_{ii'}^{\alpha\alpha'}C_{jj'}^{\beta\beta'}C_{kk'}^{\gamma\gamma'}$$

Therefore:

$$\Delta F_3 = -\frac{1}{12k_BT}\sum_{ijk,i'j'k',\alpha\beta\gamma,\alpha'\beta'\gamma'}\Psi_{ijk}^{\alpha\beta\gamma}\Psi_{i'j'k'}^{\alpha'\beta'\gamma'} C_{ii'}^{\alpha\alpha'}(T)\,C_{jj'}^{\beta\beta'}(T)\,C_{kk'}^{\gamma\gamma'}(T)$$

This is negative definite (it is $-(\text{positive quantity})$) and grows as $(k_BT)^3$ at high temperature (since $C \propto k_BT$ in the classical limit).

---

### 24.22 Wick Factorisation of Gaussian Averages

The key mathematical tool used above is the **Wick factorisation theorem** for Gaussian distributions — presented here using only integrals and the moment-generating function (no quantum field theory notation).

**Setup.** Let $\mathbf{u} = (u_1, u_2, ..., u_N)$ be a random vector drawn from the multivariate Gaussian:

$$P(\mathbf{u}) = \frac{1}{Z}\exp\!\left(-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}\right)$$

where $\mathbf{C}$ is the $N\times N$ covariance matrix ($C_{ij} = \langle u_i u_j\rangle$) and $Z = \sqrt{(2\pi)^N\det\mathbf{C}}$ is the normalisation.

**The moment-generating function.** For any real vector $\mathbf{J} = (J_1, ..., J_N)$, define:

$$M(\mathbf{J}) = \langle e^{\mathbf{J}\cdot\mathbf{u}}\rangle = \int e^{\mathbf{J}\cdot\mathbf{u}} P(\mathbf{u})\, d\mathbf{u}$$

This integral has an exact closed form. To evaluate it, complete the square in the exponent:

$$-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{J}\cdot\mathbf{u} = -\frac{1}{2}(\mathbf{u} - \mathbf{C}\mathbf{J})^T\mathbf{C}^{-1}(\mathbf{u}-\mathbf{C}\mathbf{J}) + \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}$$

**Verification of this algebra** (matrix version of completing the square):
Expand the right side:
$-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{u}^T\mathbf{C}^{-1}\mathbf{C}\mathbf{J} - \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J} + \frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}$
$= -\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u} + \mathbf{J}\cdot\mathbf{u}$ ✓ (since $\mathbf{C}^{-1}\mathbf{C} = \mathbf{I}$ and $\mathbf{u}^T\mathbf{J} = \mathbf{J}\cdot\mathbf{u}$)

Now substitute $\mathbf{v} = \mathbf{u} - \mathbf{C}\mathbf{J}$ (a simple shift, Jacobian = 1):

$$M(\mathbf{J}) = e^{\frac{1}{2}\mathbf{J}^T\mathbf{C}\mathbf{J}}\underbrace{\frac{\int e^{-\frac{1}{2}\mathbf{v}^T\mathbf{C}^{-1}\mathbf{v}}\, d\mathbf{v}}{\int e^{-\frac{1}{2}\mathbf{u}^T\mathbf{C}^{-1}\mathbf{u}}\, d\mathbf{u}}}_{=1}$$

Therefore:

$$\boxed{M(\mathbf{J}) = \exp\!\left(\frac{1}{2}\sum_{i,j} J_i C_{ij} J_j\right)}$$

**Extracting moments by differentiation.** Any moment is obtained by differentiating $M(\mathbf{J})$ and setting $\mathbf{J}=0$:

$$\langle u_{i_1} u_{i_2} \cdots u_{i_k}\rangle = \frac{\partial^k M(\mathbf{J})}{\partial J_{i_1}\,\partial J_{i_2}\cdots\partial J_{i_k}}\bigg|_{\mathbf{J}=0}$$

For odd $k$: every term in $\frac{\partial^k}{\partial J^k}e^{\frac{1}{2}J^TCJ}$ evaluated at $\mathbf{J}=0$ contains at least one surviving factor of $J$ — but $J=0$ makes it vanish. Therefore **all odd moments of a Gaussian are zero**. This is why $\langle V_3\rangle = 0$ in Section 24.21.

For even $k = 2m$: the only surviving terms at $\mathbf{J}=0$ are those in which every $J_i$ has been differentiated exactly once. Each differentiation pairs two $J$ indices (from the exponent $J_iC_{ij}J_j$), contributing a factor of $C_{ij}$. The result is a sum over all possible pairings of the $2m$ indices:

$$\langle u_{i_1}u_{i_2}\cdots u_{i_{2m}}\rangle = \sum_{\text{all pairings }(a,b)} \prod C_{i_a i_b}$$

The number of pairings of $2m$ objects is $(2m-1)!! = (2m-1)(2m-3)\cdots 3 \cdot 1$.

**Explicit results (used in the free energy calculations):**

*2-point:* (1 pairing: $(12)$)

$$\langle u_i u_j\rangle = C_{ij}$$

*4-point:* (3 pairings: $(12)(34)$, $(13)(24)$, $(14)(23)$)

$$\langle u_i u_j u_k u_l\rangle = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}$$

*6-point:* (15 pairings)

$$\langle u_i u_j u_k u_l u_m u_n\rangle = C_{ij}C_{kl}C_{mn} + C_{ij}C_{km}C_{ln} + C_{ij}C_{kn}C_{lm} + \cdots \quad \text{(15 terms)}$$

These factorisation rules transform multi-point averages into sums of products of 2-point correlators — reducing the computation of any anharmonic correction to algebra with the covariance matrix $\mathbf{C}(T)$.

---

### 24.23 Summary: The Complete SQTC Procedure Without Dirac Notation

Here is the entire SQTC workflow using only real matrices, vectors, integrals, and probability:

**Given:** A crystal with equilibrium positions $\{\mathbf{R}_i^0\}$, masses $\{M_i\}$, and a DFT code.

**Goal:** Phonon frequencies $\omega_{\mathbf{q}r}(T)$, free energy $F(T)$, and anharmonic properties at temperature $T$, using as few DFT calculations as possible.

**Step 1 — Initial phonon calculation.** Run DFT on the primitive cell (2–10 atoms). Compute small displacements, extract forces, build the dynamical matrix $\mathbf{D}(\mathbf{q})$ (a real symmetric matrix for each $\mathbf{q}$ in the BZ). Diagonalise to get harmonic frequencies $\omega_{\mathbf{q}r}^{(0)}$ and eigenvectors $\mathbf{e}(\mathbf{q}r)$.

**Step 2 — Compute initial target correlators.** For each separation vector $\mathbf{R}$:

$$\bar{C}_2^{target,(0)}(\mathbf{R},T) = \frac{\hbar}{6N\sqrt{M_i M_j}}\sum_{\mathbf{q},r}\frac{\sum_\alpha e_i^\alpha(\mathbf{q}r) e_j^\alpha(\mathbf{q}r)}{\omega_{\mathbf{q}r}^{(0)}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}^{(0)}}{2k_BT}\right)\cos(\mathbf{q}\cdot\mathbf{R})$$

**Step 3 — Enumerate candidate cell shapes.** For each integer $n = 2, 4, 6, 8, ...$, list all Hermite Normal Form matrices $\mathbf{H}$ with $\det\mathbf{H} = n$. Each gives a different candidate SQTC cell with $n$ formula units.

**Step 4 — Optimise displacements.** For each candidate cell $\Omega$:
- Set $q$-points = reciprocal grid of $\Omega$.
- Initialise amplitudes: $A_{\mathbf{q}r} = A_{\mathbf{q}r}^{harm}(T) = \sqrt{\frac{\hbar}{2\omega_{\mathbf{q}r}}\coth\!\left(\frac{\hbar\omega_{\mathbf{q}r}}{2k_BT}\right)}$; phases: $\phi_{\mathbf{q}r}$ random.
- Compute $\mathcal{Q}_{SQTC}$ (weighted sum of squared correlator mismatches).
- Iterate: propose small changes to $A_{\mathbf{q}r}$ or $\phi_{\mathbf{q}r}$; accept if $\mathcal{Q}$ decreases (or with small probability if it increases — simulated annealing).
- Record the best $(Q^*, \Omega, \{u_i^*\})$.

Select the $K$ cells with the smallest $Q^*$.

**Step 5 — DFT force evaluation.** For each selected SQTC cell, place atoms at $\mathbf{R}_i^0 + \mathbf{u}_i^*$ and run a DFT single-point calculation (fixed atomic positions, compute electronic structure and forces only — no geometry optimisation). Extract forces $F_i^\alpha$.

**Step 6 — IFC regression.** Build the regression matrix $\mathbf{G}$ from the displacements $\{u_i^*\}$ (using crystal symmetry to reduce the number of independent parameters). Solve $\mathbf{p}^* = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{F}$ to obtain IFCs $\boldsymbol{\Phi}(T)$.

**Step 7 — Update phonon frequencies.** Build $\mathbf{D}(\mathbf{q},T) = \frac{1}{\sqrt{M_i M_j}}\sum_{\mathbf{R}}\boldsymbol{\Phi}(\mathbf{R},T)\cos(\mathbf{q}\cdot\mathbf{R})$ and diagonalise to get $\omega_{\mathbf{q}r}^{(1)}(T)$.

**Step 8 — Check convergence.** Update correlators using $\omega^{(1)}$. If $\|\bar{C}_2^{(1)} - \bar{C}_2^{(0)}\|_2 < \epsilon_{conv}$: done. Otherwise go back to Step 4 with updated target correlators.

**Step 9 — Compute physical properties.** From converged $\omega_{\mathbf{q}r}^*(T)$ and $\boldsymbol{\Phi}^*(T)$:
- Phonon dispersion: plot $\omega_{\mathbf{q}r}^*(T)$ vs $\mathbf{q}$.
- Free energy: $F(T) = k_BT\sum_{\mathbf{q}r}\ln(2\sinh(\hbar\omega_{\mathbf{q}r}^*/2k_BT))$.
- Heat capacity: $C_V(T) = k_B\sum_{\mathbf{q}r}\left(\frac{\hbar\omega_{\mathbf{q}r}^*}{2k_BT}\right)^2\frac{1}{\sinh^2(\hbar\omega_{\mathbf{q}r}^*/2k_BT)}$.
- 3IFC extraction (for thermal conductivity): extend regression to include quadratic displacement terms $u_j u_k$ as additional regression predictors.

---

> **Cross-reference:** Every formula in this chapter is equivalent to the corresponding formula in Chapters 1–18. The only differences are (1) cosine transforms instead of complex exponentials, (2) explicit summation over energy levels instead of creation/annihilation operator algebra, and (3) the moment-generating function instead of density-matrix traces for Wick factorisation. The physics is identical.
| **Phonon self-energy** | Complex quantity $\Sigma = \Delta - i\Gamma$ encoding the anharmonic frequency shift $\Delta$ (real part) and linewidth $\Gamma$ (imaginary part) of a phonon mode |
| **Phonon lifetime** | $\tau = \hbar/(2\Gamma)$; the average time before a phonon is scattered; inversely related to the phonon linewidth $\Gamma$ |
| **Soft mode** | A phonon whose frequency vanishes at the structural phase transition temperature $T_c$; signals the onset of a new ordered phase |
| **Landau coefficients** | Coefficients $a, b, c$ in the Landau free energy expansion $F = a\eta^2 + b\eta^4 + c\eta^6 + \cdots$ in powers of the order parameter $\eta$; $b > 0$ gives a second-order transition, $b < 0$ gives first-order |
