#!/usr/bin/env python3
"""Schematic of the SOTC-D workflow (Fig. 1 of the JAP manuscript).

The insets are computed for a two-dimensional square lattice with
nearest- and next-nearest-neighbor springs (4x4 cell, unit masses), using
the same construction as the method: commensurate modes of a harmonic
trial, a canonical draw with Gaussian mode coordinates, and an SOTC-D
configuration with every mode at exactly its rms amplitude and its
inverse. The dispersions of panels (a) and (e) are those of the toy trial
and of the same lattice with stiffened springs; they illustrate, not
quantify, the thermal renormalization.

Writes manuscript/jap/figures/fig_workflow.pdf.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_robustness import BLUE, GRAY, INK, MUTED, ORANGE  # noqa: E402

OUT = ROOT / "manuscript" / "jap" / "figures"
L = 4  # cell edge (atoms)
K1, K2 = 1.0, 0.35  # nearest- and next-nearest-neighbor springs of the trial
KT = 0.02  # thermal energy of the illustration (classical amplitudes)


def lattice_fc(k1, k2):
    """Force constants of the periodic L x L square lattice with central springs."""
    n = L * L
    phi = np.zeros((n, 2, n, 2))
    pos = np.array([(i, j) for i in range(L) for j in range(L)], float)
    for a in range(n):
        for d, k in [((1, 0), k1), ((0, 1), k1), ((1, 1), k2), ((1, -1), k2)]:
            b = int(((pos[a, 0] + d[0]) % L) * L + (pos[a, 1] + d[1]) % L)
            e = np.array(d, float) / np.linalg.norm(d)
            block = k * np.outer(e, e)
            phi[a, :, b, :] -= block
            phi[b, :, a, :] -= block
            phi[a, :, a, :] += block
            phi[b, :, b, :] += block
    return phi.reshape(2 * n, 2 * n), pos


def branches(k1, k2, q):
    """Two branches along Gamma-X of the infinite lattice."""
    w = []
    for qx in q:
        d = np.zeros((2, 2))
        for dv, k in [((1, 0), k1), ((0, 1), k1), ((1, 1), k2), ((1, -1), k2)]:
            e = np.array(dv, float) / np.linalg.norm(dv)
            d += 2 * k * np.outer(e, e) * (1 - np.cos(qx * dv[0]))
        w.append(np.sqrt(np.clip(np.linalg.eigvalsh(d), 0, None)))
    return np.array(w)


def configurations(seed=3):
    phi, pos = lattice_fc(K1, K2)
    w2, vec = np.linalg.eigh(phi)
    keep = w2 > 1e-8  # drop the two translations
    w2, vec = w2[keep], vec[:, keep]
    sigma = np.sqrt(KT / w2)
    rng = np.random.default_rng(seed)
    q_canon = rng.normal(size=len(sigma))  # Gaussian mode coordinates / sigma
    signs = np.where(rng.random(len(sigma)) < 0.5, -1.0, 1.0)
    u = (vec @ (signs * sigma)).reshape(-1, 2)
    return pos, u, q_canon, signs, np.sqrt(w2)


def panel_box(fig, x, y, w, h, title, color):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor="#f6f5f2",
                                      edgecolor=color, linewidth=0.8, zorder=-5))
    fig.text(x + 0.008, y + h - 0.012, title, fontsize=7.2, color=INK, va="top",
             fontweight="bold")


def arrow(fig, p, q, text=None, rad=0.0, color=MUTED, text_xy=None):
    fig.patches.append(FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=9, linewidth=1.0, color=color,
                                       connectionstyle=f"arc3,rad={rad}"))
    if text:
        tx, ty = text_xy or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        fig.text(tx, ty, text, fontsize=6.3, color=color, ha="center", va="center")


def add_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.0)
    return ax


def main():
    plt.rcParams.update({"font.size": 6.0, "axes.labelsize": 6.5, "xtick.labelsize": 6.0,
                         "ytick.labelsize": 5.6, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    pos, u, q_canon, signs, freqs = configurations()
    fig = plt.figure(figsize=(7.0, 2.75))

    xs = [0.008, 0.205, 0.402, 0.599, 0.796]
    w, y0, h = 0.188, 0.235, 0.735
    titles = ["(a) Harmonic trial", "(b) Mode amplitudes", "(c) Designed $\\pm$ pair",
              "(d) DFT forces and fit", "(e) Effective FC2"]
    colors = [GRAY, BLUE, BLUE, MUTED, ORANGE]
    for x, t, c in zip(xs, titles, colors):
        panel_box(fig, x, y0, w, h, t, c)

    # (a) trial dispersion and mode amplitudes
    q = np.linspace(0, np.pi, 80)
    ax = add_axes(fig, [xs[0] + 0.03, y0 + 0.25, w - 0.045, 0.33])
    for b in branches(K1, K2, q).T:
        ax.plot(q, b, color=GRAY, lw=1.1)
    ax.set_xticks([0, np.pi], ["$\\Gamma$", "X"])
    ax.set_yticks([])
    ax.set_ylabel("$\\omega$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(xs[0] + w / 2, y0 + 0.13,
             "modes $\\omega_m$, $\\mathbf{e}_m$; rms amplitude\n"
             "$\\sigma_m^2=\\frac{\\hbar}{2\\omega_m}\\coth\\frac{\\hbar\\omega_m}{2k_BT}$",
             fontsize=5.6, ha="center", va="center", color=INK, linespacing=1.35)

    # (b) mode coordinates: canonical draw versus SOTC-D
    ax = add_axes(fig, [xs[1] + 0.05, y0 + 0.25, w - 0.062, 0.33])
    m = np.arange(len(signs))
    ax.axhline(0, color=MUTED, lw=0.5)
    for lev in (-1, 1):
        ax.axhline(lev, color=BLUE, lw=0.5, ls=":")
    ax.vlines(m - 0.18, 0, q_canon, color=GRAY, lw=1.6, label="canonical draw")
    ax.vlines(m + 0.18, 0, signs, color=BLUE, lw=1.6, label="SOTC-D")
    ax.set_ylim(-2.6, 2.6)
    ax.set_yticks([-1, 0, 1], ["$-\\sigma_m$", "0", "$\\sigma_m$"])
    ax.set_xticks([])
    ax.set_xlabel("mode $m$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=5.4, frameon=False, handlelength=0.8,
              borderaxespad=0.0, bbox_to_anchor=(1.02, 1.12), ncol=2, columnspacing=0.6)
    fig.text(xs[1] + w / 2, y0 + 0.13,
             "each mode at exactly $\\pm\\sigma_m$;\nsigns $s_m$ maximize "
             "$\\det\\mathbf{X}^{\\rm T}\\mathbf{X}$",
             fontsize=5.6, ha="center", va="center", color=INK, linespacing=1.35)

    # (c) the configuration and its inverse
    scale = 2.2 / np.abs(u).max() * 0.18
    for k, (sgn, label) in enumerate([(1, "$+\\mathbf{u}$"), (-1, "$-\\mathbf{u}$")]):
        ax = add_axes(fig, [xs[2] + 0.012 + k * 0.088, y0 + 0.25, 0.082, 0.36])
        ax.scatter(pos[:, 0], pos[:, 1], s=7, color="#c9c7c1", zorder=1)
        disp = pos + sgn * scale * u
        ax.scatter(disp[:, 0], disp[:, 1], s=9, color=BLUE, zorder=3)
        ax.quiver(pos[:, 0], pos[:, 1], sgn * scale * u[:, 0], sgn * scale * u[:, 1],
                  angles="xy", scale_units="xy", scale=1, width=0.012, color=BLUE, zorder=2)
        ax.set_xlim(-0.7, L - 0.3)
        ax.set_ylim(-0.7, L - 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=6.8, pad=1)
    fig.text(xs[2] + w / 2, y0 + 0.13,
             "$u_i=M_i^{-1/2}\\sum_m s_m\\sigma_m e_{im}$\n"
             "$\\pm$ pair cancels the cubic force",
             fontsize=5.6, ha="center", va="center", color=INK, linespacing=1.35)

    # (d) forces and least-squares fit
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 40)
    ax = add_axes(fig, [xs[3] + 0.03, y0 + 0.25, w - 0.045, 0.33])
    ax.scatter(x, -1.2 * x + 0.08 * rng.normal(size=x.size), s=4, color=MUTED, lw=0)
    ax.plot([-1, 1], [1.2, -1.2], color=ORANGE, lw=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$u$", labelpad=1)
    ax.set_ylabel("$\\Delta F$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(xs[3] + w / 2, y0 + 0.13,
             "$\\Delta\\mathbf{F}=\\mathbf{F}(\\mathbf{u})-\\mathbf{F}(\\mathbf{0})$ by DFT;\n"
             "least squares in the\nspace-group basis",
             fontsize=5.6, ha="center", va="center", color=INK, linespacing=1.35)

    # (e) effective dispersion against the trial
    ax = add_axes(fig, [xs[4] + 0.03, y0 + 0.25, w - 0.045, 0.33])
    for b in branches(K1, K2, q).T:
        ax.plot(q, b, color=GRAY, lw=0.9, ls="--")
    for b in branches(1.12 * K1, 1.25 * K2, q).T:
        ax.plot(q, b, color=ORANGE, lw=1.1)
    ax.set_xticks([0, np.pi], ["$\\Gamma$", "X"])
    ax.set_yticks([])
    ax.set_ylabel("$\\omega$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(xs[4] + w / 2, y0 + 0.13,
             "$\\Phi_{\\rm eff}(T)=\\langle\\nabla\\nabla V\\rangle$\n"
             "(gray dashed: trial)",
             fontsize=5.6, ha="center", va="center", color=INK, linespacing=1.35)

    # forward arrows between panels
    for k in range(4):
        arrow(fig, (xs[k] + w + 0.001, y0 + h / 2), (xs[k + 1] - 0.001, y0 + h / 2))

    # self-consistent loop: (e) back to (a)
    arrow(fig, (xs[4] + w / 2, y0 - 0.01), (xs[0] + w / 2, y0 - 0.01), rad=-0.06,
          color=ORANGE)
    fig.text(0.5, 0.035,
             "SOTC-D-SC: the fitted model becomes the trial for the next $\\pm$ pair; "
             "all forces are kept and refitted. Fixed point: "
             "$\\Phi_{\\rm SC}=\\Phi_{\\rm eff}"
             "[\\mathbf{C}(\\Phi_{\\rm SC},T)]$ (SCHA / sTDEP)",
             fontsize=5.9, ha="center", va="center", color=ORANGE)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_workflow.pdf")
    fig.savefig(OUT / "fig_workflow.png", dpi=200)


if __name__ == "__main__":
    main()
