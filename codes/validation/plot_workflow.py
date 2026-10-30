#!/usr/bin/env python3
"""Schematic of the SOTC-D workflow (Fig. 1 of the JAP manuscript).

Panels (b) and (c) are computed for a two-dimensional square lattice with
nearest- and next-nearest-neighbor springs (4x4 cell, unit masses), with
the construction of the method: commensurate modes of a harmonic trial,
a canonical draw with Gaussian mode coordinates, and an SOTC-D
configuration with every mode at exactly its rms amplitude, together with
its inverse. Panel (d) shows a one-dimensional anharmonic force
F(u) = -k u - c u^2 - d u^3: the slope of the secant through a +-u pair is
k + d u^2, independent of the cubic coefficient c. The dispersions of
panels (a) and (e) are those of the toy lattice before and after
stiffening its springs; they illustrate, not quantify, the thermal
renormalization.

Writes manuscript/jap/figures/fig_workflow.pdf.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_robustness import BLUE, GRAY, INK, MUTED, ORANGE  # noqa: E402

OUT = ROOT / "manuscript" / "jap" / "figures"
L = 4  # cell edge (atoms)
K1, K2 = 1.0, 0.35  # nearest- and next-nearest-neighbor springs of the trial
KT = 0.02  # thermal energy of the illustration (classical amplitudes)
N_SHOWN = 12  # modes shown in panel (b)
LIGHT = "#c9c7c1"


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
    """Longitudinal and transverse branches along Gamma-X of the infinite lattice."""
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
    q_canon = rng.normal(size=len(sigma)) * sigma
    signs = np.where(rng.random(len(sigma)) < 0.5, -1.0, 1.0)
    u = (vec @ (signs * sigma)).reshape(-1, 2)
    return pos, u, q_canon, signs * sigma, sigma


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)


def main():
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "mathtext.fontset": "dejavusans"})
    pos, u, q_canon, q_sotc, sigma = configurations()
    fig = plt.figure(figsize=(7.0, 2.9))

    # panel geometry (figure fractions)
    xs = [0.035, 0.235, 0.41, 0.65, 0.85]
    wid = [0.13, 0.14, 0.175, 0.13, 0.13]
    y_ax, h_ax = 0.42, 0.43
    titles = ["(a) harmonic trial", "(b) mode amplitudes", "(c) designed $\\pm$ pair",
              "(d) force fit", "(e) effective FC2"]
    notes = ["modes $\\omega_m$, $\\mathbf{e}_m$,\nrms amplitudes $\\sigma_m$",
             "every mode at $\\pm\\sigma_m$;\nsigns maximize $\\det\\mathbf{X}^{\\rm T}\\mathbf{X}$",
             "$\\mathbf{u}$ and $-\\mathbf{u}$:\ntwo force calls",
             "pair slope is free\nof the cubic force",
             "$\\Phi_{\\rm eff}(T)=\\langle\\nabla\\nabla V\\rangle$"]
    axes = []
    for x, w, t, n in zip(xs, wid, titles, notes):
        ax = fig.add_axes([x, y_ax, w, h_ax])
        ax.set_title(t, fontsize=7.8, loc="center", pad=7, color=INK)
        fig.text(x + w / 2, y_ax - 0.095, n, ha="center", va="top", fontsize=7,
                 color=MUTED, linespacing=1.3)
        axes.append(ax)

    # (a) trial dispersion with the commensurate wave vectors
    ax = axes[0]
    q = np.linspace(0, np.pi, 120)
    qc = np.pi * np.arange(0, L // 2 + 1) / (L // 2)
    for b, bc in zip(branches(K1, K2, q).T, branches(K1, K2, qc).T):
        ax.plot(q, b, color=GRAY, lw=1.4)
        ax.plot(qc, bc, "o", color=BLUE, ms=3.6, zorder=3, clip_on=False)
    ax.set_xticks([0, np.pi], ["$\\Gamma$", "X"])
    ax.set_yticks([])
    ax.set_ylabel("$\\omega$", labelpad=2)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, None)
    clean(ax)

    # (b) mode coordinates of a canonical draw and of SOTC-D, with the +-sigma envelope
    ax = axes[1]
    order = np.argsort(sigma)[::-1][:N_SHOWN]
    m = np.arange(N_SHOWN)
    s = sigma[order]
    ax.fill_between(m, -s, s, step="mid", color=BLUE, alpha=0.08, lw=0)
    ax.step(m, s, where="mid", color=BLUE, lw=0.6, alpha=0.6)
    ax.step(m, -s, where="mid", color=BLUE, lw=0.6, alpha=0.6)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.vlines(m + 0.17, 0, q_sotc[order], color=BLUE, lw=2.4, label="SOTC-D")
    ax.plot(m - 0.17, q_canon[order], "o", color=GRAY, ms=3.0, label="canonical")
    ax.vlines(m - 0.17, 0, q_canon[order], color=GRAY, lw=0.8)
    lim = 1.1 * max(np.abs(q_canon[order]).max(), s.max())
    ax.set_ylim(-lim, 1.55 * lim)
    ax.set_xlim(-0.6, N_SHOWN - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("mode $m$", labelpad=2)
    ax.set_ylabel("$Q_m$", labelpad=2)
    clean(ax)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.0,
              borderaxespad=0.1, bbox_to_anchor=(1.03, 1.02))

    # (c) the configuration and its inverse (displacements exaggerated)
    ax = axes[2]
    ax.axis("off")
    scale = 0.42 / np.abs(u).max()
    gap = L + 0.6
    for k, (sgn, label) in enumerate([(1, "$+\\mathbf{u}$"), (-1, "$-\\mathbf{u}$")]):
        off = np.array([k * gap, 0.0])
        p0 = pos + off
        d = sgn * scale * u
        ax.scatter(p0[:, 0], p0[:, 1], s=12, facecolor="white", edgecolor=LIGHT,
                   linewidth=0.8, zorder=1)
        ax.quiver(p0[:, 0], p0[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy",
                  scale=1, width=0.008, headwidth=0, headlength=0, headaxislength=0,
                  color=BLUE, zorder=2)
        ax.scatter(p0[:, 0] + d[:, 0], p0[:, 1] + d[:, 1], s=7, color=BLUE, zorder=3,
                   linewidth=0)
        ax.text(off[0] + (L - 1) / 2, L - 0.15, label, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-0.6, gap + L - 0.4)
    ax.set_ylim(-0.6, L + 0.4)
    ax.set_aspect("equal", adjustable="datalim")

    # (d) anharmonic force of one coordinate and the +-u secant
    ax = axes[3]
    k, c, d3, a = 1.0, 0.55, 0.35, 0.9
    x = np.linspace(-1.25, 1.25, 200)
    ax.plot(x, -k * x - c * x ** 2 - d3 * x ** 3, color=GRAY, lw=1.4, label="DFT force")
    fp, fm = (-k * a - c * a ** 2 - d3 * a ** 3), (k * a - c * a ** 2 + d3 * a ** 3)
    slope = (fp - fm) / (2 * a)
    ax.plot(x, slope * x, color=ORANGE, lw=1.2, label="fit")
    ax.plot([a, -a], [fp, fm], "o", color=BLUE, ms=4.2, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.4)
    ax.axvline(0, color=MUTED, lw=0.4)
    ax.text(a - 0.12, fp, "$+u$", ha="right", va="center", fontsize=7, color=BLUE)
    ax.text(-a + 0.12, fm, "$-u$", ha="left", va="center", fontsize=7, color=BLUE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$u$", labelpad=2)
    ax.set_ylabel("$F$", labelpad=2)
    ax.set_xlim(-1.3, 1.3)
    clean(ax)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.2,
              borderaxespad=0.1, bbox_to_anchor=(1.05, 1.02))

    # (e) effective dispersion against the trial
    ax = axes[4]
    for b in branches(K1, K2, q).T:
        ax.plot(q, b, color=GRAY, lw=1.0, ls=(0, (3, 2)))
    for b in branches(1.12 * K1, 1.25 * K2, q).T:
        ax.plot(q, b, color=ORANGE, lw=1.5)
    ax.plot([], [], color=GRAY, lw=1.0, ls=(0, (3, 2)), label="trial")
    ax.plot([], [], color=ORANGE, lw=1.5, label="effective")
    ax.set_xticks([0, np.pi], ["$\\Gamma$", "X"])
    ax.set_yticks([])
    ax.set_ylabel("$\\omega$", labelpad=2)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, None)
    clean(ax)
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=1.4,
              borderaxespad=0.1)

    # forward arrows between panels
    yc = y_ax + h_ax / 2
    for k in range(4):
        x0 = xs[k] + wid[k] + 0.01
        x1 = xs[k + 1] - (0.008 if k == 1 else 0.032)
        fig.patches.append(FancyArrowPatch((x0, yc), (x1, yc), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                           color=MUTED))

    # self-consistent loop from (e) back to (a), drawn below the notes
    y_top, y_bot = 0.205, 0.115
    xa, xe = xs[0] + wid[0] / 2, xs[4] + wid[4] / 2
    fig.add_artist(plt.Line2D([xe, xe, xa], [y_top, y_bot, y_bot], transform=fig.transFigure,
                              color=ORANGE, lw=1.3, solid_joinstyle="round"))
    fig.patches.append(FancyArrowPatch((xa, y_bot), (xa, y_top + 0.005),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=11, lw=1.3, color=ORANGE))
    fig.text(0.5, y_bot - 0.05,
             "SOTC-D-SC: the fitted model is the trial of the next pair and all forces are "
             "refitted; fixed point = SCHA / sTDEP effective FC2",
             ha="center", va="center", fontsize=7.2, color=ORANGE)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_workflow.pdf")
    fig.savefig(OUT / "fig_workflow.png", dpi=300)


if __name__ == "__main__":
    main()
