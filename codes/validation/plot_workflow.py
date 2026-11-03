#!/usr/bin/env python3
"""Schematic of the SOTC-D workflow (Fig. 1 of the JAP manuscript).

Panels (a)-(c) and (e) are computed for a two-dimensional square lattice
with nearest- and next-nearest-neighbor springs (4x4 cell, unit masses):

* (a) the dispersion of the harmonic trial along Gamma-X-M-Gamma with the
  wave vectors commensurate with the 4x4 cell;
* (b) the mode coordinates of a canonical draw (Gaussian) and of the SOTC-D
  configuration (every mode at exactly +-sigma), whose signs maximize
  log det X^T X in a four-parameter FC2 basis (longitudinal and transverse
  first- and second-neighbor constants) by greedy single flips;
* (c) the D-optimal configuration and its inverse;
* (e) the dispersion before and after stiffening the springs, which
  illustrates, but does not quantify, the thermal renormalization.

Panel (d) shows a one-dimensional anharmonic force
F(u) = -k u - c u^2 - d u^3: the slope of the secant through a +-u pair is
k + d u^2, independent of the cubic coefficient c.

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
N_SHOWN = 14  # modes shown in panel (b)
LIGHT = "#c9c7c1"
BONDS = [((1, 0), 0), ((0, 1), 0), ((1, 1), 1), ((1, -1), 1)]  # (vector, shell)


def lattice_basis():
    """Supercell FC2 basis: longitudinal and transverse constants of shells 1, 2.

    Returns the four (2n, 2n) basis matrices and the site positions; the
    trial FC2 with central springs is K1 * B[0] + K2 * B[2].
    """
    n = L * L
    pos = np.array([(i, j) for i in range(L) for j in range(L)], float)
    basis = np.zeros((4, n, 2, n, 2))
    for a in range(n):
        for d, shell in BONDS:
            b = int(((pos[a, 0] + d[0]) % L) * L + (pos[a, 1] + d[1]) % L)
            e = np.array(d, float) / np.linalg.norm(d)
            for kind, block in ((0, np.outer(e, e)), (1, np.eye(2) - np.outer(e, e))):
                p = 2 * shell + kind
                basis[p, a, :, b, :] -= block
                basis[p, b, :, a, :] -= block
                basis[p, a, :, a, :] += block
                basis[p, b, :, b, :] += block
    return basis.reshape(4, 2 * n, 2 * n), pos


def dispersion(k1, k2, qpath):
    """Branches of the infinite lattice with central springs at wave vectors qpath."""
    w = []
    for q in qpath:
        d = np.zeros((2, 2))
        for dv, shell in BONDS:
            e = np.array(dv, float) / np.linalg.norm(dv)
            k = k1 if shell == 0 else k2
            d += 2 * k * np.outer(e, e) * (1 - np.cos(q @ np.array(dv, float)))
        w.append(np.sqrt(np.clip(np.linalg.eigvalsh(d), 0, None)))
    return np.array(w)


def path(n=60):
    """Gamma-X-M-Gamma path, its cumulative length, and the commensurate points."""
    corners = np.array([[0, 0], [np.pi, 0], [np.pi, np.pi], [0, 0]], float)
    pts, dist, ticks = [], [], [0.0]
    s = 0.0
    for a, b in zip(corners[:-1], corners[1:]):
        seg = np.linspace(0, 1, n)[:, None] * (b - a) + a
        lens = np.linalg.norm(seg - a, axis=1) + s
        pts.append(seg)
        dist.append(lens)
        s += np.linalg.norm(b - a)
        ticks.append(s)
    # commensurate wave vectors 2 pi (n1, n2) / L on the path
    comm = []
    s = 0.0
    for a, b in zip(corners[:-1], corners[1:]):
        steps = int(round(np.abs(b - a).max() / (2 * np.pi / L)))
        for k in range(steps + 1):
            comm.append((a + (b - a) * k / steps, s + np.linalg.norm(b - a) * k / steps))
        s += np.linalg.norm(b - a)
    return np.concatenate(pts), np.concatenate(dist), ticks, comm


def log_det(basis, u):
    x = -np.einsum("pij,j->ip", basis, u)
    return np.linalg.slogdet(x.T @ x)[1]


def configurations(seed=3):
    basis, pos = lattice_basis()
    phi = K1 * basis[0] + K2 * basis[2]
    w2, vec = np.linalg.eigh(phi)
    keep = w2 > 1e-8  # drop the two translations
    w2, vec = w2[keep], vec[:, keep]
    sigma = np.sqrt(KT / w2)
    rng = np.random.default_rng(seed)
    patterns = vec * sigma  # columns: sigma_m e_m

    q_canon = rng.normal(size=len(sigma)) * sigma
    # greedy single-flip D-optimal search from a random start
    signs = np.where(rng.random(len(sigma)) < 0.5, -1.0, 1.0)
    best = log_det(basis, patterns @ signs)
    for _ in range(6):
        improved = False
        for m in range(len(signs)):
            signs[m] *= -1
            trial = log_det(basis, patterns @ signs)
            if trial > best + 1e-12:
                best, improved = trial, True
            else:
                signs[m] *= -1
        if not improved:
            break
    u = (patterns @ signs).reshape(-1, 2)
    return pos, u, q_canon, signs * sigma, sigma


def clean(ax):
    ax.set_facecolor("none")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in", length=2.5, pad=2)


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": INK, "pdf.fonttype": 42, "legend.fontsize": 7,
        "legend.frameon": False, "legend.handlelength": 1.5,
    })
    pos, u, q_canon, q_sotc, sigma = configurations()
    fig = plt.figure(figsize=(7.0, 1.95))
    xs = [0.035, 0.245, 0.435, 0.68, 0.87]
    wid = [0.14, 0.145, 0.165, 0.12, 0.125]
    y0, h = 0.2, 0.62
    axes = [fig.add_axes([x, y0, w, h]) for x, w in zip(xs, wid)]
    for ax, letter in zip(axes, "abcde"):
        ax.text(-0.02, 1.13, f"({letter})", transform=ax.transAxes, fontsize=9,
                va="bottom", ha="right")
    titles = ["harmonic trial", "mode coordinates", r"designed $\pm\mathbf{u}$ pair",
              "force fit", "effective FC2"]
    for ax, t in zip(axes, titles):
        ax.text(0.0, 1.13, t, transform=ax.transAxes, fontsize=8.5, va="bottom",
                ha="left")

    # (a) trial dispersion along Gamma-X-M-Gamma with the commensurate wave vectors
    qpts, dist, ticks, comm = path()
    ax = axes[0]
    for b in dispersion(K1, K2, qpts).T:
        ax.plot(dist, b, color=INK, lw=0.9)
    cq = np.array([c[0] for c in comm])
    cd = np.array([c[1] for c in comm])
    for b in dispersion(K1, K2, cq).T:
        ax.plot(cd, b, "o", color=BLUE, ms=3.2, mec=BLUE, zorder=3, clip_on=False)
    for t in ticks[1:-1]:
        ax.axvline(t, color=LIGHT, lw=0.5, zorder=0)
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_yticks([])
    ax.set_ylabel(r"$\omega$", labelpad=2)
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(0, None)
    clean(ax)

    # (b) mode coordinates: canonical draw against SOTC-D (every mode at +-sigma)
    ax = axes[1]
    order = np.argsort(sigma)[::-1][:N_SHOWN]
    m = np.arange(N_SHOWN)
    s = sigma[order]
    ax.step(m, s, where="mid", color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    ax.step(m, -s, where="mid", color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    ax.axhline(0, color=INK, lw=0.5)
    ax.vlines(m - 0.15, 0, q_canon[order], color=GRAY, lw=0.6)
    ax.plot(m - 0.15, q_canon[order], "o", mfc="white", mec=GRAY, ms=2.8, mew=0.7,
            label="canonical")
    ax.vlines(m + 0.15, 0, q_sotc[order], color=BLUE, lw=0.6)
    ax.plot(m + 0.15, q_sotc[order], "s", color=BLUE, ms=2.6, label="SOTC-D")
    lim = 1.1 * max(np.abs(q_canon[order]).max(), s.max())
    ax.set_ylim(-lim, 1.45 * lim)
    ax.set_xlim(-0.7, N_SHOWN - 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$Q_m$", labelpad=2)
    ax.set_xlabel(r"mode $m$", labelpad=2)
    clean(ax)
    ax.legend(loc="upper center", ncol=2, handletextpad=0.2, borderaxespad=0,
              columnspacing=0.8)

    # (c) the configuration and its inverse with springs (displacements exaggerated)
    ax = axes[2]
    ax.axis("off")
    scale = 0.36 / np.abs(u).max()
    gap = L + 0.8
    for k, (sgn, label) in enumerate([(1, r"$+\mathbf{u}$"), (-1, r"$-\mathbf{u}$")]):
        off = np.array([k * gap, 0.0])
        p0 = pos + off
        p1 = p0 + sgn * scale * u
        for a in range(len(pos)):
            for d, shell in BONDS[:2]:
                b_ = pos[a] + d
                if b_[0] >= L or b_[1] >= L:
                    continue
                b = int(b_[0] * L + b_[1])
                ax.plot([p1[a, 0], p1[b, 0]], [p1[a, 1], p1[b, 1]], color=LIGHT, lw=0.6,
                        zorder=1)
        ax.scatter(p0[:, 0], p0[:, 1], s=7, facecolor="white", edgecolor=MUTED,
                   linewidth=0.4, zorder=2)
        ax.quiver(p0[:, 0], p0[:, 1], p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1],
                  angles="xy", scale_units="xy", scale=1, width=0.005, headwidth=4,
                  headlength=4, headaxislength=3.5, color=INK, zorder=3)
        ax.scatter(p1[:, 0], p1[:, 1], s=13, color=BLUE, linewidth=0, zorder=4)
        ax.text(off[0] + (L - 1) / 2, -0.85, label, ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.5, gap + L - 0.5)
    ax.set_ylim(-1.5, L - 0.4)
    ax.set_aspect("equal", adjustable="datalim")

    # (d) anharmonic force of one coordinate, the +-u secant, and the harmonic tangent
    ax = axes[3]
    k, c, d3, a = 1.0, 0.55, 0.35, 0.9
    x = np.linspace(-1.25, 1.25, 200)
    force = -k * x - c * x ** 2 - d3 * x ** 3
    fp, fm = (-k * a - c * a ** 2 - d3 * a ** 3), (k * a - c * a ** 2 + d3 * a ** 3)
    slope = (fp - fm) / (2 * a)
    ax.plot(x, -k * x, color=MUTED, lw=0.7, ls=(0, (3, 2)), label="harmonic")
    ax.plot(x, force, color=INK, lw=1.0, label=r"$F(u)$")
    ax.plot(x, slope * x + (fp + fm) / 2, color=ORANGE, lw=1.0, label=r"$\pm u$ fit")
    ax.plot([a, -a], [fp, fm], "o", color=BLUE, ms=3.6, zorder=4)
    ax.axhline(0, color=LIGHT, lw=0.5, zorder=0)
    ax.axvline(0, color=LIGHT, lw=0.5, zorder=0)
    ax.text(a + 0.1, fp + 0.02, r"$+u$", ha="left", va="bottom", fontsize=8)
    ax.text(-a + 0.1, fm + 0.08, r"$-u$", ha="left", va="bottom", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$u$", labelpad=2)
    ax.set_ylabel(r"$F$", labelpad=2)
    ax.set_xlim(-1.3, 1.3)
    clean(ax)
    ax.legend(loc="lower left", borderaxespad=0, labelspacing=0.2, handlelength=1.6)

    # (e) effective dispersion against the trial
    ax = axes[4]
    trial = dispersion(K1, K2, qpts)
    eff = dispersion(1.12 * K1, 1.25 * K2, qpts)
    for bt, be in zip(trial.T, eff.T):
        ax.plot(dist, bt, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.plot(dist, be, color=ORANGE, lw=1.0)
    for t in ticks[1:-1]:
        ax.axvline(t, color=LIGHT, lw=0.5, zorder=0)
    ax.plot([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="trial")
    ax.plot([], [], color=ORANGE, lw=1.0, label=r"$\Phi_{\rm eff}(T)$")
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_yticks([])
    ax.set_ylabel(r"$\omega$", labelpad=2)
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(0, 1.5 * eff.max())
    clean(ax)
    ax.legend(loc="upper right", borderaxespad=0, labelspacing=0.2, handlelength=1.4)

    # forward arrows between the panels
    yc = y0 + h / 2
    for k in range(4):
        # leave room for the y labels of (b), (d), and (e); (c) has no axes
        x0 = xs[k] + wid[k] + 0.008
        x1 = xs[k + 1] - (0.004 if k == 1 else 0.03)
        fig.patches.append(FancyArrowPatch((x0, yc), (x1, yc), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                           color=INK))

    # self-consistent loop from (e) back to (a), below the panels
    y_top, y_bot = y0 - 0.115, 0.03
    xa, xe = xs[0] + wid[0] / 2, xs[4] + wid[4] / 2
    fig.add_artist(plt.Line2D([xe, xe, xa], [y_top, y_bot, y_bot], transform=fig.transFigure,
                              color=MUTED, lw=0.7))
    fig.patches.append(FancyArrowPatch((xa, y_bot), (xa, y_top + 0.01),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.7, color=MUTED))
    fig.text(0.5, y_bot + 0.012, "SOTC-D-SC: refit all forces, update the trial",
             ha="center", va="bottom", fontsize=7.5, color=INK, style="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_workflow.pdf")
    fig.savefig(OUT / "fig_workflow.png", dpi=300)


if __name__ == "__main__":
    main()
