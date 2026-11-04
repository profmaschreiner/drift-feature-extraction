"""
memoria.py
==========
Figura da memoria do drift score (Secao 4): decaimento de S_j apos um unico
alarme, para diferentes valores de phi_b, comparado com uma EWMA de mesmo
horizonte de memoria.

Algorithm 1 (sem novo alarme apos w = 0):
    phi(k) = min(1, phi_b * ln(1 + k)),  S(w) = prod_{k=1}^{w} (1 - phi(k))
Horizonte de memoria h (Equacao do horizonte):
    menor h com sum_{k=1}^{h} -ln(1 - phi(k)) >= 1, isto e, S(h) <= e^{-1}
EWMA de mesmo horizonte:
    lambda = 1/h,  S(w) = (1 - lambda)^w
As duas curvas cruzam perto de e^{-1} em w = h. Antes de h, o Algorithm 1
guarda mais memoria (esquecimento inicial pequeno); depois, guarda menos
(esquecimento acelera com w).

Saida: fig_2_memory_drift.{pdf,png} (eixo logaritmico, 1 a 1e5) e
      fig_2_memory_drift_linear.{pdf,png} (eixo linear, 0 a 200, como a figura antiga)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

PHI_B_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
W_MAX = 100_000
LIGHT = mcolors.to_rgb("#8FAEC4")
DARK = mcolors.to_rgb("#1B2E40")


def sequential_blues(n):
    return [mcolors.to_hex(tuple(LIGHT[j] + (i / (n - 1)) * (DARK[j] - LIGHT[j]) for j in range(3)))
            for i in range(n)]


COLORS = list(reversed(sequential_blues(len(PHI_B_VALUES))))
FONT_AXIS, FONT_TICK, FONT_LEGEND, LINEWIDTH = 10, 9, 8.5, 1.6
SAIDA = "fig_2_memory_drift"


def fmt(v):
    return rf"$10^{{{int(np.round(np.log10(v)))}}}$"


def decaimento_alg1(phi_b, w_max):
    k = np.arange(1, w_max + 1)
    phi = np.minimum(1.0, phi_b * np.log1p(k))
    log_s = np.cumsum(np.log1p(-np.minimum(phi, 1 - 1e-15)))
    return np.r_[1.0, np.exp(log_s)]


def horizonte(phi_b):
    k, soma = 0, 0.0
    while soma < 1:
        k += 1
        soma += -np.log(1 - min(1 - 1e-15, phi_b * np.log(1 + k)))
    return k


def figura(saida, escala_log, w_max):
    w = np.arange(0, w_max + 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axhline(np.exp(-1), color="#999999", lw=0.8, ls=":", zorder=1)
    ax.text(1.3 if escala_log else w_max * 0.6, np.exp(-1) + 0.02, r"$e^{-1}$", fontsize=FONT_TICK,
            color="#666666", ha="left")
    inicio = 1 if escala_log else 0          # w = 0 (o alarme) nao cabe na escala logaritmica
    for phi_b, cor in zip(PHI_B_VALUES, COLORS):
        h = horizonte(phi_b)
        lam = 1.0 / h
        ax.plot(w[inicio:], decaimento_alg1(phi_b, w_max)[inicio:], color=cor, lw=LINEWIDTH, label=fmt(phi_b))
        ax.plot(w[inicio:], (1 - lam) ** w[inicio:], color=cor, lw=LINEWIDTH * 0.8, ls=(0, (3, 2)))
        if h <= w_max:
            ax.plot(h, np.exp(-1), "o", color=cor, ms=3.5, zorder=4)
        print(f"[{saida}] phi_b={phi_b:g}  h={h}  lambda={lam:.3g}")
    if escala_log:
        ax.set_xscale("log")
        ax.set_xlim(1, w_max)
    else:
        ax.set_xlim(0, w_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("instances since the alarm", fontsize=FONT_AXIS)
    ax.set_ylabel(r"$S_j$ relative to the alarm", fontsize=FONT_AXIS)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    leg1 = ax.legend(title=r"$\phi_b$", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
                     loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True,
                     handlelength=1.3, handletextpad=0.4, columnspacing=0.8, borderpad=0.35)
    leg1.get_frame().set_edgecolor("#cccccc")
    ax.add_artist(leg1)
    estilos = [Line2D([], [], color="#52514e", lw=LINEWIDTH, label="Algorithm 1"),
               Line2D([], [], color="#52514e", lw=LINEWIDTH * 0.8, ls=(0, (3, 2)), label="EWMA, same $h$")]
    if escala_log:
        leg2 = ax.legend(handles=estilos, fontsize=FONT_LEGEND - 1, frameon=False, handlelength=2.0,
                         loc="lower left")
    else:
        leg2 = ax.legend(handles=estilos, fontsize=FONT_LEGEND - 1, frameon=False, handlelength=2.0,
                         loc="lower right", bbox_to_anchor=(1.0, 0.1))
    plt.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{saida}.{ext}", dpi=300, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"Figura salva em {saida}.pdf e .png")


def main():
    figura(SAIDA, escala_log=True, w_max=W_MAX)                    # eixo logaritmico, todos os horizontes
    figura(SAIDA + "_linear", escala_log=False, w_max=200)          # como a figura antiga, 0 a 200


if __name__ == "__main__":
    main()