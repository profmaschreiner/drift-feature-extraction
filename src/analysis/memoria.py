"""
fig2_memory_decay.py
=====================
Gera a Figura 2 do artigo: decaimento de S_j (score de memoria acumulada)
para diferentes valores de phi_b, assumindo um unico alarme na instancia
inicial (w=0) e nenhum alarme subsequente.

Formula (Algoritmo 1, Secao IV):
    phi_j = phi_b * ln(1 + w_j)
    S_j   = (1 - phi_j) * S_j   [pois nao ha novo alarme apos w=0]

Estilo: paleta sequencial de azuis ancorada em #355C7D (mesma identidade
visual da Figura 12), coluna simples IEEE (~3.5 in), saida em ingles.

Saida: fig2_memory_decay.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# CONFIGURACAO
# ============================================================

PHI_B_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_STEPS = 200  # w = 0, 1, ..., 200  ->  201 instancias (data sequence 1..201)

# Paleta sequencial de azuis: phi_b menor (decaimento lento) = mais escuro;
# phi_b maior (decaimento rapido) = mais claro. Ancorada em #355C7D.
LIGHT = mcolors.to_rgb("#8FAEC4")
DARK = mcolors.to_rgb("#1B2E40")


def sequential_blues(n):
    colors = []
    for i in range(n):
        t = i / (n - 1)
        c = tuple(LIGHT[j] + t * (DARK[j] - LIGHT[j]) for j in range(3))
        colors.append(mcolors.to_hex(c))
    return colors


# do mais escuro (phi_b menor) para o mais claro (phi_b maior)
COLORS = list(reversed(sequential_blues(len(PHI_B_VALUES))))


def format_phi_b(value: float) -> str:
    """Formata phi_b como notacao cientifica IEEE: 10^{-5} em vez de 1e-05."""
    exponent = int(np.round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


FONT_AXIS = 10
FONT_TICK = 9
FONT_LEGEND = 8.5
LINEWIDTH = 1.8

FIGURE_OUTPUT = "fig2_memory_decay.png"


# ============================================================
# SIMULACAO DO DECAIMENTO DE S_j
# ============================================================

def simulate_decay(phi_b: float, n_steps: int) -> np.ndarray:
    """
    Simula S_j para w = 0..n_steps, assumindo alarme apenas em w=0.
    Em w=0: phi=0 (ln(1)=0), S=1 (alarme dispara, beta*1 domina).
    Para w>=1: nao ha alarme, logo S_j = (1 - phi_j) * S_j (decaimento puro).
    """
    S = np.empty(n_steps + 1)
    S[0] = 1.0
    for w in range(1, n_steps + 1):
        phi = phi_b * np.log(1 + w)
        S[w] = (1 - phi) * S[w - 1]
    return S


# ============================================================
# FIGURA
# ============================================================

def main():
    w = np.arange(0, N_STEPS + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for phi_b, color in zip(PHI_B_VALUES, COLORS):
        S = simulate_decay(phi_b, N_STEPS)
        ax.plot(w + 1, S, color=color, linewidth=LINEWIDTH,
                label=format_phi_b(phi_b))

    ax.set_xlabel("data sequence", fontsize=FONT_AXIS)
    ax.set_ylabel(r"$1 - \phi_j$", fontsize=FONT_AXIS)
    ax.set_xlim(1, N_STEPS + 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(
        title=r"$\phi_b$",
        fontsize=FONT_LEGEND,
        title_fontsize=FONT_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=True,
        framealpha=0.95,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.35,
    )
    legend.get_frame().set_edgecolor("#cccccc")

    plt.tight_layout(pad=0.4)
    fig.savefig(FIGURE_OUTPUT, dpi=300, bbox_inches="tight",
                bbox_extra_artists=(legend,))
    plt.close(fig)
    print(f"Figura salva em: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()