"""
fig3_beta_weight.py
=====================
Gera a Figura 3 do artigo: peso beta_j atribuido a novos alarmes em
funcao de w_j (numero de instancias consecutivas sem alarme).

Formula (Algoritmo 1, Secao IV):
    beta_j = 1 + 1 / ln(1 + w_j)

Nota: beta_j so e definido para w_j >= 1 (ln(1+0)=0 causaria divisao por
zero); no Algoritmo 1, beta_j e calculado apenas no ramo "else" (sem
alarme), apos w_j ja ter sido incrementado para >= 1.

Estilo: identico a Figura 2 (paleta ancorada em #355C7D, coluna simples
IEEE ~3.5 in, saida em ingles), pois ambas ilustram o mesmo mecanismo.

Saida: fig3_beta_weight.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACAO
# ============================================================

N_STEPS = 200  # w_j = 1, ..., 200

# Cor unica ancorada na identidade visual do artigo (mesmo tom medio
# usado como referencia nas demais figuras azuis)
LINE_COLOR = "#355C7D"

FONT_AXIS = 10
FONT_TICK = 9
LINEWIDTH = 1.8

FIGURE_OUTPUT = "fig3_beta_weight.png"


# ============================================================
# CALCULO DE BETA_j
# ============================================================

def compute_beta(w: np.ndarray) -> np.ndarray:
    return 1.0 + 1.0 / np.log(1.0 + w)


# ============================================================
# FIGURA
# ============================================================

def main():
    w = np.arange(1, N_STEPS + 1)
    beta = compute_beta(w)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(w, beta, color=LINE_COLOR, linewidth=LINEWIDTH)

    ax.set_xlabel("data sequence", fontsize=FONT_AXIS)
    ax.set_ylabel(r"$\beta_j$", fontsize=FONT_AXIS)
    ax.set_xlim(1, N_STEPS)
    ax.set_ylim(0, beta.max() * 1.08)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    
    plt.tight_layout(pad=0.4)
    fig.savefig(FIGURE_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()