"""
fig4_causal_window_protocol.py
================================
Gera a Figura 4 do artigo: protocolo de calculo da variancia local por
janela causal de 50 instancias, usado na Secao V-D para comparar FP vs.
TN (e, por extensao, FP vs. TP) em termos de variancia local.

Ilustra dois casos lado a lado:
  - Alarme em t_a: janela causal [t_a - 50, t_a] usada para variancia
  - Ausencia de alarme em t_n (TN, ou FN se houve drift nao sinalizado):
    janela causal [t_n - 50, t_n]

Paleta categorica VP/FP/TN/FN: tons derivados exclusivamente da familia
de azuis do artigo (ancorada em #355C7D), diferenciando por luminosidade/
saturacao em vez de matiz, para manter identidade visual unica no artigo
e permanecer distinguivel em escala de cinza.

Estilo: vetorial (matplotlib), fonte LaTeX, coluna dupla IEEE (~7 in),
saida em ingles.

Saida: fig4_causal_window_protocol.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ============================================================
# PALETA CATEGORICA VP/FP/TN/FN
# Tons derivados da familia de azuis do artigo (#355C7D), distintos por
# luminosidade/saturacao, nao por matiz.
# ============================================================

COLOR_VP = "#16263A"  # mais escuro: alarme correto (evento de maior destaque)
COLOR_FP = "#3E6E94"  # azul medio-vivo: alarme incorreto (achado central da Sec. V-D)
COLOR_FN = "#7FA0B8"  # azul medio-claro: drift nao sinalizado
COLOR_TN = "#C5D3DD"  # mais claro: ausencia de alarme e ausencia de drift

# Nesta figura, apenas dois pontos sao plotados: um alarme (generico,
# usado para calculo de variancia, cor neutra escura) e um TN/FN (claro).
COLOR_ALARM = COLOR_FP   # ponto de alarme usado no exemplo (azul medio-vivo)
COLOR_TN_FN = COLOR_TN   # ponto sem alarme (TN ou FN), mais claro

FONT_LABEL = 10
FONT_ANNOT = 9
FONT_SIGNAL_LABEL = 10

FIGURE_OUTPUT = "fig4_causal_window_protocol.png"


# ============================================================
# SINAL SINTETICO (apenas ilustrativo, ruido + leve tendencia)
# ============================================================

def make_signal(n_points: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=n_points)


# ============================================================
# FIGURA
# ============================================================

def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Eixo de tempo total (unidades arbitrarias)
    t_total = 260
    t_a = 90       # instante com alarme (ponto azul)
    t_n = 200      # instante sem alarme (ponto claro), regiao TN/FN
    window = 50

    t = np.arange(0, t_total)
    signal = make_signal(t_total)
    signal_y0 = 0.0   # baseline vertical do sinal monitorado
    signal_scale = 0.35

    # ----- Camadas verticais (de baixo para cima) -----
    y_time_axis = signal_y0 - 1.0
    y_tick_label = y_time_axis - 0.30
    y_window_box_top = signal_y0 - 0.55
    y_window_box_bot = signal_y0 - 0.95
    y_short_label = signal_y0 + 0.85
    y_callout_box = signal_y0 + 2.0

    ax.plot(t, signal_y0 + signal * signal_scale,
            color="#666666", linewidth=0.5, zorder=1)

    # ----- Janela causal: alarme em t_a -----
    ax.add_patch(plt.Rectangle(
        (t_a - window, y_window_box_bot), window, y_window_box_top - y_window_box_bot,
        facecolor="none", edgecolor=COLOR_ALARM, linewidth=1.2,
        linestyle="--", zorder=2))
    ax.annotate("", xy=(t_a, signal_y0 + signal[t_a] * signal_scale + 0.05),
                xytext=(t_a, y_window_box_top),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_ALARM, lw=1.2),
                zorder=3)

    # ----- Janela causal: sem alarme em t_n -----
    ax.add_patch(plt.Rectangle(
        (t_n - window, y_window_box_bot), window, y_window_box_top - y_window_box_bot,
        facecolor="none", edgecolor="#5A7A90", linewidth=1.2,
        linestyle="--", zorder=2))
    ax.annotate("", xy=(t_n, signal_y0 + signal[t_n] * signal_scale + 0.05),
                xytext=(t_n, y_window_box_top),
                arrowprops=dict(arrowstyle="-|>", color="#5A7A90", lw=1.2),
                zorder=3)

    # ----- Pontos marcadores -----
    ax.scatter([t_a], [signal_y0 + signal[t_a] * signal_scale],
               s=70, color=COLOR_ALARM, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.scatter([t_n], [signal_y0 + signal[t_n] * signal_scale],
               s=70, color=COLOR_TN_FN, zorder=5,
               edgecolor="#666666", linewidth=0.8)

    # ----- Eixo do tempo (seta horizontal) -----
    ax.annotate("", xy=(t_total - 1, y_time_axis),
                xytext=(-5, y_time_axis),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
                zorder=2)
    ax.text(t_total + 2, y_time_axis, "time",
            ha="left", va="center", fontsize=FONT_LABEL)

    # Marcas verticais e rotulos no eixo do tempo
    for x_mark, label, color in [
        (t_a - window, r"$t_a-50$", COLOR_ALARM),
        (t_a, r"$t_a$", COLOR_ALARM),
        (t_n - window, r"$t_n-50$", "#5A7A90"),
        (t_n, r"$t_n$", "#5A7A90"),
    ]:
        ax.plot([x_mark, x_mark], [y_time_axis, y_time_axis + 0.12],
                color=color, linewidth=1.0, zorder=2)
        ax.text(x_mark, y_tick_label, label, ha="center", va="top",
                fontsize=FONT_ANNOT, color=color)

    # ----- Rotulo do sinal monitorado -----
    ax.text(-30, signal_y0, "Monitored\nfeature $x_t$",
            ha="right", va="center", fontsize=FONT_SIGNAL_LABEL)

    # ----- Rotulos curtos acima dos pontos -----
    ax.text(t_a, y_short_label, r"Alarm at $t_a$", ha="center", va="bottom",
            fontsize=FONT_ANNOT, color=COLOR_ALARM, fontweight="bold")
    ax.text(t_n, y_short_label, "No alarm at $t_n$\n(TN or FN)", ha="center", va="bottom",
            fontsize=FONT_ANNOT, color="#5A7A90", fontweight="bold")

    # ----- Caixas de anotacao (janela causal) -----
    box_props_alarm = dict(boxstyle="round,pad=0.5", facecolor="white",
                            edgecolor=COLOR_ALARM, linewidth=1.1)
    ax.text(t_a - window / 2 - 6, y_callout_box,
            "Causal window (50 past samples)\nused to compute local variance\n"
            r"$[t_a-50,\ t_a]$",
            ha="center", va="center", fontsize=FONT_ANNOT, bbox=box_props_alarm)

    box_props_tn = dict(boxstyle="round,pad=0.5", facecolor="white",
                         edgecolor="#5A7A90", linewidth=1.1)
    ax.text(t_n - window / 2 + 6, y_callout_box,
            "Causal window (50 past samples)\nused to compute local variance\n"
            r"$[t_n-50,\ t_n]$",
            ha="center", va="center", fontsize=FONT_ANNOT, bbox=box_props_tn)

    # ----- Legenda inferior (estilo caixa, consistente com Fig. 2/3) -----
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_ALARM, markeredgecolor="white",
                   markersize=8, label=r"Alarm used to compute variance (example at $t_a$)"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_TN_FN, markeredgecolor="#666666",
                   markersize=8, label="TN: no alarm and no drift (or FN: drift without alarm)"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper center",
                        bbox_to_anchor=(0.5, 0.0), ncol=1,
                        fontsize=FONT_ANNOT, frameon=True, framealpha=0.95,
                        handletextpad=0.6, borderpad=0.5)
    legend.get_frame().set_edgecolor("#cccccc")

    ax.set_xlim(-70, t_total + 25)
    ax.set_ylim(y_time_axis - 0.65, y_callout_box + 0.65)
    ax.axis("off")

    plt.tight_layout(pad=0.4)
    fig.savefig(FIGURE_OUTPUT, dpi=300, bbox_inches="tight",
                bbox_extra_artists=(legend,))
    plt.close(fig)
    print(f"Figura salva em: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()