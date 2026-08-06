import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_FILE  = "exp_otimizacao/datasets_sintetico_rf/analise_variancia_bons_correto/testes_mannwhitney_agregados_por_detector.csv"
OUTPUT_FILE = "exp_otimizacao/datasets_sintetico_rf/analise_variancia_bons_correto/figura_cliffs_delta.png"  # ou .png

# Ordem dos detectores — colunas
ORDEM_DETECTORES = [
    "HDDMA",
    "HDDMW",   
    "PH",
    "GMA",
    "EWMAC",
    "KSWIN",
    "CUSUM",
    "ADWIN",
    "SEED",
]

# Apenas as duas comparações relevantes — linhas
COMPARACOES = {
    "FP vs TN": r"$\mathrm{FP} > \mathrm{TN}$",
    "FP vs TP": r"$\mathrm{FP} \neq \mathrm{TP}$",
}

P_THRESHOLD = 0.001

# Cor unica por magnitude (consistente com o script-ancora
# test_estat_ot_default_agr.py, usado nas Fig. 7/9/10/11/12). Mesma cor
# preenche o fundo da celula e o patch da legenda — sem esquema separado
# de "fundo claro + legenda saturada".
COR_MAG = {
    "Large":      "#e06666",
    "Medium":     "#f6b26b",
    "Small":      "#6fa8dc",
    "Negligible": "#b7b7b7",
}


# ============================================================
# FUNÇÕES
# ============================================================

def magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "Negligible"
    elif d < 0.330:
        return "Small"
    elif d < 0.474:
        return "Medium"
    else:
        return "Large"


def formatar_celula(delta: float, p_value: float) -> str:
    sig = "*" if p_value < P_THRESHOLD else ""
    return f"{delta:+.2f}{sig}"


def gerar_dados(df: pd.DataFrame):
    pivot_delta = df.pivot(index="detector", columns="comparacao", values="cliffs_delta")
    pivot_p     = df.pivot(index="detector", columns="comparacao", values="p_value")

    detectores = [d for d in ORDEM_DETECTORES if d in pivot_delta.index]
    pivot_delta = pivot_delta.reindex(detectores)
    pivot_p     = pivot_p.reindex(detectores)

    return pivot_delta, pivot_p, detectores


# ============================================================
# FIGURA TRANSPOSTA
# Linhas  = comparações (2)
# Colunas = detectores  (8)
# ============================================================

def gerar_figura(pivot_delta, pivot_p, detectores):

    # Tamanhos de fonte consistentes com o script-ancora
    # (test_estat_ot_default_agr.py): font_val=14, font_hdr=13, font_row=14.
    FONT_VAL = 14
    FONT_HDR = 13
    FONT_ROW = 14

    comp_keys   = list(COMPARACOES.keys())
    comp_labels = list(COMPARACOES.values())

    n_linhas  = len(comp_keys)      # 2
    n_colunas = len(detectores)     # 8

    # Largura generosa para caber 8 colunas no IEEE
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(-1.05, n_colunas)
    ax.set_ylim(-0.5, n_linhas + 1.4)
    ax.axis("off")

    # --- Legenda no topo ---
    # Ordem Large -> Medium -> Small -> Negligible, consistente com o
    # script-ancora (test_estat_ot_default_agr.py).
    patches = [
        mpatches.Patch(color=COR_MAG["Large"],      label="Large"),
        mpatches.Patch(color=COR_MAG["Medium"],     label="Medium"),
        mpatches.Patch(color=COR_MAG["Small"],      label="Small"),
        mpatches.Patch(color=COR_MAG["Negligible"], label="Negligible"),
    ]
    legend = ax.legend(
        handles=patches,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.35),
        bbox_transform=ax.transAxes,
        ncol=4,
        frameon=True,
        fontsize=FONT_VAL,
        title=r"Cliff's $\delta$ magnitude  ($*\ p < 0{,}001$)",
        title_fontsize=FONT_VAL,
        edgecolor="gray",
    )
    ax.add_artist(legend)
    #plt.subplots_adjust(bottom=0.25)

    # --- Cabeçalho: nomes dos detectores (colunas) ---
    for j, det in enumerate(detectores):
        ax.text(
            j + 0.5, n_linhas + 0.2,
            det,
            ha="center", va="center",
            fontsize=FONT_HDR, fontweight="bold",
            color="black"
        )
    ax.text(
        -0.08, n_linhas + 0.2,
        "Variance",
        ha="right", va="center",
        fontsize=FONT_HDR, fontweight="bold",
        color="black"
    )

    # --- Bordas externas ---
    ax.axhline(n_linhas, color="black", linewidth=1.5, zorder=3)
    ax.axhline(0,        color="black", linewidth=1.5, zorder=3)
    ax.axvline(0,        color="black", linewidth=1.5, zorder=3)
    ax.axvline(n_colunas, color="black", linewidth=1.5, zorder=3)

    # --- Grades internas ---
    for i in range(1, n_linhas):
        ax.axhline(i, color="gray", linewidth=0.5, zorder=0)
    for j in range(1, n_colunas):
        ax.axvline(j, color="gray", linewidth=0.4, zorder=0)

    # --- Rótulos das linhas (comparações) no eixo esquerdo ---
    for i, label in enumerate(reversed(comp_labels)):
        ax.text(
            -0.08, i + 0.5,
            label,
            ha="right", va="center",
            fontsize=FONT_ROW, color="black"
        )

    # --- Células: linhas = comparações, colunas = detectores ---
    for i, comp_key in enumerate(reversed(comp_keys)):
        for j, det in enumerate(detectores):
            delta   = pivot_delta.loc[det, comp_key]
            p_value = pivot_p.loc[det, comp_key]

            texto     = formatar_celula(delta, p_value)
            mag       = magnitude(delta)
            cor_fundo = COR_MAG[mag]

            rect = plt.Rectangle(
                (j, i), 1, 1,
                facecolor=cor_fundo,
                edgecolor="white",
                zorder=1
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5, i + 0.5,
                texto,
                ha="center", va="center",
                fontsize=FONT_VAL,
                fontweight="bold", color="white", zorder=2
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=200,
                bbox_extra_artists=(legend,))
    print(f"Figura salva em: {OUTPUT_FILE}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Lendo: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    pivot_delta, pivot_p, detectores = gerar_dados(df)
    gerar_figura(pivot_delta, pivot_p, detectores)


if __name__ == "__main__":
    main()