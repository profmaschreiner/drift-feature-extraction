"""
tabela_heatmap.py
================================
Gera, como FIGURA matplotlib (mesma estrutura visual usada em
teste_estatistico_cenarios_agr.py: Rectangle + texto centralizado,
bordas brancas, rótulos em negrito fora da grade, legenda inferior),
o heatmap que SUBSTITUI os dois gráficos de radar
(radar_duplo_best / radar_duplo_worst) na Seção VI-B do artigo.

Versão em INGLÊS: os valores numéricos nas células usam ponto decimal
(99.84), não vírgula (99,84), pois esta figura é destinada à versão em
inglês do artigo para submissão na IEEE T-SMC. Os CSVs de entrada
continuam no formato brasileiro (vírgula decimal); a conversão ocorre
apenas na formatação de exibição.

Diferença em relação ao padrão do script de teste estatístico
----------------------------------------------------------------
Lá, a cor codifica uma MAGNITUDE CATEGÓRICA (Cliff's delta: grande/
médio/pequeno/irrelevante), com paleta fixa de 4 cores.
Aqui, a cor codifica o F1-score macro, que é uma variável CONTÍNUA,
por isso usa-se um gradiente contínuo (branco -> teal) em vez de
categorias. A lógica de desenho (Rectangle/text/bordas/legenda) é a
mesma; só a função de mapeamento valor->cor muda.

Conteúdo da tabela
----------------------------------------------------------------
Uma linha por dataset (15 linhas, Tabela III do artigo) e seis
colunas: C0, C1 melhor detector, C1 pior detector, C2, C3 melhor
detector, C3 pior detector — todas em configuração DEFAULT dos
detectores (consistente com as Figuras 10/11: a otimização de
hiperparâmetros não traz ganho estatisticamente robusto nos dados
reais, então a tabela não precisa duplicar default e otimizado).

A escala de cor é ABSOLUTA e comum às seis colunas de F1 (min-max
global), para que a cor seja comparável tanto entre datasets (linhas)
quanto entre cenários (colunas) — uma escala por coluna inverteria
essa leitura.

O nome do detector aparece como sigla de duas letras dentro da
célula (ex: "80.73\n(KS)"), com legenda das siglas abaixo da figura.

Entradas esperadas (mesmos CSVs do radar_base_delta.py):
    resumo_tabelas/tabela_media_f1.csv   — index = dataset, columns = situation
    resumo_tabelas/tabela_desvio_f1.csv  — mesma estrutura, desvio padrão

Saídas:
    tabela_heatmap_en.csv  — dados em formato plano, para inspeção/reuso
    tabela_heatmap_en.png  — figura final em inglês (mesmo padrão visual do artigo)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÕES (mesmos caminhos do radar_base_delta.py)
# ============================================================

RESULT_ROOT = Path("exp_otimizacao/result_reais_completo")
SUMMARY_DIR = RESULT_ROOT / "resumo_tabelas"

MEAN_CSV = SUMMARY_DIR / "tabela_media_f1.csv"
STD_CSV  = SUMMARY_DIR / "tabela_desvio_f1.csv"

OUTPUT_DIR = SUMMARY_DIR / "tabela_heatmap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURA_SAIDA = OUTPUT_DIR / "tabela_heatmap_en.png"

# ── Abbreviated dataset labels (mesmo dict do radar_base_delta.py) ──────────
ABBREV = {
    "gait"             : "Gait",
    #"idosos"           : "HAR70+",
    "mhealth"          : "MH",
    "occ"              : "RO",
    "pamap2_ankle"     : "P2-Ankle",
    "pamap2_chest"     : "P2-Chest",
    "pamap2_hand"      : "P2-Hand",
    "rs"               : "DR",
    "smartphone"       : "SmPh",
    "usc_had"          : "USC",
    "ward_left_ankle"  : "W-LAnkle",
    "ward_left_wrist"  : "W-LWrist",
    "ward_right_ankle" : "W-RAnkle",
    "ward_right_wrist" : "W-RWrist",
    "ward_waist"       : "W-Waist",
    "sw_har"           : "Sw",
    "sp_har"           : "Sp",
}

# Ordem de exibição das linhas (datasets não listados aqui entram no final,
# na ordem em que aparecem no CSV).
DATASET_ORDER = [
    "gait", "occ", "mhealth", "smartphone", #"idosos",
    "pamap2_hand", "pamap2_chest", "pamap2_ankle", "usc_had",
    "ward_left_wrist", "ward_right_wrist", "ward_waist",
    "ward_left_ankle", "ward_right_ankle", "rs",
]

# ── Colunas exibidas na tabela, na ordem em que aparecem ────────────────────
COLUNAS = ["C0", "C1_best", "C1_worst", "C2", "C3_best", "C3_worst"]
ROTULO_COLUNA = {
    "C0"       : "C0",
    "C1_best"  : "C1 best",
    "C1_worst" : "C1 worst",
    "C2"       : "C2",
    "C3_best"  : "C3 best",
    "C3_worst" : "C3 worst",
}
# Colunas que carregam um detector associado (para sigla dentro da célula)
COLUNAS_COM_DETECTOR = {"C1_best", "C1_worst", "C3_best", "C3_worst"}

# ── Siglas/abreviações de detector para anotação dentro da célula. Já são
# nomes suficientemente claros (KSWIN, GMA, HDDMA...), por isso não há
# mais necessidade de uma legenda de siglas de duas letras no rodapé.
DET_CODE = {
    "ADWIN"                 : "ADWIN",
    "PageHinkley"           : "PH",
    "KSWIN"                 : "KSWIN",
    "CUSUM"                 : "CUSUM",
    "EWMAChart"             : "EWMAC",
    "GeometricMovingAverage": "GMA",
    "HDDMAverage"           : "HDDMA",
    "HDDMWeighted"          : "HDDMW",
    "SEED"                  : "SEED",
}

# ── Gradiente contínuo de cor para o F1 (branco -> denim acadêmico),
# escala ABSOLUTA e comum a todas as colunas (ver docstring). Mantido em
# RGB para interpolação direta em matplotlib, sem depender de paletas
# externas.
COR_BAIXO = np.array([1.00, 1.00, 1.00])   # branco
COR_ALTO  = np.array([0.21, 0.36, 0.49])   # denim acadêmico (#355C7D)
# A partir desta fração do gradiente, o texto passa a branco para
# permanecer legível sobre células escuras.
LIMIAR_TEXTO_CLARO = 0.55

# ============================================================
# LEITURA (idêntica ao radar_base_delta.py)
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, decimal=",")
    df.index.name = "base"
    return df

mean_df = read_table(MEAN_CSV)
std_df  = read_table(STD_CSV)

# ============================================================
# CLASSIFICAÇÃO DAS COLUNAS (idêntica ao radar_base_delta.py)
# ============================================================

def classify_col(col: str):
    col = col.strip()
    if col == "baseline":
        return "C0", None, False
    if col == "catch24":
        return "C2", None, False
    m = re.fullmatch(r"catch24_drift_ot_(.+)", col)
    if m:
        return "C3", m.group(1), True
    m = re.fullmatch(r"catch24_drift_(.+)", col)
    if m:
        return "C3", m.group(1), False
    m = re.fullmatch(r"drift_ot_(.+)", col)
    if m:
        return "C1", m.group(1), True
    m = re.fullmatch(r"drift_(.+)", col)
    if m:
        return "C1", m.group(1), False
    return "OTHER", col, False

col_meta = {col: classify_col(col) for col in mean_df.columns}

# ============================================================
# AGREGAÇÃO — best / worst, configuração DEFAULT apenas
# (mesmo espírito de extreme_in_cols() do radar_base_delta.py)
# ============================================================

def extreme_in_cols(cols, bases, mode="best"):
    rows = []
    for base in bases:
        target_f1  = -np.inf if mode == "best" else np.inf
        target_col = None
        for col in cols:
            if col not in mean_df.columns:
                continue
            v = mean_df.at[base, col]
            if pd.notna(v):
                if (mode == "best" and v > target_f1) or \
                   (mode == "worst" and v < target_f1):
                    target_f1, target_col = v, col
        if target_col is None:
            continue
        _, det, _ = col_meta[target_col]
        std_val = std_df.at[base, target_col] if target_col in std_df.columns else np.nan
        rows.append({
            "base"    : base,
            "detector": det,
            "f1"      : round(float(target_f1), 2),
            "std"     : round(float(std_val), 2) if pd.notna(std_val) else np.nan,
        })
    return pd.DataFrame(rows).set_index("base") if rows else pd.DataFrame()


def get_single(cat):
    bases = mean_df.index.tolist()
    cols  = [c for c, m in col_meta.items() if m[0] == cat]
    rows  = []
    for base in bases:
        for col in cols:
            v = mean_df.at[base, col]
            if pd.notna(v):
                std_val = std_df.at[base, col] if col in std_df.columns else np.nan
                rows.append({"base": base, "f1": round(float(v), 2),
                             "std": round(float(std_val), 2) if pd.notna(std_val) else np.nan})
    return pd.DataFrame(rows).set_index("base") if rows else pd.DataFrame()


bases_all = mean_df.index.tolist()

res_C0 = get_single("C0")
res_C2 = get_single("C2")

C1_def_cols = [c for c, m in col_meta.items() if m[0] == "C1" and not m[2]]
C3_def_cols = [c for c, m in col_meta.items() if m[0] == "C3" and not m[2]]

res_C1_best  = extreme_in_cols(C1_def_cols, bases_all, mode="best")
res_C1_worst = extreme_in_cols(C1_def_cols, bases_all, mode="worst")
res_C3_best  = extreme_in_cols(C3_def_cols, bases_all, mode="best")
res_C3_worst = extreme_in_cols(C3_def_cols, bases_all, mode="worst")

RES_POR_COLUNA = {
    "C0"      : res_C0,
    "C1_best" : res_C1_best,
    "C1_worst": res_C1_worst,
    "C2"      : res_C2,
    "C3_best" : res_C3_best,
    "C3_worst": res_C3_worst,
}

# ============================================================
# MONTAGEM DA TABELA (formato plano, para o CSV e para a figura)
# ============================================================

def get_f1(res, base):
    if res is None or res.empty or base not in res.index:
        return np.nan
    return float(res.at[base, "f1"])

def get_det_code(res, base):
    if res is None or res.empty or base not in res.index:
        return None
    d = res.at[base, "detector"]
    if pd.isna(d):
        return None
    return DET_CODE.get(d, str(d)[:2].upper())

ordered_bases = [b for b in DATASET_ORDER if b in bases_all] + \
                [b for b in bases_all if b not in DATASET_ORDER]

linhas = []
for base in ordered_bases:
    linha = {"dataset": ABBREV.get(base, base)}
    for col in COLUNAS:
        res = RES_POR_COLUNA[col]
        linha[col] = get_f1(res, base)
        if col in COLUNAS_COM_DETECTOR:
            linha[f"{col}_det"] = get_det_code(res, base)
    linhas.append(linha)

tabela = pd.DataFrame(linhas)
tabela.to_csv(OUTPUT_DIR / "tabela_heatmap_en.csv", index=False)
print(f"[OK] CSV salvo em: {OUTPUT_DIR / 'tabela_heatmap_en.csv'}")

# ============================================================
# ESCALA DE COR — ABSOLUTA, COMUM A TODAS AS COLUNAS DE F1, POR PERCENTIL
# ============================================================
# O intervalo de cor é calculado sobre as seis colunas de F1 juntas (escala
# global), de modo que a cor seja comparável tanto entre datasets (linhas)
# quanto entre cenários (colunas) — uma escala por coluna inverteria essa
# leitura.
#
# Usa-se PERCENTIL em vez de min/max bruto: outliers pontuais (ex: um único
# dataset perto de 100) puxariam o topo da escala e comprimiriam toda a
# variação útil — que de fato vive entre os percentis abaixo — contra uma
# faixa de cor estreita, tornando a tabela quase monocromática.
PERCENTIL_BAIXO  = 5
PERCENTIL_ALTO   = 95

_all_f1 = tabela[COLUNAS].to_numpy(dtype=float)
_all_f1 = _all_f1[~np.isnan(_all_f1)]
F1_MIN = float(np.percentile(_all_f1, PERCENTIL_BAIXO))
F1_MAX = float(np.percentile(_all_f1, PERCENTIL_ALTO))
print(f"[INFO] Escala de cor (percentis {PERCENTIL_BAIXO}-{PERCENTIL_ALTO}): "
      f"F1 em [{F1_MIN:.2f}, {F1_MAX:.2f}] "
      f"(min/max bruto: [{_all_f1.min():.2f}, {_all_f1.max():.2f}])")


def cor_para_f1(f1):
    """Interpola entre COR_BAIXO e COR_ALTO conforme a posição do F1 na
    escala global por percentil. Valores fora de [F1_MIN, F1_MAX] saturam
    na cor extrema correspondente (não desaparecem, só não criam contraste
    adicional). Retorna (cor_rgb, usar_texto_claro)."""
    if pd.isna(f1):
        return np.array([0.85, 0.85, 0.85]), False
    frac = (f1 - F1_MIN) / (F1_MAX - F1_MIN) if F1_MAX > F1_MIN else 0.0
    frac = min(max(frac, 0.0), 1.0)
    cor = COR_BAIXO + frac * (COR_ALTO - COR_BAIXO)
    return cor, frac >= LIMIAR_TEXTO_CLARO


# ============================================================
# FIGURA — mesma estrutura visual de teste_estatistico_cenarios_agr.py
# (Rectangle + texto centralizado, bordas brancas, rótulos em negrito
# fora da grade, legenda inferior), mas com gradiente contínuo de cor
# em vez de paleta categórica por magnitude.
# ============================================================

def gerar_figura(tabela: pd.DataFrame, caminho_saida: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_rows = len(tabela)
    n_cols = len(COLUNAS)

    cell_w, cell_h = 1.6, 1.0
    hdr_h          = 0.9
    row_label_w    = 1.8
    font_val, font_hdr, font_row, font_legend = 17, 16, 16, 15

    fig_w = row_label_w + cell_w * n_cols + 1.0
    fig_h = hdr_h + cell_h * n_rows + 1.0   # +1.0 reservado p/ colorbar

    # GridSpec dedica uma faixa inferior à colorbar, evitando add_axes()
    # manual (que é incompatível com tight_layout) e o respectivo warning.
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[fig_h - 1.0, 1.0], hspace=0.0,
                            figure=fig)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    cax.axis("off")

    ax.set_xlim(0, row_label_w + cell_w * n_cols)
    ax.set_ylim(0, fig_h - 1.0)
    ax.axis("off")

    y_top = (fig_h - 1.0) - hdr_h

    # Cabeçalho das colunas
    for c, col in enumerate(COLUNAS):
        x = row_label_w + c * cell_w + cell_w / 2
        ax.text(x, y_top + hdr_h / 2, ROTULO_COLUNA[col],
                ha="center", va="center",
                fontsize=font_hdr, fontweight="bold")

    # Linhas (datasets) e células
    for r, (_, row) in enumerate(tabela.iterrows()):
        y = y_top - (r + 1) * cell_h

        ax.text(row_label_w - 0.15, y + cell_h / 2, row["dataset"],
                ha="right", va="center",
                fontsize=font_row, fontweight="bold")

        for c, col in enumerate(COLUNAS):
            f1 = row[col]
            cor, texto_claro = cor_para_f1(f1)
            x = row_label_w + c * cell_w

            ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h,
                                       facecolor=cor, edgecolor="white",
                                       linewidth=1.5))

            if pd.isna(f1):
                val_str = "–"
            else:
                val_str = f"{f1:.2f}"
                if col in COLUNAS_COM_DETECTOR:
                    det_code = row.get(f"{col}_det")
                    if det_code:
                        val_str += f"\n{det_code}"

            ax.text(x + cell_w / 2, y + cell_h / 2, val_str,
                    ha="center", va="center",
                    fontsize=font_val, fontweight="bold",
                    color="white" if texto_claro else "black",
                    multialignment="center")

    # Barra de cor contínua (gradiente) no lugar da legenda categórica.
    # Um sub-eixo é colocado dentro de cax (já reservado pelo GridSpec),
    # deixando uma faixa abaixo livre para a legenda de detectores.
    # extend="both" desenha pequenas setas nas pontas da colorbar, sinalizando
    # que valores fora de [F1_MIN, F1_MAX] (percentis 5-95) existem na tabela
    # e estão saturados na cor extrema correspondente, em vez de fingir que
    # o intervalo mostrado é o range completo dos dados.
    cmap = LinearSegmentedColormap.from_list("f1_denim", [COR_BAIXO, COR_ALTO])
    norm = Normalize(vmin=F1_MIN, vmax=F1_MAX)
    sm   = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = cax.inset_axes([0.30, 0.30, 0.45, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label("F1-score macro (%)", fontsize=font_legend)
    cbar.ax.tick_params(labelsize=font_legend - 1)

    # Legenda de siglas removida: DET_CODE já usa nomes claros (KSWIN, GMA,
    # HDDMA, ...) exibidos diretamente em cada célula, sem necessidade de
    # glossário separado no rodapé.

    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura salva em: {caminho_saida}")


gerar_figura(tabela, FIGURA_SAIDA)

# ============================================================
# RESUMO NO TERMINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMO DA TABELA HEATMAP")
print("=" * 70)
with pd.option_context("display.max_columns", None,
                       "display.width", 200,
                       "display.max_colwidth", 18):
    print(tabela.to_string(index=False))
print(f"\nSaídas em: {OUTPUT_DIR.resolve()}")