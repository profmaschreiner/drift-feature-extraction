"""
analise_custo_extracao.py
==========================
Compara o custo de extração de características (ms/amostra) entre os
cenários C1 (drift features) e C2 (catch24), por dataset.

Evidencia a Contribuição 1: C1 é drasticamente mais barato que C2,
mantendo vetor de apenas 2n dimensões contra 24n do catch24.

Estrutura de pastas esperada (mesma do analise_comparativa_shap.py):
  <PASTA_RAIZ>/
    <dataset>/
      drift/
        <detector>/
          tempo_extracao_resumo.csv
      catch24/
        tempo_extracao_resumo.csv
      catch24_drift/
        <detector>/
          tempo_extracao_resumo.csv

Configure a seção CONFIGURAÇÕES antes de executar.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================================================
# CONFIGURAÇÕES — ajuste apenas esta seção
# =========================================================

PASTA_RAIZ = "exp_otimizacao/result_reais_completo/mdi_tempos_corrigido"
#
DATASETS = [ "mhealth",  "rs", "gait", "pamap2_hand", "pamap2_chest", "pamap2_ankle", "occ", "smartphone", "usc",   "W-RAnkle", "W-LWrist", "W-RWrist", "W-Waist", "W-LAnkle",  "sw", "sp"]

# Detector usado em C1 e C3 por dataset (mesmo dicionário do script SHAP).
# Se não listado, usa fallback automático (primeiro detector encontrado).
MELHOR_DETECTOR ={
    "mhealth":      {"drift": "HDDMAverage",                     "catch24_drift": "HDDMAverage"},
    "rs":           {"drift": "ADWIN",                  "catch24_drift": "ADWIN"},
    "gait":         {"drift": "KSWIN",                  "catch24_drift": "KSWIN"},
    "pamap2_hand":  {"drift": "SEED",                   "catch24_drift": "HDDMAverage"},
    "pamap2_chest": {"drift": "GeometricMovingAverage", "catch24_drift": "GeometricMovingAverage"},
    "pamap2_ankle": {"drift": "KSWIN",                  "catch24_drift": "KSWIN"},
    "occ":          {"drift": "HDDMWeighted",           "catch24_drift": "SEED"},
    "smartphone":   {"drift": "CUSUM",                     "catch24_drift": "CUSUM"},
    "usc":          {"drift": "PageHinkley",            "catch24_drift": "PageHinkley"},
    "W-LWrist":     {"drift": "GeometricMovingAverage",  "catch24_drift": "KSWIN"},
    "W-RWrist":     {"drift": "GeometricMovingAverage",  "catch24_drift": "KSWIN"},
    "W-Waist":      {"drift": "PageHinkley",                 "catch24_drift": "GeometricMovingAverage"},
    "W-LAnkle":     {"drift": "CUSUM",                     "catch24_drift": "HDDMAverage"},
    "W-RAnkle":     {"drift": "GeometricMovingAverage",  "catch24_drift": "CUSUM"},
    "sp":           {"drift": "PageHinkley",                     "catch24_drift": "SEED"},
    "sw":           {"drift": "CUSUM",                     "catch24_drift": "GeometricMovingAverage"},
}
"""
MELHOR_DETECTOR = {
    "gait":         {"drift": "KSWIN",                  "catch24_drift": "KSWIN"},
    "occ":          {"drift": "HDDMWeighted",            "catch24_drift": "SEED"},
    "pamap2_ankle": {"drift": "KSWIN",                  "catch24_drift": "KSWIN"},
    "pamap2_chest": {"drift": "GeometricMovingAverage", "catch24_drift": "GeometricMovingAverage"},
    "pamap2_hand":  {"drift": "SEED",                   "catch24_drift": "HDDMAverage"},
    "rs":           {"drift": "ADWIN",                  "catch24_drift": "ADWIN"},
    "usc":          {"drift": "PageHinkley",             "catch24_drift": "PageHinkley"},
    "W-LWrist":          {"drift": "GeometricMovingAverage",             "catch24_drift": "KSWIN"},
    "W-RAnkle":          {"drift": "GeometricMovingAverage",             "catch24_drift": "CUSUM"},
    
}
"""
NOMES_ABREVIADOS = {
    "gait":         "Gait",
    "occ":          "RO",
    "pamap2_ankle": "P2-Ankle",
    "pamap2_chest": "P2-Chest",
    "pamap2_hand":  "P2-Hand",
    "rs":           "DR",
    "mhealth":      "MH",
    "smartphone":        "SmPh",
    "usc":               "USC",
    "W-LWrist":          "W-LWrist",
    "W-RWrist":          "W-RWrist",
    "W-Waist":           "W-Waist",
    "W-LAnkle":          "W-LAnkle",
    "W-RAnkle":          "W-RAnkle",
    "sp":                "Sp",
    "sw":                "Sw"
}

PASTA_SAIDA = os.path.join(PASTA_RAIZ, "analise_custo")

# Cenários a comparar
CENARIOS = ["drift", "catch24", "catch24_drift"]

# Coluna principal de interesse nos CSVs de tempo de extração.
#
# ATENÇÃO: t_por_amostra_*_ms_mean subestima a diferença entre detectores
# porque divide o tempo total pelo número de amostras — que pode ser muito
# grande (ex. rs ~790k amostras), diluindo diferenças reais entre detectores
# como ADWIN (~9s total) e KSWIN (~383s total no dataset rs).
#
# A métrica correta para comparar detectores é o TEMPO TOTAL de extração
# sobre todos os folds, que reflete o custo real de processar o dataset.
# Para normalizar entre datasets de tamanhos distintos, dividimos pelo
# número de folds (t_total_extracao_mean_s = média por fold).
COL_EXTRACAO    = "t_total_extracao_todos_folds_s"  # tempo total (s) — todos os folds
COL_EXTRACAO_MS = "t_total_extracao_mean_s"          # tempo médio por fold (s)
COL_TREINAMENTO = "t_fit_por_amostra_ms_mean"        # ms por amostra de treino RF

# Cores por cenário (mesma paleta usada nas demais figuras do material
# suplementar — laranja/azul/verde — em vez de uma quarta cor ad-hoc,
# para manter consistência visual entre todas as figuras do artigo).
CORES = {
    "drift":         "#E05C2A",   # laranja — C1
    "catch24":       "#2A7AE0",   # azul    — C2
    "catch24_drift": "#2EAA5B",   # verde   — C3
}

LABELS = {
    "drift":         "C1 — Drift",
    "catch24":       "C2 — catch24",
    "catch24_drift": "C3 — catch24 + Drift",
}

# Abreviações dos detectores, iguais às usadas no artigo (Tabela S-II do
# material suplementar: ADWIN, PH, KSWIN, CUSUM, EWMAC, GMA, HDDMA, HDDMW,
# SEED). Usado em qualquer figura que precise exibir o nome do detector.
DET_ABREV = {
    "ADWIN":                  "ADWIN",
    "PageHinkley":            "PH",
    "KSWIN":                  "KSWIN",
    "CUSUM":                  "CUSUM",
    "EWMAChart":              "EWMAC",
    "GeometricMovingAverage": "GMA",
    "HDDMAverage":            "HDDMA",
    "HDDMWeighted":           "HDDMW",
    "SEED":                   "SEED",
}

# Escala do eixo de tempo: "linear" ou "log"
ESCALA_Y = "log"

# Fonte (usada pelas demais figuras deste script, não padronizadas nesta etapa)
FONTSIZE = 13

# ---------------------------------------------------------------------------
# Estilo padronizado para as figuras destinadas ao material suplementar,
# igual ao usado em analise_comparativa_shap_mdi.py: fonte serifada
# compatível com o corpo do artigo (IEEEtran/Times), largura de uma coluna
# IEEE, e barras horizontais (evitam rótulos de dataset rotacionados).
# ---------------------------------------------------------------------------
ESTILO_SUPLEMENTAR = {
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":       8,
    "axes.titlesize":  8,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth":  0.6,
    "grid.linewidth":  0.5,
}

LARGURA_COLUNA_IN     = 3.45  # largura útil de uma coluna IEEE (~88 mm)
ALTURA_POR_DATASET_IN = 0.30  # altura reservada por dataset (barra horizontal)
ALTURA_MARGEM_IN      = 1.05  # margem fixa (eixo, rótulo em 2 linhas, legenda)
ALTURA_MINIMA_IN      = 1.6

# =========================================================
# UTILITÁRIOS
# =========================================================

def detectores_disponiveis(pasta_cenario: str) -> list:
    """Lista subdiretórios de detectores numa pasta de cenário."""
    if not os.path.isdir(pasta_cenario):
        return []
    return [d for d in os.listdir(pasta_cenario)
            if os.path.isdir(os.path.join(pasta_cenario, d))]


def carregar_resumo_extracao(pasta: str) -> pd.DataFrame | None:
    """Carrega tempo_extracao_resumo.csv de uma pasta."""
    caminho = os.path.join(pasta, "tempo_extracao_resumo.csv")
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)


def carregar_resumo_treinamento(pasta: str) -> pd.DataFrame | None:
    """Carrega tempo_treinamento_rf_resumo.csv de uma pasta."""
    caminho = os.path.join(pasta, "tempo_treinamento_rf_resumo.csv")
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)


def valor_col(df: pd.DataFrame, col: str) -> float | None:
    """Extrai valor escalar de coluna, retorna None se ausente."""
    if df is None or col not in df.columns:
        return None
    v = df[col].iloc[0]
    return float(v) if pd.notna(v) else None


def carregar_dados_dataset(pasta_dataset: str, dataset_nome: str) -> dict:
    """
    Carrega tempos de extração e treinamento para os três cenários.
    Retorna dict: cenario → {'extracao': float|None, 'treinamento': float|None,
                              'detector': str|None}
    """
    resultado = {}

    for cenario in CENARIOS:
        pasta_cenario = os.path.join(pasta_dataset, cenario)

        if cenario == "catch24":
            # catch24 não tem subpasta de detector
            df_ext = carregar_resumo_extracao(pasta_cenario)
            df_tre = carregar_resumo_treinamento(pasta_cenario)
            resultado[cenario] = {
                "extracao":    valor_col(df_ext, COL_EXTRACAO),
                "treinamento": valor_col(df_tre, COL_TREINAMENTO),
                "detector":    None,
            }
        else:
            # Prioridade: MELHOR_DETECTOR → fallback primeiro encontrado
            det_fixo = MELHOR_DETECTOR.get(dataset_nome, {}).get(cenario)
            if det_fixo:
                pasta_det = os.path.join(pasta_cenario, det_fixo)
                df_ext = carregar_resumo_extracao(pasta_det)
                df_tre = carregar_resumo_treinamento(pasta_det)
                detector_usado = det_fixo
            else:
                dets = detectores_disponiveis(pasta_cenario)
                df_ext, df_tre, detector_usado = None, None, None
                for d in dets:
                    pasta_det = os.path.join(pasta_cenario, d)
                    df_ext = carregar_resumo_extracao(pasta_det)
                    df_tre = carregar_resumo_treinamento(pasta_det)
                    if df_ext is not None:
                        detector_usado = d
                        break

            resultado[cenario] = {
                "extracao":    valor_col(df_ext, COL_EXTRACAO),
                "treinamento": valor_col(df_tre, COL_TREINAMENTO),
                "detector":    detector_usado,
            }

    return resultado



def carregar_todos_detectores_dataset(pasta_dataset: str) -> dict:
    """
    Para o cenário drift, carrega o custo de extração de TODOS os detectores
    disponíveis. Retorna dict:
      {detector: valor_ms | None}
    Também retorna o custo de catch24 (sem detector).
    """
    resultado = {}

    # C1 — todos os detectores
    pasta_drift = os.path.join(pasta_dataset, "drift")
    dets = detectores_disponiveis(pasta_drift)
    for det in sorted(dets):
        df = carregar_resumo_extracao(os.path.join(pasta_drift, det))
        resultado[det] = valor_col(df, COL_EXTRACAO)

    # C2 — catch24 (referência)
    df_c2 = carregar_resumo_extracao(os.path.join(pasta_dataset, "catch24"))
    resultado["__catch24__"] = valor_col(df_c2, COL_EXTRACAO)

    return resultado

# =========================================================
# FIGURA — Custo de extração por dataset (C1 vs C2 vs C3)
# =========================================================

def figura_custo_extracao(resumo: list, pasta_saida: str):
    """
    Horizontal bar chart: total extraction time (s) per dataset and
    scenario (C1/C2/C3), one dataset per row. Standardized with the other
    supplementary-material figures of the manuscript: serif font, IEEE
    single-column width, horizontal bars (avoids rotated dataset labels),
    and the same orange/blue/green palette used throughout, instead of a
    scenario-specific color. The title is intentionally omitted — as with
    the other standardized figures, that role is filled by the LaTeX
    caption below the figure.
    """
    df = pd.DataFrame(resumo)
    if df.empty:
        print("  [AVISO] Nenhum dado de extração disponível.")
        return

    datasets  = list(df["dataset"].unique())
    cenarios_presentes = [c for c in CENARIOS
                          if df[df["cenario"] == c]["extracao"].notna().any()]
    n_cen = len(cenarios_presentes)

    largura = 0.22
    gap     = 0.04
    passo   = largura + gap
    y       = np.arange(len(datasets))

    altura_fig = max(
        ALTURA_MINIMA_IN,
        len(datasets) * ALTURA_POR_DATASET_IN + ALTURA_MARGEM_IN,
    )

    with plt.rc_context(ESTILO_SUPLEMENTAR):
        fig, ax = plt.subplots(figsize=(LARGURA_COLUNA_IN, altura_fig))

        for i, cenario in enumerate(cenarios_presentes):
            sub    = df[df["cenario"] == cenario].set_index("dataset")
            offset = (i - n_cen / 2 + 0.5) * passo
            vals   = np.array([sub.loc[d, "extracao"] if d in sub.index
                               and pd.notna(sub.loc[d, "extracao"]) else np.nan
                               for d in datasets])
            errs   = np.array([sub.loc[d, "std_extracao"] if d in sub.index
                               and pd.notna(sub.loc[d, "std_extracao"]) else 0.0
                               for d in datasets])
            # Detector usado em cada dataset para este cenário (vazio para
            # catch24, que não tem detector associado).
            dets   = [sub.loc[d, "detector"] if d in sub.index
                     and pd.notna(sub.loc[d, "detector"]) else ""
                     for d in datasets]

            bars = ax.barh(y + offset, vals, height=largura,
                           color=CORES[cenario], edgecolor="white",
                           linewidth=0.4, label=cenario)

            # Barra de erro apenas para a direita (std sempre ≥ 0)
            mascara = ~np.isnan(vals)
            if mascara.any():
                ax.errorbar(vals[mascara], (y + offset)[mascara],
                            xerr=errs[mascara],
                            fmt="none", color="#333333",
                            capsize=1.5, linewidth=0.6, capthick=0.6)

            # Anotação do valor ao final de cada barra (legível na horizontal).
            # Para os cenários com detector (drift / catch24_drift), o nome
            # do detector usado naquele dataset é anexado entre parênteses.
            for bar, v, det in zip(bars, vals, dets):
                if not np.isnan(v) and v > 0:
                    sufixo = (f" ({DET_ABREV.get(det, det)})"
                             if cenario in {"drift", "catch24_drift"} and det
                             else "")
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                            f" {v:.1f}{sufixo}", ha="left", va="center", fontsize=5.5,
                            color="#333333")

        ax.set_yticks(y)
        ax.set_yticklabels([NOMES_ABREVIADOS.get(d, d) for d in datasets])
        ax.invert_yaxis()  # primeiro dataset no topo
        ax.set_xlabel("Total extraction time (s)", labelpad=6)

        if ESCALA_Y == "log":
            ax.set_xscale("log")
            ax.set_xlabel("Total extraction time (s) — log scale", labelpad=6)
        # Margem extra à direita para caber as anotações de valor + abreviação
        # do detector (ex. "GMA")
        ax.margins(x=0.25)

        ax.spines[["top", "right"]].set_visible(False)

        legenda = [mpatches.Patch(color=CORES[c], label=LABELS[c])
                   for c in cenarios_presentes]
        plt.tight_layout(rect=[0, 0.14, 1, 1])
        fig.legend(handles=legenda, loc="lower center", ncol=n_cen,
                   frameon=False, handlelength=1.2,
                   bbox_to_anchor=(0.5, 0.01))

        caminho = os.path.join(pasta_saida, "custo_extracao_por_dataset.pdf")
        fig.savefig(caminho, bbox_inches="tight", dpi=300)
        plt.close(fig)
    print(f"  [OK] {caminho}")


# =========================================================
# FIGURA — Razão C2/C1 por dataset (fator de aceleração)
# =========================================================

def figura_razao_c2_c1(resumo: list, pasta_saida: str):
    """
    Barplot horizontal: razão C2/C1 por dataset.
    Mostra diretamente quantas vezes C1 é mais rápido que C2.
    """
    df = pd.DataFrame(resumo)
    if df.empty:
        return

    rows = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].set_index("cenario")
        c1 = sub.loc["drift",  "extracao"] if "drift"  in sub.index else None
        c2 = sub.loc["catch24","extracao"] if "catch24" in sub.index else None
        if c1 and c2 and c1 > 0:
            rows.append({
                "dataset": dataset,
                "label":   NOMES_ABREVIADOS.get(dataset, dataset),
                "razao":   c2 / c1,
                "c1_ms":   c1,
                "c2_ms":   c2,
            })

    if not rows:
        print("  [AVISO] Nenhum par C1/C2 disponível para razão.")
        return

    df_r = pd.DataFrame(rows).sort_values("razao", ascending=True)
    y    = np.arange(len(df_r))

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_r) * 0.6)))
    bars = ax.barh(y, df_r["razao"], color=CORES["catch24"],
                   edgecolor="none", height=0.55)
    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(df_r["label"], fontsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.set_xlabel("Razão C2 / C1  (vezes mais lento)", fontsize=FONTSIZE)
    ax.set_title("C2 é X vezes mais lento que C1 por amostra",
                 fontsize=FONTSIZE, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Anotação: C1=X ms | C2=Y ms
    for bar, (_, row) in zip(bars, df_r.iterrows()):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"C1={row['c1_ms']:.2f} ms | C2={row['c2_ms']:.2f} ms",
                va="center", fontsize=FONTSIZE - 3, color="#555555")

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "razao_c2_c1_por_dataset.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")



# =========================================================
# FIGURA — Todos os detectores vs catch24 por dataset
# =========================================================

def figura_todos_detectores_vs_catch24(resumo_completo: list, pasta_saida: str):
    """
    Uma figura com N painéis (um por dataset).
    Em cada painel:
      - Um ponto laranja por detector de drift (custo C1 individual)
      - Uma linha horizontal azul tracejada = custo catch24 (C2)
      - Eixo Y em escala log para lidar com ordens de magnitude distintas

    Permite ao leitor ver que TODOS os detectores são mais baratos que
    catch24, independente da escolha — argumento robusto para defesa da
    metodologia sem depender do melhor detector.
    """
    if not resumo_completo:
        return

    datasets = list({r["dataset"] for r in resumo_completo})
    datasets = sorted(datasets)
    n_ds     = len(datasets)

    # Layout: até 4 colunas
    n_cols = min(4, n_ds)
    n_rows = int(np.ceil(n_ds / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.8),
                             sharey=False)
    axes = np.array(axes).reshape(n_rows, n_cols)

    # Coleta todos os detectores encontrados (para ordenação consistente)
    todos_dets = sorted({r["detector"] for r in resumo_completo
                         if r["detector"] != "__catch24__"})

    # Mapeamento detector → posição X (para scatter consistente entre painéis)
    det_x = {d: i for i, d in enumerate(todos_dets)}

    for idx, dataset_nome in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row, col]

        # Dados deste dataset
        rows_ds = [r for r in resumo_completo if r["dataset"] == dataset_nome]
        catch24_val = next((r["extracao"] for r in rows_ds
                            if r["detector"] == "__catch24__"), None)
        dets_ds = [(r["detector"], r["extracao"]) for r in rows_ds
                   if r["detector"] != "__catch24__" and r["extracao"] is not None]

        # Linha horizontal catch24
        if catch24_val is not None:
            ax.axhline(catch24_val, color=CORES["catch24"],
                       linewidth=1.8, linestyle="--",
                       label="C2 — catch24" if idx == 0 else "_nolegend_")
            ax.text(len(todos_dets) - 0.3, catch24_val * 1.25,
                    f"{catch24_val:.1f} s",
                    color=CORES["catch24"], fontsize=FONTSIZE - 5,
                    va="bottom", ha="right")

        # Pontos por detector
        xs, ys, labels_det = [], [], []
        for det, val in dets_ds:
            if det in det_x:
                xs.append(det_x[det])
                ys.append(val)
                labels_det.append(det)

        if xs:
            ax.scatter(xs, ys, color=CORES["drift"], zorder=5,
                       s=60, label="C1 — Drift (por detector)" if idx == 0 else "_nolegend_")
            # Anotação do valor abaixo de cada ponto
            for xi, yi, lbl in zip(xs, ys, labels_det):
                ax.text(xi, yi * 0.78, f"{yi:.1f}s",
                        ha="center", va="top",
                        fontsize=FONTSIZE - 6, color="#555555")

        ax.set_yscale("log")
        ax.set_xticks(range(len(todos_dets)))
        ax.set_xticklabels([DET_ABREV.get(d, d) for d in todos_dets],
                           rotation=45, ha="right", fontsize=FONTSIZE - 4)
        ax.tick_params(axis="y", labelsize=FONTSIZE - 3)
        ax.set_title(NOMES_ABREVIADOS.get(dataset_nome, dataset_nome),
                     fontsize=FONTSIZE - 1, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
        ax.spines[["top", "right"]].set_visible(False)

        if col == 0:
            ax.set_ylabel("Tempo total de extração (s) — log", fontsize=FONTSIZE - 2)

    # Oculta painéis vazios
    for idx in range(n_ds, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Legenda global
    legenda = [
        mpatches.Patch(color=CORES["drift"],   label="C1 — Drift (cada detector)"),
        mpatches.Patch(color=CORES["catch24"], label="C2 — catch24 (referência)"),
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=2,
               fontsize=FONTSIZE - 1, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Tempo total de extração — C1 (por detector) vs. C2 (catch24)",
                 fontsize=FONTSIZE + 1, fontweight="bold", y=1.01)
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "custo_todos_detectores_vs_catch24.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")

# =========================================================
# TABELA RESUMO — min / mediana / max / melhor detector / C2 / razão
# =========================================================

def tabela_resumo_custo(resumo_completo: list, pasta_saida: str):
    """
    Gera CSV com uma linha por dataset contendo:
      dataset, label,
      c1_min_s, c1_min_detector,
      c1_mediana_s,
      c1_max_s, c1_max_detector,
      c1_melhor_s, c1_melhor_detector,   ← detector definido em MELHOR_DETECTOR
      c2_s,
      razao_c2_mediana,                  ← C2 / mediana C1
      razao_c2_melhor                    ← C2 / melhor detector C1
    """
    if not resumo_completo:
        return

    rows = []
    datasets = sorted({r["dataset"] for r in resumo_completo})

    for dataset_nome in datasets:
        dets_ds = [r for r in resumo_completo
                   if r["dataset"] == dataset_nome
                   and r["detector"] != "__catch24__"
                   and r["extracao"] is not None]
        c2_row = next((r for r in resumo_completo
                       if r["dataset"] == dataset_nome
                       and r["detector"] == "__catch24__"), None)

        if not dets_ds:
            continue

        vals    = [(r["detector"], r["extracao"]) for r in dets_ds]
        tempos  = [v[1] for v in vals]
        detects = [v[0] for v in vals]

        min_idx = int(np.argmin(tempos))
        max_idx = int(np.argmax(tempos))
        mediana = float(np.median(tempos))
        c2_val  = c2_row["extracao"] if c2_row else None

        # Melhor detector conforme MELHOR_DETECTOR (para cenário drift)
        melhor_det_nome = MELHOR_DETECTOR.get(dataset_nome, {}).get("drift")
        melhor_val = next((t for d, t in vals if d == melhor_det_nome), None)

        row = {
            "dataset":            dataset_nome,
            "label":              NOMES_ABREVIADOS.get(dataset_nome, dataset_nome),
            "c1_min_s":           round(tempos[min_idx], 2),
            "c1_min_detector":    detects[min_idx],
            "c1_mediana_s":       round(mediana, 2),
            "c1_max_s":           round(tempos[max_idx], 2),
            "c1_max_detector":    detects[max_idx],
            "c1_melhor_s":        round(melhor_val, 2) if melhor_val else None,
            "c1_melhor_detector": melhor_det_nome or "",
            "c2_s":               round(c2_val, 2) if c2_val else None,
            "razao_c2_mediana":   round(c2_val / mediana, 1) if c2_val and mediana else None,
            "razao_c2_melhor":    round(c2_val / melhor_val, 1) if c2_val and melhor_val else None,
        }
        rows.append(row)
        print(f"  {row['label']}: C1 min={row['c1_min_s']}s ({row['c1_min_detector']}) "
              f"| med={row['c1_mediana_s']}s | max={row['c1_max_s']}s ({row['c1_max_detector']}) "
              f"| melhor={row['c1_melhor_s']}s ({row['c1_melhor_detector']}) "
              f"| C2={row['c2_s']}s | razão_med={row['razao_c2_mediana']}x")

    df = pd.DataFrame(rows)
    caminho = os.path.join(pasta_saida, "tabela_resumo_custo.csv")
    df.to_csv(caminho, index=False)
    print(f"  [OK] {caminho}")
    return df


# =========================================================
# CSV CONSOLIDADO
# =========================================================

def salvar_csv(resumo: list, pasta_saida: str):
    caminho = os.path.join(pasta_saida, "resumo_custo_extracao.csv")
    pd.DataFrame(resumo).to_csv(caminho, index=False)
    print(f"  [OK] {caminho}")


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    if DATASETS is not None:
        datasets_lista = DATASETS
    else:
        datasets_lista = sorted([
            d for d in os.listdir(PASTA_RAIZ)
            if os.path.isdir(os.path.join(PASTA_RAIZ, d))
            and d != os.path.basename(PASTA_SAIDA)
        ])

    print(f"Datasets encontrados: {datasets_lista}\n")

    resumo          = []
    resumo_completo = []   # um registro por dataset × detector

    for dataset_nome in datasets_lista:
        pasta_dataset = os.path.join(PASTA_RAIZ, dataset_nome)
        print(f"{'=' * 50}")
        print(f"Dataset: {dataset_nome}")

        # --- Dados por cenário (melhor detector por dataset) ---
        dados = carregar_dados_dataset(pasta_dataset, dataset_nome)

        for cenario, info in dados.items():
            ext = info["extracao"]
            tre = info["treinamento"]
            det = info["detector"]
            status = f"{ext:.3f} ms/am" if ext else "N/A"
            print(f"  {LABELS.get(cenario, cenario)}"
                  + (f" [{det}]" if det else "")
                  + f": extração={status}")

            resumo.append({
                "dataset":       dataset_nome,
                "label":         NOMES_ABREVIADOS.get(dataset_nome, dataset_nome),
                "cenario":       cenario,
                "detector":      det or "",
                "extracao":      ext,
                "std_extracao":  0.0,
                "treinamento":   tre,
            })

        # --- Dados de todos os detectores (para figura comparativa) ---
        todos = carregar_todos_detectores_dataset(pasta_dataset)
        for det, val in todos.items():
            resumo_completo.append({
                "dataset":  dataset_nome,
                "detector": det,
                "extracao": val,
            })
            lbl = "catch24" if det == "__catch24__" else det
            status = f"{val:.4f} ms/am" if val else "N/A"
            print(f"    {lbl}: {status}")

        print()

    figura_custo_extracao(resumo, PASTA_SAIDA)
    figura_razao_c2_c1(resumo, PASTA_SAIDA)
    figura_todos_detectores_vs_catch24(resumo_completo, PASTA_SAIDA)
    salvar_csv(resumo, PASTA_SAIDA)
    print("\nGerando tabela resumo de custo...")
    tabela_resumo_custo(resumo_completo, PASTA_SAIDA)

    print("\nAnálise concluída.")
    print(f"Resultados em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()