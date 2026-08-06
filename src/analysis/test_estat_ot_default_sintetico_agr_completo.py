"""
test_estat_ot_default_sintetico_agregado.py
=============================================
Versão agregada por dataset do teste estatístico default vs. otimizado,
aplicada aos 20 conjuntos de dados sintéticos (gerados via CaDrift).

Diferença em relação ao script original (datasets reais)
-----------------------------------------------------------
Idêntico em lógica ao script para os conjuntos de dados reais: o Wilcoxon
é aplicado pareado por dataset (média dos folds), não por (dataset × fold),
resultando em ~20 pares genuinamente independentes por detector — seguindo
a recomendação de Demšar (2006). Apenas a pasta raiz e os rótulos textuais
foram ajustados para os conjuntos de dados sintéticos.

Motivação
---------
Folds de um mesmo dataset compartilham dados de treinamento e não são
estritamente independentes. A agregação por dataset elimina essa dependência,
ancorando o teste em unidades de observação independentes entre si.

Saídas
------
* Console : tabela por detector × cenário (F1 e Acc separados)
* CSV     : analise_default_vs_otimizado_agregado_sintetico_{metrica}.csv
* PNG     : tabela_otimizacao_sintetico_agregado.png

Configuração: ajuste apenas PASTA_RAIZ.
"""

import os
import warnings
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# CONFIGURAÇÃO — ajuste apenas esta seção
# ============================================================

PASTA_RAIZ = "exp_otimizacao/result_sintetico_completo"

DETECTORES = [
    "ADWIN",
    "PageHinkley",
    "KSWIN",
    "CUSUM",
    "EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    "SEED",
]

PARES_CENARIO = [
    ("drift",         "drift_ot",         "C1"),
    ("catch24_drift", "catch24_drift_ot", "C3"),
]

METRICAS = [
    ("df_f1_all.csv",  "F1-score macro", "f1"),
    ("df_acc_all.csv", "Acurácia",       "acc"),
]

COLUNA_METRICA = "RF"
ALPHA_FDR      = 0.05

# Filtro por sufixo do nome da pasta do dataset. Se "" (vazio), processa
# todas as pastas encontradas. Se preenchido (ex.: "_D_"), processa apenas
# as pastas cujo nome TERMINA com esse sufixo.
FILTRO_SUFIXO = "_D_"

# Subpasta fixa intermediária presente em todos os datasets sintéticos,
# entre a pasta do dataset e as pastas de cenário (drift, drift_ot,
# catch24_drift, catch24_drift_ot, etc.). Ex.: estrutura real é
# <dataset>/resultados_catch24_drift/<cenario>/<detector>/df_f1_all.csv
SUBPASTA_EXPERIMENTO = "resultados_catch24_drift"

FIGURA_SAIDA = os.path.join(PASTA_RAIZ, "tabela_otimizacao_sintetico_agregado.png")

# ============================================================
# DESCOBERTA DE PASTAS
# ============================================================

def descobrir_pastas_saida(pasta_raiz: str, filtro_sufixo: str = "",
                           subpasta_experimento: str = "") -> list:
    raiz = Path(pasta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Pasta raiz não encontrada: {pasta_raiz}\n"
            "Ajuste a variável PASTA_RAIZ no início do script."
        )

    candidatos = sorted([p for p in raiz.iterdir() if p.is_dir()])

    if filtro_sufixo:
        antes = len(candidatos)
        candidatos = [p for p in candidatos if p.name.endswith(filtro_sufixo)]
        print(
            f"Filtro de sufixo ativo ('{filtro_sufixo}'): "
            f"{len(candidatos)} de {antes} pastas de dataset mantidas."
        )

    pastas = []
    for p in candidatos:
        base = (p / subpasta_experimento) if subpasta_experimento else p
        if base.is_dir() and ((base / "drift").is_dir() or (base / "drift_ot").is_dir()):
            pastas.append(str(base))

    if not pastas:
        if filtro_sufixo:
            raise RuntimeError(
                f"Nenhuma pasta com nome terminando em '{filtro_sufixo}' "
                f"encontrada em '{pasta_raiz}' contendo a subpasta "
                f"'{subpasta_experimento}' com drift/ ou drift_ot/."
            )
        raise RuntimeError(
            f"Nenhuma subpasta válida encontrada em '{pasta_raiz}'.\n"
            f"Verifique se as pastas contêm '{subpasta_experimento}/drift' "
            f"ou '{subpasta_experimento}/drift_ot'."
        )
    print(f"Datasets sintéticos encontrados ({len(pastas)}): "
          f"{[Path(p).parent.name for p in pastas]}")
    return pastas


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def cliff_delta_paired(default: np.ndarray, otimizado: np.ndarray) -> float:
    d = otimizado - default
    n = len(d)
    if n == 0:
        return np.nan
    return float((np.sum(d > 0) - np.sum(d < 0)) / n)


def magnitude_cliff(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "irrelevante"
    if ad < 0.330:
        return "pequeno"
    if ad < 0.474:
        return "médio"
    return "grande"


def bh_correction(p_values: list, alpha: float = 0.05) -> list:
    n = len(p_values)
    if n == 0:
        return []
    orden      = np.argsort(p_values)
    p_sorted   = np.array(p_values)[orden]
    threshold  = (np.arange(1, n + 1) / n) * alpha
    rej_sorted = p_sorted <= threshold
    rej_sorted = np.maximum.accumulate(rej_sorted[::-1])[::-1]
    rejeita    = np.empty(n, dtype=bool)
    rejeita[orden] = rej_sorted
    return rejeita.tolist()


def load_csv(pasta_saida: str, scenario: str, detector: str,
             arquivo: str) -> pd.DataFrame:
    path = os.path.join(pasta_saida, scenario, detector, arquivo)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols_needed = {"run", "fold", COLUNA_METRICA}
    if not cols_needed.issubset(df.columns):
        warnings.warn(f"Arquivo sem colunas esperadas: {path}")
        return pd.DataFrame()
    return df[["run", "fold", COLUNA_METRICA]].copy()


# ============================================================
# COLETA DE PARES (fold-level — igual ao original)
# ============================================================

def coletar_pares(pastas_saida: list, detectores: list,
                  pares_cenario: list, arquivo: str) -> pd.DataFrame:
    """
    Coleta pares (default × otimizado) pareados por (dataset_pasta × fold).
    Idêntico ao script original — a agregação ocorre em etapa posterior.
    """
    rows = []

    for pasta, (scen_def, scen_ot, rotulo), detector in iproduct(
        pastas_saida, pares_cenario, detectores
    ):
        df_def = load_csv(pasta, scen_def, detector, arquivo)
        df_ot  = load_csv(pasta, scen_ot,  detector, arquivo)

        if df_def.empty or df_ot.empty:
            continue

        dataset_pasta = Path(pasta).parent.name if SUBPASTA_EXPERIMENTO else Path(pasta).name

        df_def = df_def.rename(columns={
            "run": "run_default",
            COLUNA_METRICA: "val_default",
        })
        df_ot = df_ot.rename(columns={
            "run": "run_otimizado",
            COLUNA_METRICA: "val_otimizado",
        })

        df_def["dataset_pasta"] = dataset_pasta
        df_ot["dataset_pasta"]  = dataset_pasta

        merged = pd.merge(
            df_def, df_ot,
            on=["dataset_pasta", "fold"],
            how="inner",
        )

        if merged.empty:
            warnings.warn(
                f"Nenhum fold em comum: pasta={dataset_pasta}, "
                f"cenário={rotulo}, detector={detector}"
            )
            continue

        merged["pasta"]    = pasta
        merged["cenario"]  = rotulo
        merged["detector"] = detector

        rows.append(merged[[
            "dataset_pasta", "pasta", "cenario", "detector",
            "fold", "run_default", "run_otimizado",
            "val_default", "val_otimizado",
        ]])

    if not rows:
        raise RuntimeError(
            f"Nenhum par (default, otimizado) encontrado para arquivo={arquivo}."
        )

    return pd.concat(rows, ignore_index=True)


# ============================================================
# AGREGAÇÃO POR DATASET  ← núcleo desta versão
# ============================================================

def agregar_por_dataset(df_pares: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os pares por dataset calculando a média de val_default e
    val_otimizado sobre todos os folds de cada dataset.

    Entrada : df_pares com colunas dataset_pasta, cenario, detector,
              fold, val_default, val_otimizado
    Saída   : df com uma linha por (dataset_pasta, cenario, detector),
              com val_default e val_otimizado sendo a média dos folds
    """
    agg = (
        df_pares
        .groupby(["dataset_pasta", "cenario", "detector"], as_index=False)
        .agg(
            val_default   = ("val_default",   "mean"),
            val_otimizado = ("val_otimizado",  "mean"),
            n_folds       = ("fold",           "count"),
        )
    )
    return agg


# ============================================================
# ANÁLISE ESTATÍSTICA SOBRE PARES AGREGADOS
# ============================================================

def analisar_agregado(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Wilcoxon + Cliff's Delta sobre os pares por detector,
    onde cada par é a média de um dataset (independente dos demais).
    Segue Demšar (2006): unidade de pareamento = dataset.
    """
    resultados = []

    for (cenario, detector), grupo in df_agg.groupby(["cenario", "detector"]):

        def_arr     = grupo["val_default"].to_numpy(dtype=float)
        ot_arr      = grupo["val_otimizado"].to_numpy(dtype=float)
        dif         = ot_arr - def_arr

        n_datasets  = len(dif)
        n_folds_med = round(grupo["n_folds"].mean(), 1)
        n_nao_zero  = int(np.sum(dif != 0))
        media_dif   = float(np.mean(dif))
        mediana_dif = float(np.median(dif))
        delta       = cliff_delta_paired(def_arr, ot_arr)
        mag         = magnitude_cliff(delta)

        if n_nao_zero < 1:
            p_value = 1.0
            stat    = np.nan
        else:
            try:
                stat, p_value = wilcoxon(dif[dif != 0], alternative="two-sided")
            except Exception as e:
                warnings.warn(f"Wilcoxon falhou ({cenario}, {detector}): {e}")
                stat, p_value = np.nan, 1.0

        resultados.append({
            "cenario"       : cenario,
            "detector"      : detector,
            "n_datasets"    : n_datasets,
            "n_folds_medio" : n_folds_med,
            "n_nao_zero"    : n_nao_zero,
            "media_dif_pp"  : round(media_dif   * 100, 2),
            "mediana_dif_pp": round(mediana_dif  * 100, 2),
            "cliff_delta"   : round(delta, 4),
            "magnitude"     : mag,
            "wilcoxon_stat" : round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value"       : round(p_value, 6),
        })

    df_res = pd.DataFrame(resultados)
    df_res["sig_bh"] = False

    # Correção Benjamini-Hochberg por cenário (9 detectores por cenário)
    for cenario, grupo in df_res.groupby("cenario"):
        idx     = grupo.index.tolist()
        p_vals  = grupo["p_value"].tolist()
        rejeita = bh_correction(p_vals, alpha=ALPHA_FDR)
        df_res.loc[idx, "sig_bh"] = rejeita

    df_res["sig_bh"] = df_res["sig_bh"].astype(bool)

    def recomendacao(row):
        if row["sig_bh"] and row["magnitude"] != "irrelevante" and row["media_dif_pp"] > 0:
            return "SIM"
        if row["sig_bh"] and row["magnitude"] != "irrelevante" and row["media_dif_pp"] < 0:
            return "NÃO (piora)"
        return "NÃO (efeito irrelevante ou não significativo)"

    df_res["recomenda_otimizar"] = df_res.apply(recomendacao, axis=1)

    return df_res.sort_values(["cenario", "detector"]).reset_index(drop=True)


# ============================================================
# TABELA DE MÉDIAS — C1.A/C1.B/C3.A/C3.B por detector
# ============================================================

NOME_PARA_LINHA_MEDIAS = {
    "ADWIN"                  : "ADWIN",
    "PageHinkley"             : "PH",
    "KSWIN"                   : "KSWIN",
    "CUSUM"                   : "CUSUM",
    "EWMAChart"               : "EWMAC",
    "GeometricMovingAverage"  : "GMA",
    "HDDMAverage"             : "HDDMA",
    "HDDMWeighted"            : "HDDMW",
    "SEED"                    : "SEED",
}

# Rótulo de cada (cenário, configuração) na tabela de médias final.
# .A = hiperparâmetros default | .B = hiperparâmetros otimizados,
# notação alinhada à já usada na Seção V-C do artigo.
ROTULO_COL_MEDIAS = {
    ("C1", "default")  : "C1.A",
    ("C1", "otimizado"): "C1.B",
    ("C3", "default")  : "C3.A",
    ("C3", "otimizado"): "C3.B",
}


def montar_tabela_medias(df_agg: pd.DataFrame, detectores: list) -> pd.DataFrame:
    """
    Constrói uma tabela [detector x (C1.A, C1.B, C3.A, C3.B)] com a média
    de F1 (ou Acc) em percentual, calculada com o MESMO pipeline de
    agregação usado no teste estatístico (média por fold -> média entre
    folds -> média simples entre datasets do grupo filtrado por
    FILTRO_SUFIXO). Cada dataset tem peso igual.

    Entrada : df_agg (saída de agregar_por_dataset), com uma linha por
              (dataset_pasta, cenario, detector) e colunas val_default,
              val_otimizado já médias dos folds daquele dataset.
    Saída   : DataFrame indexado por detector (rótulo abreviado), com as
              4 colunas C1.A, C1.B, C3.A, C3.B em percentual (0-100).
    """
    colunas_finais = [ROTULO_COL_MEDIAS[("C1", "default")],
                       ROTULO_COL_MEDIAS[("C1", "otimizado")],
                       ROTULO_COL_MEDIAS[("C3", "default")],
                       ROTULO_COL_MEDIAS[("C3", "otimizado")]]

    tabela = pd.DataFrame(
        index=[NOME_PARA_LINHA_MEDIAS[d] for d in detectores],
        columns=colunas_finais, dtype=float,
    )

    for detector in detectores:
        rotulo_linha = NOME_PARA_LINHA_MEDIAS[detector]
        for cenario in ["C1", "C3"]:
            grupo = df_agg[
                (df_agg["detector"] == detector) & (df_agg["cenario"] == cenario)
            ]
            if grupo.empty:
                continue
            # Média simples entre datasets (cada dataset já é a média dos
            # seus próprios folds, vinda de agregar_por_dataset).
            media_default   = float(grupo["val_default"].mean())   * 100.0
            media_otimizado = float(grupo["val_otimizado"].mean()) * 100.0
            tabela.loc[rotulo_linha, ROTULO_COL_MEDIAS[(cenario, "default")]]   = media_default
            tabela.loc[rotulo_linha, ROTULO_COL_MEDIAS[(cenario, "otimizado")]] = media_otimizado

    return tabela


# ============================================================
# IMPRESSÃO FORMATADA
# ============================================================

def imprimir_tabela(df_res: pd.DataFrame, rotulo_metrica: str):
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"ANÁLISE ESTATÍSTICA (SINTÉTICO): HIPERPARÂMETROS DEFAULT vs. OTIMIZADOS — {rotulo_metrica.upper()}")
    print("Pareamento: por dataset (média dos folds) — ~20 pares independentes por detector")
    print("Referência: Demšar (2006) — unidade de observação = dataset")
    print("Teste: Wilcoxon signed-rank bilateral | Efeito: Cliff's Delta")
    print(f"Correção múltiplas comparações: Benjamini-Hochberg (α={ALPHA_FDR}) por cenário")
    print(sep)

    for cenario, grupo in df_res.groupby("cenario"):
        print(f"\n{'─'*130}")
        print(f"  CENÁRIO {cenario}")
        print(f"{'─'*130}")
        print(
            f"  {'Detector':<26} {'n_ds':>5} {'n_folds_med':>11} {'n≠0':>5} "
            f"{'média Δ(pp)':>12} {'mediana Δ(pp)':>14} "
            f"{'δ Cliff':>9} {'magnitude':<13} {'p-valor':>10} {'sig?':>5}  "
            f"{'Compensa otimizar?'}"
        )
        print(f"  {'─'*126}")
        for _, row in grupo.iterrows():
            sig_str = "✓" if row["sig_bh"] else "✗"
            print(
                f"  {row['detector']:<26} {row['n_datasets']:>5} "
                f"{row['n_folds_medio']:>11.1f} {row['n_nao_zero']:>5} "
                f"{row['media_dif_pp']:>+12.2f} {row['mediana_dif_pp']:>+14.2f} "
                f"{row['cliff_delta']:>+9.4f} {row['magnitude']:<13} "
                f"{row['p_value']:>10.4f} {sig_str:>5}   {row['recomenda_otimizar']}"
            )

    print(f"\n{sep}")
    print("LEGENDA")
    print("  n_ds        : número de datasets (pares independentes no teste)")
    print("  n_folds_med : média de folds por dataset antes da agregação")
    print("  n≠0         : datasets com diferença ≠ 0 usados no Wilcoxon")
    print("  Δ (pp)      : diferença otimizado − default em pontos percentuais")
    print("  δ Cliff     : Cliff's Delta; positivo = otimizado tende a superar default")
    print("  magnitude   : irrelevante |δ|<0.147 | pequeno <0.330 | médio <0.474 | grande ≥0.474")
    print("  sig?        : ✓ = significativo após correção BH | ✗ = não significativo")
    print("  Compensa    : SIM apenas quando sig?, magnitude≠irrelevante e ganho positivo")
    print(sep)


# ============================================================
# TABELA COLORIDA — C1 e C3 (F1 + Acc)
# ============================================================

def _ordenar_colunas_por_magnitude(df_f1_cenario, nome_para_col):
    from collections import defaultdict as _dd
    grupos = _dd(list)
    for _, row in df_f1_cenario.iterrows():
        col = nome_para_col.get(row["detector"], row["detector"])
        grupos[row["magnitude"]].append((col, row["cliff_delta"]))
    ordem = []
    for mag in ["grande", "médio", "pequeno", "irrelevante"]:
        ordem.extend([c for c, _ in sorted(grupos[mag], key=lambda x: -x[1])])
    return ordem


def _indexar(df, nome_para_col):
    out = {}
    for _, row in df.iterrows():
        col = nome_para_col.get(row["detector"], row["detector"])
        out[col] = {
            "delta": row["cliff_delta"],
            "mag"  : row["magnitude"],
            "sig"  : row["sig_bh"],
            "pval" : row["p_value"],
        }
    return out


def gerar_tabela_colorida(resultados: dict, caminho_saida: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COR_MAG = {
        "grande"     : "#e06666",
        "médio"      : "#f6b26b",
        "pequeno"    : "#6fa8dc",
        "irrelevante": "#b7b7b7",
    }

    NOME_PARA_COL = {
        "HDDMAverage"           : "HDDMA",
        "HDDMWeighted"          : "HDDMW",
        "PageHinkley"           : "PH",
        "GeometricMovingAverage": "GMA",
        "EWMAChart"             : "EWMAC",
        "KSWIN"                 : "KSWIN",
        "CUSUM"                 : "CUSUM",
        "ADWIN"                 : "ADWIN",
        "SEED"                  : "SEED",
    }

    CENARIOS   = ["C1", "C3"]
    ROT_LINHAS = ["F1", "Acc"]
    n_rows     = len(ROT_LINHAS)
    n_cenarios = len(CENARIOS)

    cell_w   = 1.0
    cell_h   = 1.0   # reduzida: célula agora tem só uma linha (delta, sem p-valor)
    gap_v    = 1.2
    font_val = 14
    font_hdr = 13
    font_row = 14
    font_tit = 14

    ordens = {
        cen: _ordenar_colunas_por_magnitude(resultados[cen]["f1"], NOME_PARA_COL)
        for cen in CENARIOS
    }
    n_cols_max = max(len(ordens[cen]) for cen in CENARIOS)

    fig_w  = cell_w * n_cols_max + 2.8
    fig_h  = (n_rows * n_cenarios * cell_h
              + (n_cenarios - 1) * gap_v
              + 3.2)
    total_h = fig_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols_max)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    y_base = {}
    y_base["C1"] = total_h - 1.5 - n_rows * cell_h
    y_base["C3"] = y_base["C1"] - gap_v - n_rows * cell_h

    for cen in CENARIOS:
        ordem_cols = ordens[cen]
        yb         = y_base[cen]

        idx_f1  = _indexar(resultados[cen]["f1"],  NOME_PARA_COL)
        idx_acc = _indexar(resultados[cen]["acc"], NOME_PARA_COL)
        indices_por_linha = [idx_f1, idx_acc]

        ax.text(-0.08, yb + n_rows * cell_h + 0.70, cen,
                ha="right", va="center",
                fontsize=font_tit + 1, fontweight="bold")

        for c, col in enumerate(ordem_cols):
            ax.text(c * cell_w + cell_w / 2,
                    yb + n_rows * cell_h + 0.14, col,
                    ha="center", va="bottom",
                    fontsize=font_hdr, fontweight="bold")

        for r, rot in enumerate(ROT_LINHAS):
            ax.text(-0.08,
                    yb + (n_rows - 1 - r) * cell_h + cell_h / 2, rot,
                    ha="right", va="center",
                    fontsize=font_row, fontweight="bold")

        for r, idx in enumerate(indices_por_linha):
            for c, col in enumerate(ordem_cols):
                d = idx.get(col, {"delta": np.nan, "mag": "irrelevante",
                                  "sig": False, "pval": np.nan})
                if np.isnan(d["delta"]):
                    val_str = "–"
                elif d["sig"]:
                    val_str = f"{d['delta']:+.2f}*"
                else:
                    val_str = f"{d['delta']:+.2f}"
                cor = COR_MAG[d["mag"]]
                x = c * cell_w
                y = yb + (n_rows - 1 - r) * cell_h
                ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h,
                                           facecolor=cor, edgecolor="white",
                                           linewidth=1.5))
                ax.text(x + cell_w / 2, y + cell_h / 2, val_str,
                        ha="center", va="center",
                        fontsize=font_val, fontweight="bold", color="white",
                        multialignment="center")

    patches = [
        mpatches.Patch(color=COR_MAG["grande"],      label="Large"),
        mpatches.Patch(color=COR_MAG["médio"],       label="Medium"),
        mpatches.Patch(color=COR_MAG["pequeno"],     label="Small"),
        mpatches.Patch(color=COR_MAG["irrelevante"], label="Negligible"),
    ]
    y_leg = y_base["C3"] - 0.12
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(0.0, y_leg), bbox_transform=ax.transData,
              ncol=4, fontsize=font_val, frameon=True,
              title=r"Cliff's $\delta$ magnitude  (* $p$ < 0.05 BH)",
              title_fontsize=font_val)

    plt.tight_layout(rect=[0.07, 0.03, 1.0, 1.0])
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura salva em: {caminho_saida}")


# ============================================================
# MAIN
# ============================================================

def main():
    pastas_saida = descobrir_pastas_saida(PASTA_RAIZ, FILTRO_SUFIXO, SUBPASTA_EXPERIMENTO)

    resultados_por_metrica = {}

    for arquivo, rotulo_metrica, sufixo in METRICAS:

        print(f"\n{'#'*60}")
        print(f"  MÉTRICA: {rotulo_metrica}  ({arquivo})")
        print(f"{'#'*60}")

        print(f"\nColetando pares (fold-level)...", flush=True)
        df_pares = coletar_pares(pastas_saida, DETECTORES, PARES_CENARIO, arquivo)
        print(f"  {len(df_pares)} pares fold-level | "
              f"{df_pares['dataset_pasta'].nunique()} datasets")

        print(f"Agregando por dataset...", flush=True)
        df_agg = agregar_por_dataset(df_pares)
        print(f"  {len(df_agg)} pares dataset-level após agregação")

        print(f"Aplicando testes estatísticos...", flush=True)
        df_res = analisar_agregado(df_agg)

        imprimir_tabela(df_res, rotulo_metrica)

        out_res = os.path.join(
            PASTA_RAIZ, f"analise_default_vs_otimizado_agregado_sintetico_{sufixo}.csv"
        )
        df_res.to_csv(out_res, index=False)
        print(f"\nResultados salvos em: {out_res}")

        # --- Tabela de médias C1.A/C1.B/C3.A/C3.B (detector x configuração) ---
        tabela_medias = montar_tabela_medias(df_agg, DETECTORES)
        print(f"\nTabela de médias ({rotulo_metrica}, "
              f"filtro_sufixo='{FILTRO_SUFIXO or '(nenhum)'}'):")
        print(tabela_medias.round(2).to_string())

        sufixo_filtro = FILTRO_SUFIXO.strip("_").lower() if FILTRO_SUFIXO else "todos"
        out_medias = os.path.join(
            PASTA_RAIZ,
            f"tabela_medias_default_otimizado_sintetico_{sufixo_filtro}_{sufixo}.csv",
        )
        tabela_medias.round(4).to_csv(out_medias, index=True, index_label="detector")
        print(f"Tabela de médias salva em: {out_medias}")

        resultados_por_metrica[sufixo] = {}
        for cen, grupo in df_res.groupby("cenario"):
            resultados_por_metrica[sufixo][cen] = grupo.reset_index(drop=True)

    # Gera figura
    if "f1" in resultados_por_metrica and "acc" in resultados_por_metrica:
        cenarios_disponiveis = (
            set(resultados_por_metrica["f1"].keys()) &
            set(resultados_por_metrica["acc"].keys())
        )
        if {"C1", "C3"}.issubset(cenarios_disponiveis):
            resultados_figura = {
                cen: {
                    "f1" : resultados_por_metrica["f1"][cen],
                    "acc": resultados_por_metrica["acc"][cen],
                }
                for cen in ["C1", "C3"]
            }
            gerar_tabela_colorida(resultados_figura, caminho_saida=FIGURA_SAIDA)
        else:
            print(f"\n⚠  Cenários disponíveis: {cenarios_disponiveis}. "
                  "Figura não gerada (requer C1 e C3).")


if __name__ == "__main__":
    main()