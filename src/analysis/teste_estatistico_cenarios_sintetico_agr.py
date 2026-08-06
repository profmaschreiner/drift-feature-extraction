"""
teste_estatistico_cenarios_sintetico_agregado.py
==================================================
Versão alternativa do teste estatístico que agrega os resultados
POR DATASET antes de aplicar o Wilcoxon, em vez de usar pares
dataset × fold diretamente — aplicada aos conjuntos de dados sintéticos
(gerados via CaDrift).

Motivação
---------
No script original, o Wilcoxon é pareado por (dataset × fold), resultando
em muitos pares por detector. Contudo, folds dentro do mesmo dataset não são
estritamente independentes (os conjuntos de treino se sobrepõem). Esta versão
agrega as métricas por dataset (média dos folds) antes do teste, resultando
em pares verdadeiramente independentes (um por dataset sintético).

Comparação entre as duas abordagens:
  - Original  : muitos pares | folds dentro de um dataset NÃO são independentes
  - Agregada  : poucos pares | datasets são genuinamente independentes

Esta versão serve para verificar se as conclusões do paper se mantêm sob
a abordagem mais conservadora. Sendo os efeitos de magnitude grande,
espera-se convergência entre as duas abordagens.

Filtro opcional
---------------
A flag FILTRO_SUFIXO permite restringir a análise apenas aos datasets
sintéticos cujo nome de pasta TERMINA com um dado sufixo (ex.: "_D_").
Se FILTRO_SUFIXO = "", todos os datasets encontrados são processados.

Saídas
------
* Console : tabela comparando original vs. agregado por comparação × detector
* CSV     : analise_cenarios_agregado_sintetico_{metrica}.csv
* PNG     : tabela_comparacao_cenarios_sintetico_agregado.png  (mesma estrutura visual)

Configuração: ajuste PASTA_RAIZ e, se necessário, FILTRO_SUFIXO.
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

COMPARACOES = [
    ("baseline",   "drift",         "C0_vs_C1", "C1 supera C0? (drift agrega valor ao baseline?)"),
    ("baseline",   "catch24",       "C0_vs_C2", "C2 supera C0? (catch24 agrega valor ao baseline?)"),
    ("drift",      "catch24",       "C1_vs_C2", "C2 supera C1? (catch24 vs metodologia proposta)"),
    ("catch24",    "catch24_drift", "C2_vs_C3", "C3 supera C2? (drift complementa o catch24?)"),
]

COMPARACOES_FIGURA = ["C0_vs_C1", "C1_vs_C2", "C2_vs_C3"]

METRICAS = [
    ("df_f1_all.csv",  "F1-score macro", "f1"),
    ("df_acc_all.csv", "Acurácia",       "acc"),
]

COLUNA_METRICA         = "RF"
ALPHA_FDR              = 0.05
CENARIOS_SEM_DETECTOR  = {"baseline", "catch24"}

# Filtro por sufixo do nome da pasta do dataset. Se "" (vazio), processa
# todas as pastas encontradas. Se preenchido (ex.: "_D_"), processa apenas
# as pastas cujo nome TERMINA com esse sufixo.
FILTRO_SUFIXO = "_D_"

# Subpasta fixa intermediária presente em todos os datasets sintéticos,
# entre a pasta do dataset e as pastas de cenário (baseline, drift,
# catch24, catch24_drift, etc.). Ex.: estrutura real é
# <dataset>/resultados_catch24_drift/<cenario>/<detector>/df_f1_all.csv
SUBPASTA_EXPERIMENTO = "resultados_catch24_drift"

FIGURA_SAIDA = os.path.join(PASTA_RAIZ, "tabela_comparacao_cenarios_sintetico_agregado.png")

# ============================================================
# DESCOBERTA DE PASTAS
# ============================================================

def descobrir_pastas_datasets(pasta_raiz: str, filtro_sufixo: str = "",
                              subpasta_experimento: str = "") -> list:
    raiz = Path(pasta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Pasta raiz não encontrada: {pasta_raiz}\n"
            "Ajuste a variável PASTA_RAIZ no início do script."
        )
    cenarios_validos = {c[0] for c in COMPARACOES} | {c[1] for c in COMPARACOES}

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
        if base.is_dir() and any((base / c).is_dir() for c in cenarios_validos):
            pastas.append(str(base))

    if not pastas:
        if filtro_sufixo:
            raise RuntimeError(
                f"Nenhuma pasta com nome terminando em '{filtro_sufixo}' "
                f"encontrada em '{pasta_raiz}' contendo a subpasta "
                f"'{subpasta_experimento}' com algum de: {sorted(cenarios_validos)}."
            )
        raise RuntimeError(
            f"Nenhuma subpasta válida encontrada em '{pasta_raiz}'.\n"
            f"Esperado ao menos um de: {sorted(cenarios_validos)} "
            f"dentro de '{subpasta_experimento}'."
        )
    print(f"Datasets sintéticos encontrados ({len(pastas)}): "
          f"{[Path(p).parent.name for p in pastas]}")
    return pastas


# ============================================================
# CARREGAMENTO FLEXÍVEL DE CSV
# ============================================================

def load_csv_flexivel(pasta_dataset: str, cenario: str, detector: str,
                      arquivo: str) -> pd.DataFrame:
    candidatos = [
        os.path.join(pasta_dataset, cenario, detector, arquivo),
        os.path.join(pasta_dataset, cenario, arquivo),
    ]
    for path in candidatos:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols_needed = {"run", "fold", COLUNA_METRICA}
        if not cols_needed.issubset(df.columns):
            warnings.warn(f"Arquivo sem colunas esperadas {cols_needed}: {path}")
            return pd.DataFrame()
        return df[["run", "fold", COLUNA_METRICA]].copy()
    return pd.DataFrame()


# ============================================================
# FUNÇÕES ESTATÍSTICAS
# ============================================================

def cliff_delta_paired(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    d = arr_b - arr_a
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
    ordem      = np.argsort(p_values)
    p_sorted   = np.array(p_values)[ordem]
    threshold  = (np.arange(1, n + 1) / n) * alpha
    rej_sorted = p_sorted <= threshold
    rej_sorted = np.maximum.accumulate(rej_sorted[::-1])[::-1]
    rejeita    = np.empty(n, dtype=bool)
    rejeita[ordem] = rej_sorted
    return rejeita.tolist()


# ============================================================
# COLETA DE PARES (igual ao original)
# ============================================================

def coletar_pares(pastas_datasets: list, detectores: list,
                  comparacoes: list, arquivo: str) -> pd.DataFrame:
    rows = []

    for pasta, (cen_a, cen_b, rotulo, _) in iproduct(pastas_datasets, comparacoes):
        dataset_pasta = Path(pasta).parent.name if SUBPASTA_EXPERIMENTO else Path(pasta).name

        # Se ambos os cenários são independentes de detector (ex: C0 vs C2),
        # os pares são coletados uma única vez com detector="(none)" para
        # evitar duplicação dos mesmos pares para cada um dos 9 detectores.
        sem_detector_a = cen_a in CENARIOS_SEM_DETECTOR
        sem_detector_b = cen_b in CENARIOS_SEM_DETECTOR
        detectores_iter = ["(none)"] if (sem_detector_a and sem_detector_b) \
                          else detectores

        for detector in detectores_iter:
            df_a = load_csv_flexivel(pasta, cen_a, detector, arquivo)
            df_b = load_csv_flexivel(pasta, cen_b, detector, arquivo)

            if df_a.empty or df_b.empty:
                continue

            df_a = df_a.rename(columns={"run": "run_A", COLUNA_METRICA: "val_A"})
            df_b = df_b.rename(columns={"run": "run_B", COLUNA_METRICA: "val_B"})

            df_a["dataset_pasta"] = dataset_pasta
            df_b["dataset_pasta"] = dataset_pasta

            merged = pd.merge(df_a, df_b, on=["dataset_pasta", "fold"], how="inner")

            if merged.empty:
                warnings.warn(
                    f"Nenhum fold em comum: dataset={dataset_pasta}, "
                    f"comp={rotulo}, detector={detector}"
                )
                continue

            merged["comparacao"] = rotulo
            merged["detector"]   = detector
            merged["pasta"]      = pasta

            rows.append(merged[[
                "dataset_pasta", "pasta", "comparacao", "detector",
                "fold", "run_A", "run_B", "val_A", "val_B",
            ]])

    if not rows:
        raise RuntimeError(
            f"Nenhum par encontrado para arquivo={arquivo}. "
            "Verifique PASTA_RAIZ e a estrutura de pastas."
        )

    return pd.concat(rows, ignore_index=True)


# ============================================================
# AGREGAÇÃO POR DATASET  ← núcleo desta versão
# ============================================================

def agregar_por_dataset(df_pares: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os pares por dataset, calculando a média de val_A e val_B
    sobre todos os folds de cada dataset.

    Entrada : df_pares com colunas dataset_pasta, comparacao, detector,
              fold, val_A, val_B
    Saída   : df com uma linha por (dataset_pasta, comparacao, detector),
              com val_A e val_B sendo a média dos folds
    """
    agg = (
        df_pares
        .groupby(["dataset_pasta", "comparacao", "detector"], as_index=False)
        .agg(
            val_A    = ("val_A", "mean"),
            val_B    = ("val_B", "mean"),
            n_folds  = ("fold",  "count"),
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
    """
    resultados = []

    for (comparacao, detector), grupo in df_agg.groupby(["comparacao", "detector"]):
        arr_a       = grupo["val_A"].to_numpy(dtype=float)
        arr_b       = grupo["val_B"].to_numpy(dtype=float)
        dif         = arr_b - arr_a

        n_datasets  = len(dif)
        n_nao_zero  = int(np.sum(dif != 0))
        media_dif   = float(np.mean(dif))
        mediana_dif = float(np.median(dif))
        delta       = cliff_delta_paired(arr_a, arr_b)
        mag         = magnitude_cliff(delta)

        if n_nao_zero < 1:
            p_value = 1.0
            stat    = np.nan
        else:
            try:
                stat, p_value = wilcoxon(dif[dif != 0], alternative="two-sided")
            except Exception as e:
                warnings.warn(f"Wilcoxon falhou ({comparacao}, {detector}): {e}")
                stat, p_value = np.nan, 1.0

        resultados.append({
            "comparacao"    : comparacao,
            "detector"      : detector,
            "n_datasets"    : n_datasets,
            "n_folds_medio" : round(grupo["n_folds"].mean(), 1),
            "n_nao_zero"    : n_nao_zero,
            "media_dif_pp"  : round(media_dif   * 100, 2),
            "mediana_dif_pp": round(mediana_dif  * 100, 2),
            "cliff_delta"   : round(delta, 4),
            "magnitude"     : mag,
            "wilcoxon_stat" : round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value"       : round(p_value, 6),
        })

    df_res = pd.DataFrame(resultados)

    # Correção BH por comparação
    df_res["sig_bh"] = False
    for comparacao, grupo in df_res.groupby("comparacao"):
        idx     = grupo.index.tolist()
        p_vals  = grupo["p_value"].tolist()
        rejeita = bh_correction(p_vals, alpha=ALPHA_FDR)
        for i, rej in zip(idx, rejeita):
            df_res.at[i, "sig_bh"] = bool(rej)

    def interpretar(row):
        if row["sig_bh"] and row["magnitude"] != "irrelevante":
            return f"B > A ({row['magnitude']})" if row["media_dif_pp"] > 0 else f"A > B ({row['magnitude']})"
        if row["sig_bh"] and row["magnitude"] == "irrelevante":
            return "sig. mas irrelevante"
        return "não significativo"

    df_res["resultado"] = df_res.apply(interpretar, axis=1)

    ordem_comp = {c[2]: i for i, c in enumerate(COMPARACOES)}
    df_res["_ordem"] = df_res["comparacao"].map(ordem_comp)
    df_res = df_res.sort_values(["_ordem", "detector"]).drop(columns="_ordem")
    return df_res.reset_index(drop=True)


# ============================================================
# IMPRESSÃO COMPARATIVA
# ============================================================

def imprimir_comparacao(df_orig: pd.DataFrame, df_agg: pd.DataFrame,
                        rotulo_metrica: str):
    """
    Imprime tabela lado a lado: resultado original (fold-level)
    vs. resultado agregado (dataset-level), destacando divergências.
    """
    hipoteses = {c[2]: c[3] for c in COMPARACOES}
    sep = "=" * 140
    print(f"\n{sep}")
    print(f"COMPARAÇÃO (SINTÉTICO): PARES fold-level (original) vs. dataset-level (agregado) — {rotulo_metrica.upper()}")
    print(f"Método original : Wilcoxon pareado por dataset × fold")
    print(f"Método agregado : Wilcoxon pareado por dataset (independentes)")
    print(f"Correção BH aplicada sobre os 9 detectores dentro de cada comparação (α={ALPHA_FDR})")
    print(sep)

    df_merged = pd.merge(
        df_orig[["comparacao", "detector", "n_pares", "cliff_delta", "magnitude", "sig_bh", "p_value"]],
        df_agg [["comparacao", "detector", "n_datasets", "cliff_delta", "magnitude", "sig_bh", "p_value"]],
        on=["comparacao", "detector"],
        suffixes=("_orig", "_agg"),
    )

    ordem_comp = {c[2]: i for i, c in enumerate(COMPARACOES)}
    df_merged["_ordem"] = df_merged["comparacao"].map(ordem_comp)
    df_merged = df_merged.sort_values(["_ordem", "detector"]).drop(columns="_ordem")

    for comparacao, grupo in df_merged.groupby("comparacao", sort=False):
        hip = hipoteses.get(comparacao, "")
        print(f"\n{'─'*140}")
        print(f"  COMPARAÇÃO: {comparacao}  —  {hip}")
        print(f"{'─'*140}")
        print(
            f"  {'Detector':<26} "
            f"{'── ORIGINAL (fold-level) ──':^42}  "
            f"{'── AGREGADO (dataset-level) ──':^42}  "
            f"{'Diverge?'}"
        )
        print(
            f"  {'':26} "
            f"{'n_pares':>7} {'δ Cliff':>9} {'magnitude':<13} {'sig?':>5}  "
            f"{'n_ds':>5} {'δ Cliff':>9} {'magnitude':<13} {'sig?':>5}  "
        )
        print(f"  {'─'*136}")
        for _, row in grupo.iterrows():
            sig_o = "✓" if row["sig_bh_orig"] else "✗"
            sig_a = "✓" if row["sig_bh_agg"]  else "✗"

            # Divergência: magnitude ou significância diferem
            diverge = (row["magnitude_orig"] != row["magnitude_agg"]) or \
                      (row["sig_bh_orig"]    != row["sig_bh_agg"])
            flag = "  ⚠ DIVERGE" if diverge else ""

            print(
                f"  {row['detector']:<26} "
                f"{row['n_pares']:>7} {row['cliff_delta_orig']:>+9.4f} "
                f"{row['magnitude_orig']:<13} {sig_o:>5}  "
                f"{row['n_datasets']:>5} {row['cliff_delta_agg']:>+9.4f} "
                f"{row['magnitude_agg']:<13} {sig_a:>5}  "
                f"{flag}"
            )

    print(f"\n{sep}")
    print("LEGENDA")
    print("  δ Cliff     : Cliff's Delta; positivo = B tende a superar A")
    print("  magnitude   : irrelevante |δ|<0.147 | pequeno <0.330 | médio <0.474 | grande ≥0.474")
    print("  sig?        : ✓ = significativo após correção BH (α=0.05) | ✗ = não significativo")
    print("  ⚠ DIVERGE  : magnitude ou significância diferem entre as duas abordagens")
    print(sep)


# ============================================================
# FIGURA (mesma estrutura visual do script original)
# ============================================================

def _ordenar_colunas(df_f1_comp, nome_para_col):
    from collections import defaultdict as _dd
    grupos = _dd(list)
    for _, row in df_f1_comp.iterrows():
        col = nome_para_col.get(row["detector"], row["detector"])
        grupos[row["magnitude"]].append((col, row["cliff_delta"]))
    ordem = []
    for mag in ["grande", "médio", "pequeno", "irrelevante"]:
        ordem.extend([c for c, _ in sorted(grupos[mag], key=lambda x: -x[1])])
    return ordem


def _indexar(df, nome_para_col):
    return {
        nome_para_col.get(r["detector"], r["detector"]): {
            "delta": r["cliff_delta"],
            "mag"  : r["magnitude"],
            "sig"  : r["sig_bh"],
            "pval" : r["p_value"],
        }
        for _, r in df.iterrows()
    }


def gerar_figura(resultados, caminho_saida):
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

    ROTULO_COMP = {
        "C0_vs_C1": "C0 vs C1",
        "C0_vs_C2": "C0 vs C2",
        "C1_vs_C2": "C1 vs C2",
        "C2_vs_C3": "C2 vs C3",
    }

    COMPS = [c for c in COMPARACOES_FIGURA if c in resultados]
    ROT_LINHAS = ["F1", "Acc"]
    n_rows = len(ROT_LINHAS)
    n_comp = len(COMPS)

    cell_w = 1.0; cell_h = 1.2; gap_v = 1.0; hdr_h = 1.0
    font_val = 14; font_hdr = 12; font_row = 14; font_tit = 13

    ordens = {
        comp: _ordenar_colunas(resultados[comp]["f1"], NOME_PARA_COL)
        for comp in COMPS
    }
    n_cols_max = max(len(ordens[c]) for c in COMPS)

    fig_w  = cell_w * n_cols_max + 3.0
    fig_h  = n_rows * n_comp * cell_h + (n_comp - 1) * gap_v + n_comp * hdr_h
    total_h = fig_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols_max)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    y_bases = {}
    cursor = total_h - hdr_h
    for comp in COMPS:
        y_bases[comp] = cursor - n_rows * cell_h
        cursor = y_bases[comp] - gap_v - hdr_h

    for comp in COMPS:
        oc  = ordens[comp]
        yb  = y_bases[comp]
        n_c = len(oc)

        idx_f1  = _indexar(resultados[comp]["f1"],  NOME_PARA_COL)
        idx_acc = _indexar(resultados[comp]["acc"], NOME_PARA_COL)
        indices_por_linha = [idx_f1, idx_acc]

        ax.text(-0.08, yb + n_rows * cell_h + 0.68,
                ROTULO_COMP.get(comp, comp),
                ha="right", va="center",
                fontsize=font_tit + 1, fontweight="bold")

        for c, col in enumerate(oc):
            ax.text(c * cell_w + cell_w / 2, yb + n_rows * cell_h + 0.14, col,
                    ha="center", va="bottom",
                    fontsize=font_hdr, fontweight="bold")

        for r, rot in enumerate(ROT_LINHAS):
            ax.text(-0.08, yb + (n_rows - 1 - r) * cell_h + cell_h / 2, rot,
                    ha="right", va="center",
                    fontsize=font_row, fontweight="bold")

        for r, idx in enumerate(indices_por_linha):
            for c, col in enumerate(oc):
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
    y_leg = y_bases[COMPS[-1]] - 0.12
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(0.0, y_leg), bbox_transform=ax.transData,
              ncol=4, fontsize=font_val, frameon=True,
              title=r"Cliff's $\delta$ magnitude  (* $p$ < 0.05 BH)",
              title_fontsize=font_val)

    # Nota no rodapé informando que é a versão agregada
    #fig.text(0.5, 0.01, "Pairs aggregated by dataset (mean over folds) — ~15 independent pairs per detector",
    #         ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout(rect=[0.10, 0.03, 1.0, 1.0])
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura (agregada, sintético) salva em: {caminho_saida}")


# ============================================================
# MAIN
# ============================================================

def main():
    pastas_datasets = descobrir_pastas_datasets(PASTA_RAIZ, FILTRO_SUFIXO, SUBPASTA_EXPERIMENTO)

    resultados_orig = {comp: {} for comp in [c[2] for c in COMPARACOES]}
    resultados_agg  = {comp: {} for comp in [c[2] for c in COMPARACOES]}

    # Importa a função analisar do script original para comparação
    # Se preferir rodar isolado, a função analisar() abaixo é uma cópia funcional
    def analisar_original(df_pares):
        """Versão original: Wilcoxon sobre pares dataset × fold."""
        resultados = []
        for (comparacao, detector), grupo in df_pares.groupby(["comparacao", "detector"]):
            arr_a = grupo["val_A"].to_numpy(dtype=float)
            arr_b = grupo["val_B"].to_numpy(dtype=float)
            dif   = arr_b - arr_a
            n_pares    = len(dif)
            n_datasets = grupo["dataset_pasta"].nunique()
            n_nao_zero = int(np.sum(dif != 0))
            media_dif  = float(np.mean(dif))
            mediana_dif= float(np.median(dif))
            delta = cliff_delta_paired(arr_a, arr_b)
            mag   = magnitude_cliff(delta)
            if n_nao_zero < 1:
                p_value = 1.0; stat = np.nan
            else:
                try:
                    stat, p_value = wilcoxon(dif[dif != 0], alternative="two-sided")
                except Exception as e:
                    warnings.warn(f"Wilcoxon falhou: {e}")
                    stat, p_value = np.nan, 1.0
            resultados.append({
                "comparacao": comparacao, "detector": detector,
                "n_datasets": n_datasets, "n_pares": n_pares,
                "n_nao_zero": n_nao_zero,
                "media_dif_pp": round(media_dif*100, 2),
                "mediana_dif_pp": round(mediana_dif*100, 2),
                "cliff_delta": round(delta, 4), "magnitude": mag,
                "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
                "p_value": round(p_value, 6),
            })
        df_res = pd.DataFrame(resultados)
        df_res["sig_bh"] = False
        for comparacao, grupo in df_res.groupby("comparacao"):
            idx = grupo.index.tolist()
            rejeita = bh_correction(grupo["p_value"].tolist(), alpha=ALPHA_FDR)
            for i, rej in zip(idx, rejeita):
                df_res.at[i, "sig_bh"] = bool(rej)
        ordem_comp = {c[2]: i for i, c in enumerate(COMPARACOES)}
        df_res["_ordem"] = df_res["comparacao"].map(ordem_comp)
        df_res = df_res.sort_values(["_ordem", "detector"]).drop(columns="_ordem")
        return df_res.reset_index(drop=True)

    for arquivo, rotulo_metrica, sufixo in METRICAS:

        print(f"\n{'#'*70}")
        print(f"  MÉTRICA: {rotulo_metrica}  ({arquivo})")
        print(f"{'#'*70}")

        print(f"\nColetando pares — {rotulo_metrica}...", flush=True)
        df_pares = coletar_pares(pastas_datasets, DETECTORES, COMPARACOES, arquivo)

        print(f"  {len(df_pares)} pares fold-level | "
              f"{df_pares['dataset_pasta'].nunique()} datasets")

        # Análise original (fold-level)
        df_res_orig = analisar_original(df_pares)

        # Agregação por dataset e análise
        df_agg = agregar_por_dataset(df_pares)
        print(f"  {len(df_agg)} pares dataset-level após agregação")
        df_res_agg = analisar_agregado(df_agg)

        # Impressão comparativa
        imprimir_comparacao(df_res_orig, df_res_agg, rotulo_metrica)

        # Persiste CSVs
        out_orig = os.path.join(PASTA_RAIZ, f"analise_cenarios_sintetico_{sufixo}.csv")
        out_agg  = os.path.join(PASTA_RAIZ, f"analise_cenarios_agregado_sintetico_{sufixo}.csv")
        df_res_orig.to_csv(out_orig, index=False)
        df_res_agg.to_csv(out_agg,  index=False)
        print(f"\nResultados originais salvos em: {out_orig}")
        print(f"Resultados agregados salvos em: {out_agg}")

        # Acumula para figura
        for comp, grupo in df_res_agg.groupby("comparacao"):
            resultados_agg[comp][sufixo] = grupo.reset_index(drop=True)

    # Gera figura com resultados agregados
    comps_disponiveis = [
        comp for comp in COMPARACOES_FIGURA
        if "f1" in resultados_agg.get(comp, {})
        and "acc" in resultados_agg.get(comp, {})
    ]
    if len(comps_disponiveis) == len(COMPARACOES_FIGURA):
        gerar_figura(resultados_agg, caminho_saida=FIGURA_SAIDA)
    else:
        faltando = [c for c in COMPARACOES_FIGURA if c not in comps_disponiveis]
        print(f"\n⚠  Figura não gerada. Comparações sem dados: {faltando}")


if __name__ == "__main__":
    main()