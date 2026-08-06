"""
test_estat_ot_default_fonte.py
===============================
Versão do teste default vs. otimizado agregada por FONTE INDEPENDENTE,
corrigindo o mesmo problema de pseudorreplicação (Hurlbert, 1984,
Ecological Monographs 54(2):187-211) identificado no teste de cenários.

Motivação
---------
O script `test_estat_ot_default_agr.py` já corrige o pareamento por fold
(agregando fold -> dataset antes do teste, seguindo Demšar 2006), mas
ainda pareia por split de sensor (~16 "datasets"). Oito desses 16 splits
vêm de apenas duas fontes reais, PAMAP2 (P2-Hand/Chest/Ankle) e WARD
(W-LWrist/RWrist/Waist/LAnkle/RAnkle): mesmos sujeitos, mesmas sessões
de coleta, mesma estrutura de folds, diferindo só no canal de sensor
extraído. Tratar esses 8 splits como observações independentes infla o
n efetivo do Wilcoxon pareado e, com isso, a significância aparente.

Este script adiciona uma segunda agregação, split de sensor -> fonte,
antes do teste, reduzindo o pareamento de ~16 splits para as 10 fontes
reais verdadeiramente independentes (Tabela II do artigo). O critério
de agregação entre splits de uma mesma fonte é controlado por
MEIO_AGREGACAO, com a MESMA semântica usada em
`teste_estatistico_cenarios_agregado_fonte.py`:

    "AVG" : média de val_default e val_otimizado entre os splits da
            mesma fonte (critério recomendado).
    "MAX" : maior valor de val_default e de val_otimizado entre os
            splits, calculado independentemente por coluna.
    "MIN" : menor valor de val_default e de val_otimizado entre os
            splits, calculado independentemente por coluna.

Fontes de split único (DR, Gait, RO, MH, SmPh, Sp, Sw, USC) não são
afetadas pela escolha de MEIO_AGREGACAO.

IMPORTANTE — antes de confiar nos resultados
---------------------------------------------
Ajuste GRUPOS_FONTE para refletir os nomes reais das pastas em
PASTA_RAIZ (o script imprime o mapeamento inferido dataset -> fonte
no início da execução; confira essa tabela antes de interpretar o
p-valor). Mantenha GRUPOS_FONTE e MEIO_AGREGACAO consistentes com o
script `teste_estatistico_cenarios_agregado_fonte.py`.

Saídas
------
* Console : mapeamento dataset -> fonte, e tabela comparando split de
            sensor (n~16, pseudorreplicado) vs. fonte (n=10,
            independente) por cenário x detector, sinalizando
            divergências.
* CSV     : analise_default_vs_otimizado_fonte_{metrica}.csv (resultado
            primário, n = fontes independentes)
            analise_default_vs_otimizado_split_{metrica}.csv
            (diagnóstico, n = splits, mantido para referência)
* PNG     : tabela_otimizacao_reais_fonte_{meio}.png

Configuração: ajuste PASTA_RAIZ, GRUPOS_FONTE e MEIO_AGREGACAO.
"""

import os
import warnings
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# CONFIGURAÇÃO — ajuste esta seção
# ============================================================

PASTA_RAIZ = "exp_otimizacao/result_reais_completo"

# Critério de agregação dos splits de sensor de uma mesma fonte,
# aplicado independentemente a val_default e a val_otimizado.
#   "AVG" (recomendado) | "MAX" | "MIN"
# Mantenha consistente com teste_estatistico_cenarios_agregado_fonte.py
MEIO_AGREGACAO = "AVG"

_MEIOS_VALIDOS = {"AVG", "MAX", "MIN"}
if MEIO_AGREGACAO not in _MEIOS_VALIDOS:
    raise ValueError(
        f"MEIO_AGREGACAO deve ser um de {sorted(_MEIOS_VALIDOS)}, "
        f"recebido: {MEIO_AGREGACAO!r}"
    )

_FUNC_AGREGACAO = {
    "AVG": lambda arr: float(np.mean(arr)),
    "MAX": lambda arr: float(np.max(arr)),
    "MIN": lambda arr: float(np.min(arr)),
}

# Mapeamento dataset (nome da pasta) -> fonte independente.
# Datasets não listados aqui são tratados como fonte de split único
# (o próprio nome da pasta é a fonte). AJUSTE os nomes de pasta abaixo
# para bater exatamente com os nomes reais em PASTA_RAIZ.
GRUPOS_FONTE = {
    "P2": ["pamap2_hand", "pamap2_chest", "pamap2_ankle"],
    "W":  ["ward_left_wrist", "ward_right_wrist", "ward_waist", "ward_left_ankle", "ward_right_ankle"],
    "sp_sw":  ["sp_har", "sw_har"],    
}

# Prefixos usados apenas para alertar sobre possíveis splits não
# mapeados (heurística de checagem, não afeta o cálculo).
_PREFIXOS_ALERTA = ("P2-", "W-", "PAMAP2", "WARD")

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

FIGURA_SAIDA = os.path.join(
    PASTA_RAIZ, f"tabela_otimizacao_reais_fonte_{MEIO_AGREGACAO.lower()}.png"
)

# ============================================================
# DESCOBERTA DE PASTAS
# ============================================================

def descobrir_pastas_saida(pasta_raiz: str) -> list:
    raiz = Path(pasta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Pasta raiz não encontrada: {pasta_raiz}\n"
            "Ajuste a variável PASTA_RAIZ no início do script."
        )
    pastas = sorted([
        str(p) for p in raiz.iterdir()
        if p.is_dir() and (
            (p / "drift").is_dir() or (p / "drift_ot").is_dir()
        )
    ])
    if not pastas:
        raise RuntimeError(
            f"Nenhuma subpasta válida encontrada em '{pasta_raiz}'.\n"
            "Verifique se as pastas contêm os subdiretórios drift/ ou drift_ot/."
        )
    print(f"Datasets encontrados ({len(pastas)}): {[Path(p).name for p in pastas]}")
    return pastas


# ============================================================
# MAPEAMENTO DATASET -> FONTE
# ============================================================

def mapear_fonte(dataset_pasta: str) -> str:
    """Retorna a fonte independente à qual um split de sensor pertence.

    Datasets não listados em GRUPOS_FONTE são considerados fontes de
    split único: a fonte é o próprio nome do dataset.
    """
    for fonte, splits in GRUPOS_FONTE.items():
        if dataset_pasta in splits:
            return fonte
    return dataset_pasta


def imprimir_mapeamento_fonte(pastas_saida: list) -> None:
    print("\n" + "=" * 70)
    print("MAPEAMENTO DATASET (SPLIT DE SENSOR) -> FONTE INDEPENDENTE")
    print("Confira esta tabela contra a estrutura real de pastas ANTES de")
    print("confiar nos p-valores calculados a seguir.")
    print("=" * 70)

    fontes = {}
    for p in pastas_saida:
        nome = Path(p).name
        fonte = mapear_fonte(nome)
        fontes.setdefault(fonte, []).append(nome)

    todos_splits_mapeados = {s for splits in GRUPOS_FONTE.values() for s in splits}
    for fonte, splits in sorted(fontes.items()):
        marcador = " (múltiplos splits agregados)" if len(splits) > 1 else ""
        print(f"  {fonte:<10} <- {splits}{marcador}")

        for nome in splits:
            if nome not in todos_splits_mapeados and nome.startswith(_PREFIXOS_ALERTA):
                warnings.warn(
                    f"'{nome}' parece ser um split de sensor não mapeado "
                    f"em GRUPOS_FONTE (foi tratado como fonte própria). "
                    f"Verifique se isso está correto."
                )

    print(f"\nTotal de datasets (splits): {len(pastas_saida)}")
    print(f"Total de fontes independentes: {len(fontes)}")
    print(f"Critério de agregação entre splits: MEIO_AGREGACAO = {MEIO_AGREGACAO}")
    print("=" * 70)


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
    ordem      = np.argsort(p_values)
    p_sorted   = np.array(p_values)[ordem]
    threshold  = (np.arange(1, n + 1) / n) * alpha
    rej_sorted = p_sorted <= threshold
    rej_sorted = np.maximum.accumulate(rej_sorted[::-1])[::-1]
    rejeita    = np.empty(n, dtype=bool)
    rejeita[ordem] = rej_sorted
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
    Coleta pares (default x otimizado) pareados por (dataset_pasta x fold).
    A agregação split -> fonte ocorre em etapa posterior.
    """
    rows = []

    for pasta, (scen_def, scen_ot, rotulo), detector in iproduct(
        pastas_saida, pares_cenario, detectores
    ):
        df_def = load_csv(pasta, scen_def, detector, arquivo)
        df_ot  = load_csv(pasta, scen_ot,  detector, arquivo)

        if df_def.empty or df_ot.empty:
            continue

        dataset_pasta = Path(pasta).name

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
# NÍVEL 1 DE AGREGAÇÃO: fold -> split de sensor (dataset_pasta)
# ============================================================

def agregar_por_split(df_pares: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os pares fold-level por split de sensor (dataset_pasta),
    calculando a média de val_default e val_otimizado sobre todos os
    folds. Cada split ainda NÃO é necessariamente uma fonte
    independente (ver agregar_por_fonte).
    """
    agg = (
        df_pares
        .groupby(["dataset_pasta", "cenario", "detector"], as_index=False)
        .agg(
            val_default   = ("val_default",   "mean"),
            val_otimizado = ("val_otimizado", "mean"),
            n_folds       = ("fold",          "count"),
        )
    )
    return agg


# ============================================================
# NÍVEL 2 DE AGREGAÇÃO: split de sensor -> fonte independente
# ============================================================

def agregar_por_fonte(df_split: pd.DataFrame, meio: str = MEIO_AGREGACAO) -> pd.DataFrame:
    """
    Agrega os splits de sensor (P2-Hand/Chest/Ankle, W-*) em suas
    respectivas fontes independentes, usando o critério MEIO_AGREGACAO.

    val_default e val_otimizado são agregados independentemente entre
    os splits da mesma fonte. Fontes de split único não são afetadas.
    """
    func = _FUNC_AGREGACAO[meio]

    df = df_split.copy()
    df["fonte"] = df["dataset_pasta"].map(mapear_fonte)

    linhas = []
    for (fonte, cenario, detector), grupo in df.groupby(["fonte", "cenario", "detector"]):
        linhas.append({
            "fonte"         : fonte,
            "cenario"       : cenario,
            "detector"      : detector,
            "val_default"   : func(grupo["val_default"].to_numpy(dtype=float)),
            "val_otimizado" : func(grupo["val_otimizado"].to_numpy(dtype=float)),
            "n_splits"      : int(grupo["dataset_pasta"].nunique()),
            "splits"        : ",".join(sorted(grupo["dataset_pasta"].unique())),
            "n_folds_total" : int(grupo["n_folds"].sum()),
        })
    return pd.DataFrame(linhas)


# ============================================================
# ANÁLISE ESTATÍSTICA (genérica: usada para split e fonte)
# ============================================================

def _wilcoxon_seguro(dif: np.ndarray):
    n_nao_zero = int(np.sum(dif != 0))
    if n_nao_zero < 1:
        return np.nan, 1.0
    try:
        stat, p_value = wilcoxon(dif[dif != 0], alternative="two-sided")
        return stat, p_value
    except Exception as e:
        warnings.warn(f"Wilcoxon falhou: {e}")
        return np.nan, 1.0


def _analisar(df: pd.DataFrame, col_unidade: str, rotulo_unidade: str) -> pd.DataFrame:
    """
    Núcleo comum de análise: agrupa por (cenario, detector), aplica
    Wilcoxon + Cliff's Delta, e corrige por BH dentro de cada cenário.

    col_unidade    : nome da coluna que identifica a unidade pareada
                      ("dataset_pasta" para split, "fonte" para fonte).
    rotulo_unidade : rótulo usado no nome da coluna de contagem de saída
                      (ex.: "n_splits" ou "n_fontes").
    """
    resultados = []

    for (cenario, detector), grupo in df.groupby(["cenario", "detector"]):
        def_arr = grupo["val_default"].to_numpy(dtype=float)
        ot_arr  = grupo["val_otimizado"].to_numpy(dtype=float)
        dif     = ot_arr - def_arr

        n_unidades  = len(dif)
        n_folds_med = round(grupo["n_folds"].mean(), 1) if "n_folds" in grupo else np.nan
        n_nao_zero  = int(np.sum(dif != 0))
        media_dif   = float(np.mean(dif))
        mediana_dif = float(np.median(dif))
        delta       = cliff_delta_paired(def_arr, ot_arr)
        mag         = magnitude_cliff(delta)
        stat, p_value = _wilcoxon_seguro(dif)

        resultados.append({
            "cenario"       : cenario,
            "detector"      : detector,
            rotulo_unidade  : n_unidades,
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

    for cenario, grupo in df_res.groupby("cenario"):
        idx     = grupo.index.tolist()
        rejeita = bh_correction(grupo["p_value"].tolist(), alpha=ALPHA_FDR)
        for i, rej in zip(idx, rejeita):
            df_res.at[i, "sig_bh"] = bool(rej)

    def recomendacao(row):
        if row["sig_bh"] and row["magnitude"] != "irrelevante" and row["media_dif_pp"] > 0:
            return "SIM"
        if row["sig_bh"] and row["magnitude"] != "irrelevante" and row["media_dif_pp"] < 0:
            return "NÃO (piora)"
        return "NÃO (efeito irrelevante ou não significativo)"

    df_res["recomenda_otimizar"] = df_res.apply(recomendacao, axis=1)

    return df_res.sort_values(["cenario", "detector"]).reset_index(drop=True)


def analisar_split(df_split: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon sobre pares por split de sensor (~16 unidades).
    Diagnóstico: ainda contém pseudorreplicação de P2/W."""
    return _analisar(df_split, "dataset_pasta", "n_splits")


def analisar_fonte(df_fonte: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon sobre pares por fonte independente (n=10).
    Resultado primário recomendado."""
    return _analisar(df_fonte, "fonte", "n_fontes")


# ============================================================
# IMPRESSÃO COMPARATIVA — split vs. fonte
# ============================================================

def imprimir_comparacao(df_split: pd.DataFrame, df_fonte: pd.DataFrame,
                        rotulo_metrica: str):
    sep = "=" * 150
    print(f"\n{sep}")
    print(f"DEFAULT vs. OTIMIZADO — {rotulo_metrica.upper()}")
    print(f"  split : Wilcoxon pareado por split de sensor    (n~16, P2/W pseudorreplicados)")
    print(f"  fonte : Wilcoxon pareado por fonte independente (n=10, agregação={MEIO_AGREGACAO})")
    print(f"Correção BH aplicada sobre os 9 detectores dentro de cada cenário (α={ALPHA_FDR})")
    print(sep)

    df_merged = pd.merge(
        df_split[["cenario", "detector", "n_splits", "cliff_delta", "magnitude", "sig_bh", "media_dif_pp"]]
        .rename(columns={"cliff_delta": "delta_split", "magnitude": "mag_split", "sig_bh": "sig_split"}),
        df_fonte[["cenario", "detector", "n_fontes", "cliff_delta", "magnitude", "sig_bh", "media_dif_pp", "recomenda_otimizar"]]
        .rename(columns={"cliff_delta": "delta_fonte", "magnitude": "mag_fonte", "sig_bh": "sig_fonte",
                          "media_dif_pp": "media_dif_pp_fonte"}),
        on=["cenario", "detector"],
    )

    for cenario, grupo in df_merged.groupby("cenario"):
        print(f"\n{'─'*150}")
        print(f"  CENÁRIO {cenario}")
        print(f"{'─'*150}")
        print(
            f"  {'Detector':<24} "
            f"{'-- split (n~16) --':^28}  {'-- fonte (n=10) --':^28}  {'split→fonte diverge?'}  {'Compensa otimizar?'}"
        )
        print(f"  {'─'*146}")
        for _, row in grupo.iterrows():
            sig_sp = "✓" if row["sig_split"] else "✗"
            sig_fo = "✓" if row["sig_fonte"] else "✗"
            diverge = (row["mag_split"] != row["mag_fonte"]) or (row["sig_split"] != row["sig_fonte"])
            flag = "⚠ DIVERGE" if diverge else ""
            print(
                f"  {row['detector']:<24} "
                f"n={row['n_splits']:<4} δ={row['delta_split']:+.2f} {row['mag_split']:<11} {sig_sp:>3}  "
                f"n={row['n_fontes']:<4} δ={row['delta_fonte']:+.2f} {row['mag_fonte']:<11} {sig_fo:>3}  "
                f"{flag:<20}  {row['recomenda_otimizar']}"
            )

    print(f"\n{sep}")
    print("LEGENDA")
    print("  δ           : Cliff's Delta; positivo = otimizado tende a superar default")
    print("  magnitude   : irrelevante |δ|<0.147 | pequeno <0.330 | médio <0.474 | grande ≥0.474")
    print("  sig (✓/✗)   : significativo após correção BH (α=0.05)")
    print("  ⚠ DIVERGE  : magnitude ou significância mudam entre split-de-sensor e fonte")
    print("  Compensa    : SIM apenas quando sig, magnitude≠irrelevante e ganho positivo (nível fonte)")
    print(sep)


# ============================================================
# TABELA COLORIDA — C1 e C3 (F1 + Acc), nível fonte
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
    cell_h   = 1.0
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

    #fig.text(0.5, 0.01,
    #          f"Paired by independent source (n=10) — sensor splits aggregated via {MEIO_AGREGACAO}",
    #          ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout(rect=[0.07, 0.03, 1.0, 1.0])
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura salva em: {caminho_saida}")


# ============================================================
# MAIN
# ============================================================

def main():
    pastas_saida = descobrir_pastas_saida(PASTA_RAIZ)
    imprimir_mapeamento_fonte(pastas_saida)

    resultados_por_metrica = {}

    for arquivo, rotulo_metrica, sufixo in METRICAS:

        print(f"\n{'#'*60}")
        print(f"  MÉTRICA: {rotulo_metrica}  ({arquivo})")
        print(f"{'#'*60}")

        print(f"\nColetando pares (fold-level)...", flush=True)
        df_pares = coletar_pares(pastas_saida, DETECTORES, PARES_CENARIO, arquivo)
        print(f"  {len(df_pares)} pares fold-level | "
              f"{df_pares['dataset_pasta'].nunique()} splits de sensor")

        print(f"Agregando fold -> split de sensor...", flush=True)
        df_split = agregar_por_split(df_pares)
        print(f"  {len(df_split)} linhas após agregação "
              f"({df_split['dataset_pasta'].nunique()} splits)")

        print(f"Agregando split -> fonte (MEIO_AGREGACAO={MEIO_AGREGACAO})...", flush=True)
        df_fonte = agregar_por_fonte(df_split, meio=MEIO_AGREGACAO)
        n_fontes = df_fonte["fonte"].nunique()
        print(f"  {len(df_fonte)} linhas após agregação ({n_fontes} fontes independentes)")

        # Diagnóstico: valores agregados por fonte antes do teste, para
        # conferir se AVG/MAX/MIN produzem val_default/val_otimizado
        # distintos entre execuções.
        out_valores = os.path.join(
            PASTA_RAIZ, f"valores_agregados_fonte_ot_{MEIO_AGREGACAO.lower()}_{sufixo}.csv"
        )
        df_fonte.assign(dif=df_fonte["val_otimizado"] - df_fonte["val_default"]) \
                .sort_values(["cenario", "detector", "fonte"]) \
                .to_csv(out_valores, index=False)
        print(f"  Valores agregados por fonte (diagnóstico) salvos em: {out_valores}")

        print(f"Aplicando testes estatísticos...", flush=True)
        df_res_split = analisar_split(df_split)
        df_res_fonte = analisar_fonte(df_fonte)

        imprimir_comparacao(df_res_split, df_res_fonte, rotulo_metrica)

        out_fonte = os.path.join(
            PASTA_RAIZ, f"analise_default_vs_otimizado_fonte_{sufixo}.csv"
        )
        out_split = os.path.join(
            PASTA_RAIZ, f"analise_default_vs_otimizado_split_{sufixo}.csv"
        )
        df_res_fonte.to_csv(out_fonte, index=False)
        df_res_split.to_csv(out_split, index=False)
        print(f"\nResultado primário (por fonte, n=10) salvo em: {out_fonte}")
        print(f"Resultado diagnóstico (por split, n~16) salvo em: {out_split}")

        resultados_por_metrica[sufixo] = {}
        for cen, grupo in df_res_fonte.groupby("cenario"):
            resultados_por_metrica[sufixo][cen] = grupo.reset_index(drop=True)

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