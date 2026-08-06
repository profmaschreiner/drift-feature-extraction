"""
teste_estatistico_cenarios_agregado_fonte.py
=============================================
Versão do teste estatístico que agrega os resultados POR FONTE
INDEPENDENTE antes de aplicar o Wilcoxon signed-rank, corrigindo um
problema de pseudorreplicação (Hurlbert, 1984, Ecological Monographs
54(2):187-211) presente na versão agregada por dataset.

Motivação
---------
O script `teste_estatistico_cenarios_agr.py` já corrige o problema de
pareamento por fold (folds dentro do mesmo dataset não são independentes,
pois os conjuntos de treino se sobrepõem), agregando fold -> dataset antes
do teste. Isso segue Demšar (2006): o teste deve ser pareado pela unidade
verdadeiramente independente.

Entretanto, 8 dos 16 "datasets" avaliados (P2-Hand/Chest/Ankle e
W-LWrist/RWrist/Waist/LAnkle/RAnkle) não são fontes independentes entre
si: compartilham os mesmos sujeitos, as mesmas sessões de coleta e os
mesmos folds estruturais, diferindo apenas no canal de sensor extraído.
Tratar esses 8 splits como 8 observações independentes no Wilcoxon
pareado por dataset é pseudorreplicação: infla o n efetivo do teste e,
com isso, a significância aparente.

Este script aplica uma segunda agregação, split-de-sensor -> fonte,
antes do teste, reduzindo o n de ~16 datasets para as 10 fontes reais
verdadeiramente independentes (Tabela II do artigo). O critério de
agregação entre splits da mesma fonte é controlado pela constante
MEIO_AGREGACAO.

MEIO_AGREGACAO
--------------
    "AVG" : média de val_A e val_B entre os splits da mesma fonte
            (critério recomendado; análogo à média entre folds já
            usada na agregação fold -> dataset).
    "MAX" : maior valor de val_A e de val_B entre os splits, calculado
            independentemente para cada coluna (visão otimista / "melhor
            sensor", útil como checagem de sensibilidade).
    "MIN" : menor valor de val_A e de val_B entre os splits, calculado
            independentemente para cada coluna (visão pessimista / "pior
            sensor", útil como checagem de sensibilidade).

Fontes com um único split (DR, Gait, RO, MH, SmPh, Sp, Sw, USC) não são
afetadas pela escolha de MEIO_AGREGACAO: seu valor agregado é o próprio
valor do split único.

IMPORTANTE — antes de confiar nos resultados
---------------------------------------------
Ajuste GRUPOS_FONTE para refletir os nomes reais das pastas de dataset
em PASTA_RAIZ. O script imprime o mapeamento inferido (dataset -> fonte)
no início da execução; confira essa tabela antes de interpretar o p-valor.

Saídas
------
* Console : mapeamento dataset -> fonte, e tabela comparando os três
            níveis de agregação (fold / split-de-sensor / fonte) por
            comparação x detector, sinalizando divergências entre
            split-de-sensor e fonte (o teste relevante para este ajuste).
* CSV     : analise_cenarios_fonte_{metrica}.csv  (resultado primário,
            n = fontes independentes)
            analise_cenarios_split_{metrica}.csv  (diagnóstico, n = splits,
            mantido para referência/comparação)
* PNG     : tabela_comparacao_cenarios_fonte.png  (mesma estrutura visual
            do script original, calculada sobre as fontes independentes)

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
# aplicado independentemente a val_A e a val_B.
#   "AVG" (recomendado) | "MAX" | "MIN"
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

FIGURA_SAIDA = os.path.join(
    PASTA_RAIZ, f"tabela_comparacao_cenarios_fonte_{MEIO_AGREGACAO.lower()}.png"
)

# ============================================================
# DESCOBERTA DE PASTAS
# ============================================================

def descobrir_pastas_datasets(pasta_raiz: str) -> list:
    raiz = Path(pasta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Pasta raiz não encontrada: {pasta_raiz}\n"
            "Ajuste a variável PASTA_RAIZ no início do script."
        )
    cenarios_validos = {c[0] for c in COMPARACOES} | {c[1] for c in COMPARACOES}
    pastas = sorted([
        str(p) for p in raiz.iterdir()
        if p.is_dir() and any((p / c).is_dir() for c in cenarios_validos)
    ])
    if not pastas:
        raise RuntimeError(
            f"Nenhuma subpasta válida encontrada em '{pasta_raiz}'.\n"
            f"Esperado ao menos um de: {sorted(cenarios_validos)}"
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


def imprimir_mapeamento_fonte(pastas_datasets: list) -> None:
    print("\n" + "=" * 70)
    print("MAPEAMENTO DATASET (SPLIT DE SENSOR) -> FONTE INDEPENDENTE")
    print("Confira esta tabela contra a estrutura real de pastas ANTES de")
    print("confiar nos p-valores calculados a seguir.")
    print("=" * 70)

    fontes = {}
    for p in pastas_datasets:
        nome = Path(p).name
        fonte = mapear_fonte(nome)
        fontes.setdefault(fonte, []).append(nome)

    todos_splits_mapeados = {s for splits in GRUPOS_FONTE.values() for s in splits}
    for fonte, splits in sorted(fontes.items()):
        marcador = " (múltiplos splits agregados)" if len(splits) > 1 else ""
        print(f"  {fonte:<10} <- {splits}{marcador}")

        # Alerta heurístico: nome parece ser split de sensor mas não
        # está listado em GRUPOS_FONTE.
        for nome in splits:
            if nome not in todos_splits_mapeados and nome.startswith(_PREFIXOS_ALERTA):
                warnings.warn(
                    f"'{nome}' parece ser um split de sensor não mapeado "
                    f"em GRUPOS_FONTE (foi tratado como fonte própria). "
                    f"Verifique se isso está correto."
                )

    print(f"\nTotal de datasets (splits): {len(pastas_datasets)}")
    print(f"Total de fontes independentes: {len(fontes)}")
    print(f"Critério de agregação entre splits: MEIO_AGREGACAO = {MEIO_AGREGACAO}")
    print("=" * 70)


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
# COLETA DE PARES (fold-level, igual ao original)
# ============================================================

def coletar_pares(pastas_datasets: list, detectores: list,
                  comparacoes: list, arquivo: str) -> pd.DataFrame:
    rows = []

    for pasta, (cen_a, cen_b, rotulo, _) in iproduct(pastas_datasets, comparacoes):
        dataset_pasta = Path(pasta).name

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
# NÍVEL 1 DE AGREGAÇÃO: fold -> split de sensor (dataset_pasta)
# ============================================================

def agregar_por_split(df_pares: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os pares fold-level por split de sensor (dataset_pasta),
    calculando a média de val_A e val_B sobre todos os folds.

    Cada split de sensor ainda NÃO é necessariamente uma fonte
    independente (ver agregar_por_fonte).
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
# NÍVEL 2 DE AGREGAÇÃO: split de sensor -> fonte independente
# ============================================================

def agregar_por_fonte(df_split: pd.DataFrame, meio: str = MEIO_AGREGACAO) -> pd.DataFrame:
    """
    Agrega os splits de sensor (P2-Hand/Chest/Ankle, W-*) em suas
    respectivas fontes independentes, usando o critério MEIO_AGREGACAO.

    val_A e val_B são agregados independentemente entre os splits da
    mesma fonte (ex.: MAX pega o maior val_A entre os sensores e o maior
    val_B entre os sensores, não necessariamente do mesmo split).

    Fontes de split único (a maioria) não são afetadas: seu valor
    agregado é o próprio valor do split.
    """
    func = _FUNC_AGREGACAO[meio]

    df = df_split.copy()
    df["fonte"] = df["dataset_pasta"].map(mapear_fonte)

    linhas = []
    for (fonte, comparacao, detector), grupo in df.groupby(["fonte", "comparacao", "detector"]):
        linhas.append({
            "fonte"        : fonte,
            "comparacao"   : comparacao,
            "detector"     : detector,
            "val_A"        : func(grupo["val_A"].to_numpy(dtype=float)),
            "val_B"        : func(grupo["val_B"].to_numpy(dtype=float)),
            "n_splits"     : int(grupo["dataset_pasta"].nunique()),
            "splits"       : ",".join(sorted(grupo["dataset_pasta"].unique())),
            "n_folds_total": int(grupo["n_folds"].sum()),
        })
    return pd.DataFrame(linhas)


# ============================================================
# ANÁLISE ESTATÍSTICA (genérica: usada para fold / split / fonte)
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


def analisar_fold(df_pares: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon sobre pares (dataset_pasta x fold). Diagnóstico apenas:
    folds dentro do mesmo split não são independentes."""
    resultados = []
    for (comparacao, detector), grupo in df_pares.groupby(["comparacao", "detector"]):
        arr_a = grupo["val_A"].to_numpy(dtype=float)
        arr_b = grupo["val_B"].to_numpy(dtype=float)
        dif   = arr_b - arr_a
        stat, p_value = _wilcoxon_seguro(dif)
        delta = cliff_delta_paired(arr_a, arr_b)
        resultados.append({
            "comparacao": comparacao, "detector": detector,
            "n_unidades": len(dif),
            "cliff_delta": round(delta, 4), "magnitude": magnitude_cliff(delta),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value": round(p_value, 6),
        })
    return _finalizar_analise(pd.DataFrame(resultados))


def analisar_split(df_split: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon sobre pares por split de sensor (~16 unidades). Ainda
    contém pseudorreplicação: P2 e W contribuem 3 e 5 unidades
    correlacionadas, respectivamente."""
    resultados = []
    for (comparacao, detector), grupo in df_split.groupby(["comparacao", "detector"]):
        arr_a = grupo["val_A"].to_numpy(dtype=float)
        arr_b = grupo["val_B"].to_numpy(dtype=float)
        dif   = arr_b - arr_a
        stat, p_value = _wilcoxon_seguro(dif)
        delta = cliff_delta_paired(arr_a, arr_b)
        resultados.append({
            "comparacao": comparacao, "detector": detector,
            "n_unidades": len(dif),
            "cliff_delta": round(delta, 4), "magnitude": magnitude_cliff(delta),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value": round(p_value, 6),
        })
    return _finalizar_analise(pd.DataFrame(resultados))


def analisar_fonte(df_fonte: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon sobre pares por fonte independente (n=10). Este é o
    teste primário recomendado: as 10 fontes reais não compartilham
    sujeitos, sessões de coleta nem estrutura de folds entre si."""
    resultados = []
    for (comparacao, detector), grupo in df_fonte.groupby(["comparacao", "detector"]):
        arr_a = grupo["val_A"].to_numpy(dtype=float)
        arr_b = grupo["val_B"].to_numpy(dtype=float)
        dif   = arr_b - arr_a
        stat, p_value = _wilcoxon_seguro(dif)
        delta = cliff_delta_paired(arr_a, arr_b)
        resultados.append({
            "comparacao": comparacao, "detector": detector,
            "n_unidades": len(dif),
            "n_splits_total": int(grupo["n_splits"].sum()),
            "cliff_delta": round(delta, 4), "magnitude": magnitude_cliff(delta),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value": round(p_value, 6),
        })
    return _finalizar_analise(pd.DataFrame(resultados))


def _finalizar_analise(df_res: pd.DataFrame) -> pd.DataFrame:
    if df_res.empty:
        return df_res

    df_res["sig_bh"] = False
    for comparacao, grupo in df_res.groupby("comparacao"):
        idx     = grupo.index.tolist()
        rejeita = bh_correction(grupo["p_value"].tolist(), alpha=ALPHA_FDR)
        for i, rej in zip(idx, rejeita):
            df_res.at[i, "sig_bh"] = bool(rej)

    def interpretar(row):
        if row["sig_bh"] and row["magnitude"] != "irrelevante":
            return "B > A" if row["cliff_delta"] > 0 else "A > B"
        if row["sig_bh"]:
            return "sig. mas irrelevante"
        return "não significativo"

    df_res["resultado"] = df_res.apply(interpretar, axis=1)

    ordem_comp = {c[2]: i for i, c in enumerate(COMPARACOES)}
    df_res["_ordem"] = df_res["comparacao"].map(ordem_comp)
    df_res = df_res.sort_values(["_ordem", "detector"]).drop(columns="_ordem")
    return df_res.reset_index(drop=True)


# ============================================================
# IMPRESSÃO COMPARATIVA — 3 NÍVEIS
# ============================================================

def imprimir_comparacao(df_fold: pd.DataFrame, df_split: pd.DataFrame,
                        df_fonte: pd.DataFrame, rotulo_metrica: str):
    """
    Compara os três níveis de agregação lado a lado, destacando
    divergências entre split-de-sensor (n~16, pseudorreplicado) e
    fonte (n=10, independente) — o teste relevante para este ajuste.
    """
    hipoteses = {c[2]: c[3] for c in COMPARACOES}
    sep = "=" * 150
    print(f"\n{sep}")
    print(f"COMPARAÇÃO ENTRE NÍVEIS DE AGREGAÇÃO — {rotulo_metrica.upper()}")
    print(f"  fold  : Wilcoxon pareado por dataset x fold        (n alto,  folds correlacionados)")
    print(f"  split : Wilcoxon pareado por split de sensor       (n~16,   P2/W pseudorreplicados)")
    print(f"  fonte : Wilcoxon pareado por fonte independente    (n=10,   agregação={MEIO_AGREGACAO})")
    print(f"Correção BH aplicada sobre os 9 detectores dentro de cada comparação (α={ALPHA_FDR})")
    print(sep)

    df_merged = (
        df_fold[["comparacao", "detector", "n_unidades", "cliff_delta", "magnitude", "sig_bh"]]
        .rename(columns={"n_unidades": "n_fold", "cliff_delta": "delta_fold", "magnitude": "mag_fold", "sig_bh": "sig_fold"})
        .merge(
            df_split[["comparacao", "detector", "n_unidades", "cliff_delta", "magnitude", "sig_bh"]]
            .rename(columns={"n_unidades": "n_split", "cliff_delta": "delta_split", "magnitude": "mag_split", "sig_bh": "sig_split"}),
            on=["comparacao", "detector"],
        )
        .merge(
            df_fonte[["comparacao", "detector", "n_unidades", "cliff_delta", "magnitude", "sig_bh"]]
            .rename(columns={"n_unidades": "n_fonte", "cliff_delta": "delta_fonte", "magnitude": "mag_fonte", "sig_bh": "sig_fonte"}),
            on=["comparacao", "detector"],
        )
    )

    ordem_comp = {c[2]: i for i, c in enumerate(COMPARACOES)}
    df_merged["_ordem"] = df_merged["comparacao"].map(ordem_comp)
    df_merged = df_merged.sort_values(["_ordem", "detector"]).drop(columns="_ordem")

    for comparacao, grupo in df_merged.groupby("comparacao", sort=False):
        hip = hipoteses.get(comparacao, "")
        print(f"\n{'─'*150}")
        print(f"  COMPARAÇÃO: {comparacao}  —  {hip}")
        print(f"{'─'*150}")
        print(
            f"  {'Detector':<24} "
            f"{'-- fold --':^20}  {'-- split --':^20}  {'-- fonte --':^20}  {'split→fonte diverge?'}"
        )
        print(f"  {'─'*146}")
        for _, row in grupo.iterrows():
            sig_sp = "✓" if row["sig_split"] else "✗"
            sig_fo = "✓" if row["sig_fonte"] else "✗"

            # Divergência relevante: entre split (pseudorreplicado) e
            # fonte (independente) — é isso que este ajuste deve revelar.
            diverge = (row["mag_split"] != row["mag_fonte"]) or (row["sig_split"] != row["sig_fonte"])
            flag = "  ⚠ DIVERGE" if diverge else ""

            print(
                f"  {row['detector']:<24} "
                f"n={row['n_fold']:<4} δ={row['delta_fold']:+.2f} {row['mag_fold']:<11}  "
                f"n={row['n_split']:<4} δ={row['delta_split']:+.2f} {row['mag_split']:<11}  "
                f"n={row['n_fonte']:<4} δ={row['delta_fonte']:+.2f} {row['mag_fonte']:<11} {sig_fo:>3}  "
                f"{flag}"
            )

    print(f"\n{sep}")
    print("LEGENDA")
    print("  δ           : Cliff's Delta; positivo = B tende a superar A")
    print("  magnitude   : irrelevante |δ|<0.147 | pequeno <0.330 | médio <0.474 | grande ≥0.474")
    print("  sig (✓/✗)   : significativo após correção BH (α=0.05)")
    print("  ⚠ DIVERGE  : magnitude ou significância mudam entre split-de-sensor e fonte")
    print("                (indica que a pseudorreplicação de P2/W estava afetando a conclusão)")
    print(sep)


# ============================================================
# FIGURA (mesma estrutura visual do script original, sobre fonte)
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

    #fig.text(0.5, 0.01,
    #          f"Paired by independent source (n=10) — sensor splits aggregated via {MEIO_AGREGACAO}",
    #          ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout(rect=[0.10, 0.03, 1.0, 1.0])
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura (agregada por fonte) salva em: {caminho_saida}")


# ============================================================
# MAIN
# ============================================================

def main():
    pastas_datasets = descobrir_pastas_datasets(PASTA_RAIZ)
    imprimir_mapeamento_fonte(pastas_datasets)

    resultados_fonte = {comp: {} for comp in [c[2] for c in COMPARACOES]}

    for arquivo, rotulo_metrica, sufixo in METRICAS:

        print(f"\n{'#'*70}")
        print(f"  MÉTRICA: {rotulo_metrica}  ({arquivo})")
        print(f"{'#'*70}")

        print(f"\nColetando pares — {rotulo_metrica}...", flush=True)
        df_pares = coletar_pares(pastas_datasets, DETECTORES, COMPARACOES, arquivo)
        print(f"  {len(df_pares)} pares fold-level | "
              f"{df_pares['dataset_pasta'].nunique()} splits de sensor")

        df_split = agregar_por_split(df_pares)
        print(f"  {len(df_split)} linhas após agregação fold -> split "
              f"({df_split['dataset_pasta'].nunique()} splits)")

        df_fonte = agregar_por_fonte(df_split, meio=MEIO_AGREGACAO)
        n_fontes = df_fonte["fonte"].nunique()
        print(f"  {len(df_fonte)} linhas após agregação split -> fonte "
              f"({n_fontes} fontes independentes, MEIO_AGREGACAO={MEIO_AGREGACAO})")

        # Diagnóstico: valores agregados por fonte ANTES do teste, para
        # conferir se AVG/MAX/MIN realmente produzem val_A/val_B distintos
        # (se o teste der o mesmo resultado, este CSV mostra se é porque
        # os valores agregados mudaram mas o sinal da diferença não, ou
        # porque os valores em si não mudaram entre execuções).
        out_valores = os.path.join(
            PASTA_RAIZ, f"valores_agregados_fonte_{MEIO_AGREGACAO.lower()}_{sufixo}.csv"
        )
        df_fonte.assign(dif=df_fonte["val_B"] - df_fonte["val_A"]) \
                .sort_values(["comparacao", "detector", "fonte"]) \
                .to_csv(out_valores, index=False)
        print(f"  Valores agregados por fonte (diagnóstico) salvos em: {out_valores}")


        df_res_fold  = analisar_fold(df_pares)
        df_res_split = analisar_split(df_split)
        df_res_fonte = analisar_fonte(df_fonte)

        imprimir_comparacao(df_res_fold, df_res_split, df_res_fonte, rotulo_metrica)

        out_fonte = os.path.join(PASTA_RAIZ, f"analise_cenarios_fonte_{sufixo}.csv")
        out_split = os.path.join(PASTA_RAIZ, f"analise_cenarios_split_{sufixo}.csv")
        df_res_fonte.to_csv(out_fonte, index=False)
        df_res_split.to_csv(out_split, index=False)
        print(f"\nResultado primário (por fonte, n=10) salvo em: {out_fonte}")
        print(f"Resultado diagnóstico (por split, n~16) salvo em: {out_split}")

        for comp, grupo in df_res_fonte.groupby("comparacao"):
            resultados_fonte[comp][sufixo] = grupo.reset_index(drop=True)

    comps_disponiveis = [
        comp for comp in COMPARACOES_FIGURA
        if "f1" in resultados_fonte.get(comp, {})
        and "acc" in resultados_fonte.get(comp, {})
    ]
    if len(comps_disponiveis) == len(COMPARACOES_FIGURA):
        gerar_figura(resultados_fonte, caminho_saida=FIGURA_SAIDA)
    else:
        faltando = [c for c in COMPARACOES_FIGURA if c not in comps_disponiveis]
        print(f"\n⚠  Figura não gerada. Comparações sem dados: {faltando}")


if __name__ == "__main__":
    main()