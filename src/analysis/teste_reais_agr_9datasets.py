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
from itertools import product as iproduct, combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range, spearmanr, kendalltau

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

NOME_PARA_COL = {
    "HDDMAverage": "HDDMA", "HDDMWeighted": "HDDMW", "PageHinkley": "PH",
    "GeometricMovingAverage": "GMA", "EWMAChart": "EWMAC", "KSWIN": "KSWIN",
    "CUSUM": "CUSUM", "ADWIN": "ADWIN", "SEED": "SEED",
}

# Cenarios BRUTOS (nomes de pasta) que variam por detector e para os
# quais faz sentido calcular ranking/Friedman/Nemenyi entre os 9
# detectores individuais, junto com a familia de comparacao oficial
# (COMPARACOES acima) cujo cliff_delta/p-valor deve ser usado como
# criterio primario de priorizacao (ver montar_ranking_priorizacao).
#
#   "c1" -> cenario "drift" (features-drift isoladas), comparado ao
#           baseline na familia C0_vs_C1.
#   "c2" -> cenario "catch24_drift" (catch22 + drift), comparado ao
#           catch22 isolado na familia C2_vs_C3 -- e essa comparacao,
#           nao C0_vs_C3 (inexistente no protocolo), que responde a
#           pergunta relevante para esse cenario ("o drift complementa
#           o catch22?"). NOTA: na nomenclatura ASOC este cenario e
#           chamado C3 (catch22+drift); aqui usamos o rotulo "c2" so
#           para manter a mesma convencao de nomes de arquivo usada no
#           script do MiniRocket (teste_estatistico_minirocket.py).
CENARIOS_RANKING = [
    ("c1", "drift",         "C0_vs_C1"),
    ("c2", "catch24_drift", "C2_vs_C3"),
]

# ============================================================
# CONFIGURAÇÃO — RESULTADOS DO MINIROCKET (pipeline separado)
# ============================================================
# Usado apenas pela analise de TRANSFERENCIA DE RANKING POR DATASET
# (ver secao "TRANSFERENCIA DE RANKING ENTRE PIPELINES" mais abaixo).
# Estrutura de pastas esperada (mesma convencao de
# teste_estatistico_minirocket.py):
#   <MINIROCKET_PASTA_RAIZ>/<dataset_pasta_mr>/minirocket_c0/df_f1_all.csv
#   <MINIROCKET_PASTA_RAIZ>/<dataset_pasta_mr>/minirocket_c1/<detector>/df_f1_all.csv
#   <MINIROCKET_PASTA_RAIZ>/<dataset_pasta_mr>/minirocket_c2/<detector>/df_f1_all.csv
# onde df_f1_all.csv/df_acc_all.csv tem uma coluna POR CLASSIFICADOR
# (nao uma unica coluna "RF" como neste script).
MINIROCKET_PASTA_RAIZ = "exp_otimizacao/result_reais_miniroket"
MINIROCKET_CLASSIFICADORES_AVALIAR = [
    "passive_aggressive", "minirocket_softmax", "sgd_log_loss", "sgd_hinge",
]

# IMPORTANTE -- MAPEAMENTO ENTRE OS DOIS PIPELINES.
# Os pipelines ASOC (este script) e MiniRocket usam nomes de PASTA
# diferentes para os MESMOS splits de sensor. O mapeamento abaixo foi
# inferido a partir dos nomes de dataset_pasta ja vistos em analises
# anteriores desta conversa (heatmap por split do MiniRocket), e esta
# marcado por confianca:
#   "confirmado"    -> nome de pasta MiniRocket ja observado diretamente
#   "nao_confirmado" -> inferido por padrao/eliminacao, PRECISA ser
#                       conferido contra a estrutura real de pastas
#                       antes de confiar nos resultados dessa analise
#                       (o script imprime esse mapeamento no console
#                       logo no inicio, para conferencia).
# Ajuste os valores do dicionario abaixo se algum nome estiver errado.
MAPEAMENTO_DATASET_MINIROCKET = {
    # nome usado neste script (ASOC, coluna 'dataset_pasta')
    #     -> nome da pasta correspondente em MINIROCKET_PASTA_RAIZ
    "gait":             "gait",              # confirmado
    "occ":              "occ",               # confirmado
    "rs":               "rs",                # confirmado
    "mhealth":          "mhealt",            # confirmado (typo preservado de proposito)
    "sw_har":           "sw",                # confirmado
    "usc_had":          "usc",               # confirmado
    "pamap2_chest":     "P2-Chest",          # confirmado
    "ward_left_wrist":  "W-LWrist",          # confirmado
    "ward_right_wrist": "W-RWrist",          # confirmado
    "pamap2_hand":      "P2-Hand",           # nao_confirmado (padrao P2-*)
    "pamap2_ankle":     "P2-Ankle",          # nao_confirmado (padrao P2-*)
    "ward_waist":       "W-Waist",           # nao_confirmado (padrao W-*)
    "ward_left_ankle":  "W-LAnkle",          # nao_confirmado (padrao W-*)
    "ward_right_ankle": "W-RAnkle",          # nao_confirmado (padrao W-*)
    "sp_har":           "sp",                # nao_confirmado (padrao analogo a sw->sw)
    "smartphone":       "SmPh",              # nao_confirmado -- CONFERIR (nome nunca visto)
}

# Comparacoes de transferencia de ranking POR DATASET (ver secao mais
# abaixo). cenario_a e sempre 'drift' (ASOC, 2n features, barato).
# rotulo, origem ('asoc' ou 'minirocket'), cenario_b.
COMPARACOES_TRANSFERENCIA = [
    ("drift_vs_catch24drift",  "asoc",       "catch24_drift"),
    ("drift_vs_minirocket_c1", "minirocket", "c1"),
    ("drift_vs_minirocket_c2", "minirocket", "c2"),
]

COMPARACOES = [
    ("baseline",   "drift",         "C0_vs_C1", "C1 supera C0? (drift agrega valor ao baseline?)"),
    ("baseline",   "catch24",       "C0_vs_C2", "C2 supera C0? (catch24 agrega valor ao baseline?)"),
    ("drift",      "catch24",       "C1_vs_C2", "C2 supera C1? (catch24 vs metodologia proposta)"),
    ("catch24",    "catch24_drift", "C2_vs_C3", "C3 supera C2? (drift complementa o catch24?)"),
    # Controle C2.raw (catch22 + variáveis originais, sem drift scores, sem otimização).
    # Mantidas no fim da lista para não alterar a ordem das comparações acima.
    ("catch24_raw_c3", "catch24_drift", "C2raw_vs_C3",
     "C3 supera C2.raw (hiperparâmetros do C3)? (ganho vem dos drift scores e não das originais?)"),
    ("catch24_raw_c2", "catch24_drift", "C2rawC2_vs_C3",
     "C3 supera C2.raw (hiperparâmetros do C2)?"),
]

COMPARACOES_FIGURA = ["C0_vs_C1", "C1_vs_C2", "C2_vs_C3"]

METRICAS = [
    ("df_f1_all.csv",  "F1-score macro", "f1"),
    ("df_acc_all.csv", "Acurácia",       "acc"),
]

COLUNA_METRICA         = "RF"
ALPHA_FDR              = 0.05
CENARIOS_SEM_DETECTOR  = {"baseline", "catch24", "catch24_raw_c3", "catch24_raw_c2"}

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


def agregar_por_fonte_seletivo(df_split: pd.DataFrame, grupos_agregar: dict,
                                meio: str = MEIO_AGREGACAO) -> pd.DataFrame:
    """Como agregar_por_fonte, mas agregando SOMENTE as fontes listadas
    em `grupos_agregar` (ex.: so {'W': GRUPOS_FONTE['W']}), mantendo os
    demais datasets com seus proprios nomes de split, SEM agregar --
    mesmo que estejam listados em GRUPOS_FONTE (ex.: P2, sp_sw
    permanecem como splits individuais se nao estiverem em
    `grupos_agregar`).

    Motivacao: a analise de correlacao intra-fonte mostrou concordancia
    forte entre os 5 splits de W (rho mediano ~0.5-0.55, quase sempre
    positivo) mas concordancia fraca/negativa em P2 e sp_sw. Isso
    justifica um esquema de agregacao DIFERENCIADO por fonte para a
    analise de transferencia de ranking: colapsar W (onde ha evidencia
    real de redundancia), mas preservar P2/sp_sw como unidades
    distintas (onde a evidencia aponta o contrario).

    Retorna uma tabela no MESMO FORMATO de df_split (colunas
    'dataset_pasta', 'comparacao', 'detector', 'val_A', 'val_B',
    'n_folds'), pronta para ser usada diretamente pelas funcoes ja
    existentes (extrair_valores_cenario_split, correlacao_rank_por_dataset,
    correlacao_rank_intra_fonte, etc.) -- a fonte agregada aparece como
    "mais um dataset_pasta" (com o proprio nome da fonte, ex. 'W').
    """
    func = _FUNC_AGREGACAO[meio]
    df = df_split.copy()

    def _mapear_seletivo(nome: str) -> str:
        for fonte, splits in grupos_agregar.items():
            if nome in splits:
                return fonte
        return nome  # nao esta no grupo a agregar -- mantem o proprio nome

    df["dataset_pasta_agregado"] = df["dataset_pasta"].map(_mapear_seletivo)

    linhas = []
    agrup_cols = ["dataset_pasta_agregado", "comparacao", "detector"]
    for (ds_agg, comparacao, detector), grupo in df.groupby(agrup_cols):
        linha = {
            "dataset_pasta": ds_agg, "comparacao": comparacao, "detector": detector,
            "val_A": func(grupo["val_A"].to_numpy(dtype=float)),
            "val_B": func(grupo["val_B"].to_numpy(dtype=float)),
        }
        if "n_folds" in grupo.columns:
            linha["n_folds"] = int(grupo["n_folds"].sum())
        linhas.append(linha)

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
            "n_pos": int(np.sum(dif > 0)),
            "n_neg": int(np.sum(dif < 0)),
            "cliff_delta": round(delta, 4), "magnitude": magnitude_cliff(delta),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p_value": p_value,                 # sem arredondar: usado no BH
        })
    return _finalizar_analise(pd.DataFrame(resultados))


def _bh_ajustado(p: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Benjamini-Hochberg (equivalente a
    scipy.stats.false_discovery_control(p, method='bh'))."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    ordem = np.argsort(p)
    ajust = p[ordem] * n / np.arange(1, n + 1)
    ajust = np.minimum.accumulate(ajust[::-1])[::-1]
    out = np.empty(n)
    out[ordem] = np.minimum(ajust, 1.0)
    return out


def _finalizar_analise(df_res: pd.DataFrame) -> pd.DataFrame:
    if df_res.empty:
        return df_res

    # p ajustado por Benjamini-Hochberg, entre os detectores de cada
    # comparacao (a funcao e chamada por metrica: familia = comparacao x metrica)
    df_res["p_bh"] = np.nan
    for comparacao, grupo in df_res.groupby("comparacao"):
        df_res.loc[grupo.index, "p_bh"] = _bh_ajustado(grupo["p_value"].to_numpy(dtype=float))
    df_res["sig_bh"] = df_res["p_bh"] <= ALPHA_FDR

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
# RANKING DE PRIORIZAÇÃO DE DETECTORES + TESTE DE FRIEDMAN/NEMENYI
# (Demšar, 2006), sobre os cenários que variam por detector ("drift" e
# "catch24_drift", ver CENARIOS_RANKING). Uso EXPLORATÓRIO/DESCRITIVO,
# complementar à família oficial Wilcoxon+BH -- não substitui a
# escolha do "melhor detector individual" nem a comparação oficial
# contra o baseline/catch22.
# ============================================================

def extrair_valores_cenario(df_fonte: pd.DataFrame, cenario_bruto: str,
                             comparacoes: list = COMPARACOES) -> tuple:
    """A partir de df_fonte (saida de agregar_por_fonte, colunas
    [fonte, comparacao, detector, val_A, val_B, ...]), extrai os
    valores de UM cenario bruto especifico (ex.: 'drift',
    'catch24_drift') como uma serie unica por (fonte, detector),
    reaproveitando os pares ja coletados -- sem re-ler CSVs.

    Um cenario bruto pode aparecer como lado A ou lado B em mais de
    uma comparacao configurada (ex.: 'drift' e val_B em C0_vs_C1 e
    val_A em C1_vs_C2); os valores agregados por fonte sao os mesmos
    nos dois casos, entao a primeira ocorrencia encontrada em
    `comparacoes` (na ordem declarada) e usada.

    Retorna (df_valores, familia_onde_foi_encontrado), com df_valores
    tendo colunas ['fonte', 'detector', 'val']. Se o cenario nao for
    encontrado em nenhuma comparacao, retorna um DataFrame vazio e
    familia=None.
    """
    for cen_a, cen_b, rotulo, _ in comparacoes:
        if cen_b == cenario_bruto:
            sub = df_fonte.loc[
                df_fonte["comparacao"] == rotulo, ["fonte", "detector", "val_B"]
            ].rename(columns={"val_B": "val"})
            return sub.reset_index(drop=True), rotulo
        if cen_a == cenario_bruto:
            sub = df_fonte.loc[
                df_fonte["comparacao"] == rotulo, ["fonte", "detector", "val_A"]
            ].rename(columns={"val_A": "val"})
            return sub.reset_index(drop=True), rotulo
    return pd.DataFrame(columns=["fonte", "detector", "val"]), None


def montar_ranking_priorizacao(df_fonte: pd.DataFrame, df_res_fonte: pd.DataFrame,
                                metrica: str, cenario_bruto: str,
                                familia_delta: str) -> pd.DataFrame:
    """Analogo ao ranking de priorizacao do script MiniRocket
    (teste_estatistico_minirocket.py): combina, por detector,

    1. cliff_delta da familia oficial `familia_delta` (ex.:
       'C0_vs_C1' para o cenario 'drift', 'C2_vs_C3' para o cenario
       'catch24_drift') -- direcao/magnitude do efeito com TODAS as
       fontes pareadas de uma vez.
    2. rank_medio_fonte -- para cada fonte, os 9 detectores (dentro do
       cenario_bruto) sao ordenados pelo valor naquela fonte (1 =
       melhor); a media entre fontes captura consistencia mesmo para
       um detector que nunca e o "melhor" isolado (ver discussao do
       caso ADWIN no script do MiniRocket).

    Ordena por (cliff_delta desc, rank_medio_fonte asc). Uso EXPLORATORIO
    para decidir ordem de execucao/analise, nao substitui a escolha
    oficial de "melhor detector individual" nem a familia Wilcoxon+BH.
    """
    sub_valores, familia_origem = extrair_valores_cenario(df_fonte, cenario_bruto)
    sub_valores = sub_valores[sub_valores["detector"].isin(DETECTORES)]
    colunas_vazias = [
        "prioridade", "cenario_bruto", "familia_delta", "detector", "detector_cod",
        "cliff_delta", "magnitude", "sig_bh", "p_value",
        "rank_medio_fonte", "rank_desvio_fonte", "n_fontes_rank",
    ]
    if sub_valores.empty:
        warnings.warn(
            f"[Ranking priorizacao] cenario bruto '{cenario_bruto}' nao "
            f"encontrado em nenhuma comparacao configurada -- ranking vazio."
        )
        return pd.DataFrame(columns=colunas_vazias)

    pivot = sub_valores.pivot_table(index="fonte", columns="detector", values="val")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    rank_medio = ranks.mean(axis=0, skipna=True)
    rank_desvio = ranks.std(axis=0, ddof=0, skipna=True)
    n_fontes_rank = ranks.count(axis=0)

    df_delta = df_res_fonte[
        (df_res_fonte["comparacao"] == familia_delta)
        & (df_res_fonte["detector"].isin(DETECTORES))
    ].set_index("detector")

    linhas = []
    for det in DETECTORES:
        linhas.append({
            "cenario_bruto": cenario_bruto, "familia_delta": familia_delta,
            "detector": det, "detector_cod": NOME_PARA_COL.get(det, det),
            "cliff_delta": df_delta.loc[det, "cliff_delta"] if det in df_delta.index else np.nan,
            "magnitude": df_delta.loc[det, "magnitude"] if det in df_delta.index else "irrelevante",
            "sig_bh": bool(df_delta.loc[det, "sig_bh"]) if det in df_delta.index else False,
            "p_value": df_delta.loc[det, "p_value"] if det in df_delta.index else np.nan,
            "rank_medio_fonte": round(float(rank_medio.get(det, np.nan)), 3),
            "rank_desvio_fonte": round(float(rank_desvio.get(det, np.nan)), 3),
            "n_fontes_rank": int(n_fontes_rank.get(det, 0)),
        })

    df_ranking = pd.DataFrame(linhas)
    df_ranking = df_ranking.sort_values(
        by=["cliff_delta", "rank_medio_fonte"], ascending=[False, True], na_position="last",
    ).reset_index(drop=True)
    df_ranking.insert(0, "prioridade", np.arange(1, len(df_ranking) + 1))
    return df_ranking


def imprimir_ranking_priorizacao(df_ranking: pd.DataFrame, cenario_bruto: str,
                                  rotulo_metrica: str) -> None:
    if df_ranking.empty:
        print(f"[AVISO] Ranking de priorizacao ('{cenario_bruto}', "
              f"{rotulo_metrica}) vazio -- sem dados suficientes.")
        return
    print(f"\n[Ranking de priorizacao de detectores -- cenario '{cenario_bruto}', "
          f"{rotulo_metrica}]")
    print("(uso exploratorio: ordem de consistencia entre detectores; NAO e a "
          "escolha oficial de 'melhor detector individual')")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_ranking[[
            "prioridade", "detector_cod", "cliff_delta", "magnitude", "sig_bh",
            "p_value", "rank_medio_fonte", "rank_desvio_fonte", "n_fontes_rank",
        ]].to_string(index=False))


def teste_friedman_nemenyi(df_fonte: pd.DataFrame, metrica: str, cenario_bruto: str,
                            alpha: float = 0.05):
    """Teste de Friedman + post-hoc de Nemenyi (Demsar, 2006) sobre os
    9 detectores individuais dentro de um cenario_bruto (ex.: 'drift',
    'catch24_drift'), usando fonte independente como bloco.

    Comparacao DETECTOR-VS-DETECTOR (distinta da familia oficial que
    compara cada detector contra o baseline/catch22). Exige design
    balanceado (todos os detectores com valor na mesma fonte); fontes
    incompletas sao descartadas e reportadas via warning.

    Retorna (df_friedman, df_nemenyi), mesma estrutura de
    teste_estatistico_minirocket.py.
    """
    sub_valores, _ = extrair_valores_cenario(df_fonte, cenario_bruto)
    sub_valores = sub_valores[sub_valores["detector"].isin(DETECTORES)]
    if sub_valores.empty:
        warnings.warn(
            f"[Friedman/Nemenyi] cenario bruto '{cenario_bruto}' nao "
            f"encontrado -- teste nao calculado."
        )
        return pd.DataFrame(), pd.DataFrame()

    pivot = sub_valores.pivot_table(index="fonte", columns="detector", values="val")
    pivot = pivot.reindex(columns=DETECTORES)

    pivot_completo = pivot.dropna(axis=0, how="any")
    fontes_descartadas = sorted(set(pivot.index) - set(pivot_completo.index))
    if fontes_descartadas:
        warnings.warn(
            f"[Friedman/Nemenyi '{cenario_bruto}'/{metrica}] "
            f"{len(fontes_descartadas)} fonte(s) descartada(s) por faltar "
            f"valor de algum detector: {fontes_descartadas}"
        )

    k = len(DETECTORES)
    n = len(pivot_completo)
    if n < 2:
        warnings.warn(
            f"[Friedman/Nemenyi '{cenario_bruto}'/{metrica}] apenas {n} "
            f"fonte(s) completa(s) -- teste nao calculado (minimo 2 blocos)."
        )
        return pd.DataFrame(), pd.DataFrame()

    arrays = [pivot_completo[det].to_numpy(dtype=float) for det in DETECTORES]
    try:
        stat_friedman, p_friedman = friedmanchisquare(*arrays)
    except Exception as e:
        warnings.warn(f"[Friedman '{cenario_bruto}'/{metrica}] falhou: {e}")
        return pd.DataFrame(), pd.DataFrame()

    ranks = pivot_completo.rank(axis=1, ascending=False, method="average")
    rank_medio = ranks.mean(axis=0)

    se = float(np.sqrt(k * (k + 1) / (6.0 * n)))
    q_alpha = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    cd_alpha = q_alpha * se

    df_friedman = pd.DataFrame([{
        "cenario_bruto": cenario_bruto, "metrica": metrica,
        "k_detectores": k, "n_fontes_completas": n,
        "fontes_descartadas": len(fontes_descartadas),
        "estatistica_qui2": round(float(stat_friedman), 4),
        "p_value": round(float(p_friedman), 6),
        "sig_alpha": bool(p_friedman < alpha),
        "alpha": alpha, "cd_nemenyi": round(cd_alpha, 4),
    }])

    linhas_nem = []
    for det_i, det_j in combinations(DETECTORES, 2):
        r_i, r_j = float(rank_medio[det_i]), float(rank_medio[det_j])
        diff = r_i - r_j
        z = diff / se
        q_stat = abs(z) * np.sqrt(2)
        p_val = float(studentized_range.sf(q_stat, k, np.inf))
        linhas_nem.append({
            "cenario_bruto": cenario_bruto, "metrica": metrica,
            "detector_i": det_i, "detector_j": det_j,
            "detector_i_cod": NOME_PARA_COL.get(det_i, det_i),
            "detector_j_cod": NOME_PARA_COL.get(det_j, det_j),
            "rank_medio_i": round(r_i, 3), "rank_medio_j": round(r_j, 3),
            "diferenca_rank": round(diff, 3),
            "acima_cd": bool(abs(diff) > cd_alpha),
            "p_value": round(p_val, 6),
            "sig_alpha": bool(p_val < alpha),
        })
    df_nemenyi = pd.DataFrame(linhas_nem).sort_values("p_value").reset_index(drop=True)

    return df_friedman, df_nemenyi


def imprimir_friedman_nemenyi(df_friedman: pd.DataFrame, df_nemenyi: pd.DataFrame,
                               cenario_bruto: str, rotulo_metrica: str) -> None:
    if df_friedman.empty:
        print(f"[AVISO] Friedman/Nemenyi ('{cenario_bruto}', {rotulo_metrica}) "
              f"nao calculado -- ver avisos acima.")
        return
    row = df_friedman.iloc[0]
    print(f"\n[Friedman -- cenario '{cenario_bruto}', {rotulo_metrica}]")
    print(f"  k={int(row['k_detectores'])} detectores, "
          f"n={int(row['n_fontes_completas'])} fontes completas "
          f"({int(row['fontes_descartadas'])} descartada(s) por dado faltante)")
    print(f"  qui2={row['estatistica_qui2']:.4f}  p={row['p_value']:.6f}  "
          f"{'SIGNIFICATIVO' if row['sig_alpha'] else 'nao significativo'} "
          f"a alpha={row['alpha']}")
    print(f"  Diferenca critica de Nemenyi (CD): {row['cd_nemenyi']:.4f} "
          f"(em unidades de rank medio)")

    if row["sig_alpha"]:
        print(f"\n[Nemenyi post-hoc -- pares com p < {row['alpha']}]")
        sig = df_nemenyi[df_nemenyi["sig_alpha"]]
        if sig.empty:
            print("  Nenhum par individualmente significativo apos Nemenyi "
                  "(Friedman rejeitou H0, mas nenhuma diferenca par a par "
                  "sobrevive a correcao).")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(sig[[
                    "detector_i_cod", "detector_j_cod", "rank_medio_i",
                    "rank_medio_j", "diferenca_rank", "acima_cd", "p_value",
                ]].to_string(index=False))
    else:
        print("  Friedman nao rejeitou H0 -- post-hoc de Nemenyi nao e "
              "interpretado como confirmatorio (calculado e salvo em CSV "
              "apenas como referencia descritiva).")


# ============================================================
# TRANSFERÊNCIA DE RANKING ENTRE CENÁRIOS, POR DATASET (SPLIT)
# ============================================================
# Pergunta diferente das duas secoes anteriores: nao "qual detector e
# melhor em media entre fontes" (isso e o teste oficial Wilcoxon+BH e
# o ranking/Friedman/Nemenyi acima), mas "PARA O MESMO DATASET, o
# detector que funciona bem no modelo barato (features originais +
# drift, 2n features) tambem funciona bem no modelo caro (catch22+
# drift)?".
#
# Unidade de analise: o proprio SPLIT DE SENSOR (dataset_pasta), nao a
# fonte independente agregada. Isso e INTENCIONAL aqui, nao um erro de
# pseudorreplicacao -- a pergunta e sobre consistencia DENTRO de um
# dataset especifico (ex.: P2-Hand), nao sobre generalizacao entre
# fontes independentes; canais de sensor diferentes da mesma fonte
# (P2-Hand vs P2-Chest) podem ter dinamicas de drift genuinamente
# distintas, entao usa-los como unidades separadas aqui e apropriado.
# Ainda assim, ao resumir/testar os coeficientes de correlacao entre
# datasets, reportamos tambem a versao restrita as fontes de split
# unico, como checagem de sensibilidade (ver
# resumir_correlacao_datasets).
# ============================================================

def extrair_valores_cenario_split(df_split: pd.DataFrame, cenario_bruto: str,
                                   comparacoes: list = COMPARACOES) -> tuple:
    """Igual a extrair_valores_cenario, mas a partir de df_split (saida
    de agregar_por_split, ANTES da agregacao split -> fonte), preservando
    cada dataset_pasta (ate 16) como uma linha propria."""
    for cen_a, cen_b, rotulo, _ in comparacoes:
        if cen_b == cenario_bruto:
            sub = df_split.loc[
                df_split["comparacao"] == rotulo, ["dataset_pasta", "detector", "val_B"]
            ].rename(columns={"val_B": "val"})
            return sub.reset_index(drop=True), rotulo
        if cen_a == cenario_bruto:
            sub = df_split.loc[
                df_split["comparacao"] == rotulo, ["dataset_pasta", "detector", "val_A"]
            ].rename(columns={"val_A": "val"})
            return sub.reset_index(drop=True), rotulo
    return pd.DataFrame(columns=["dataset_pasta", "detector", "val"]), None


# ------------------------------------------------------------
# CARREGAMENTO DIRETO DO PIPELINE MINIROCKET (pasta separada,
# estrutura de arquivos diferente -- ver MINIROCKET_PASTA_RAIZ e
# MAPEAMENTO_DATASET_MINIROCKET na configuracao no topo do arquivo).
# ------------------------------------------------------------

def imprimir_mapeamento_minirocket() -> None:
    """Imprime o mapeamento dataset ASOC -> pasta MiniRocket, incluindo
    quais entradas ja foram confirmadas e quais sao inferidas, e alerta
    se a pasta correspondente nao existe em disco (esperado para os
    cenarios ainda nao rodados do MiniRocket)."""
    print("\n" + "=" * 70)
    print("MAPEAMENTO DATASET (ASOC) -> PASTA MINIROCKET")
    print("Este mapeamento e PARCIALMENTE INFERIDO -- confira contra a")
    print("estrutura real de pastas antes de confiar na analise de")
    print("transferencia de ranking envolvendo o MiniRocket.")
    print("=" * 70)
    for nome_asoc, nome_mr in MAPEAMENTO_DATASET_MINIROCKET.items():
        existe = os.path.isdir(os.path.join(MINIROCKET_PASTA_RAIZ, nome_mr))
        marcador = "" if existe else "  [AUSENTE EM DISCO -- experimento ainda nao rodado?]"
        print(f"  {nome_asoc:<18} -> {nome_mr:<12}{marcador}")
    print("=" * 70)


def caminho_tarefa_minirocket(pasta_dataset_mr: str, cenario_mr: str,
                               detector: str, arquivo: str) -> str:
    """cenario_mr in {'c0','c1','c2'}. Mesma convencao de pastas de
    teste_estatistico_minirocket.py."""
    base = os.path.join(MINIROCKET_PASTA_RAIZ, pasta_dataset_mr, f"minirocket_{cenario_mr}")
    if cenario_mr == "c0":
        return os.path.join(base, arquivo)
    return os.path.join(base, detector, arquivo)


def carregar_media_classificadores_minirocket(caminho: str, classificadores: list,
                                               contexto: str):
    """Le um df_f1_all.csv/df_acc_all.csv do MiniRocket (uma coluna POR
    CLASSIFICADOR, nao uma unica coluna 'RF') e retorna a media entre
    folds da media POR FOLD dos classificadores presentes. Retorna
    None se o arquivo nao existir ou nenhum classificador estiver
    presente -- situacao ESPERADA, dado que os experimentos do
    MiniRocket ainda nao foram finalizados."""
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    if "fold" not in df.columns:
        warnings.warn(f"[MiniRocket] arquivo sem coluna 'fold': {caminho}")
        return None
    cols_presentes = [c for c in classificadores if c in df.columns]
    if not cols_presentes:
        warnings.warn(
            f"[{contexto}] nenhum classificador de {classificadores} "
            f"encontrado em {caminho}"
        )
        return None
    media_fold = df[cols_presentes].mean(axis=1, skipna=True)
    return float(media_fold.mean()) if not media_fold.empty else None


def extrair_valores_minirocket_split(cenario_mr: str, arquivo: str,
                                      mapeamento: dict = MAPEAMENTO_DATASET_MINIROCKET) -> pd.DataFrame:
    """Monta uma tabela ['dataset_pasta' (nome ASOC), 'detector', 'val']
    lendo DIRETAMENTE de MINIROCKET_PASTA_RAIZ para um cenario do
    MiniRocket ('c1' ou 'c2'), convertendo nome de pasta ASOC ->
    MiniRocket via `mapeamento`.

    Combinacoes dataset/detector ainda nao rodadas (esperado, dado que
    o MiniRocket nao foi finalizado) sao simplesmente omitidas da
    tabela resultante, com a contagem total reportada via warning ao
    final -- NAO interrompe a execucao.
    """
    linhas = []
    faltando = []
    for nome_asoc, nome_mr in mapeamento.items():
        pasta_mr = os.path.join(MINIROCKET_PASTA_RAIZ, nome_mr)
        if not os.path.isdir(pasta_mr):
            faltando.append((nome_asoc, nome_mr, "pasta_ausente"))
            continue
        for det in DETECTORES:
            caminho = caminho_tarefa_minirocket(nome_mr, cenario_mr, det, arquivo)
            contexto = f"minirocket_{cenario_mr}/{det}/{nome_mr}"
            val = carregar_media_classificadores_minirocket(
                caminho, MINIROCKET_CLASSIFICADORES_AVALIAR, contexto,
            )
            if val is None:
                faltando.append((nome_asoc, nome_mr, det))
                continue
            linhas.append({"dataset_pasta": nome_asoc, "detector": det, "val": val})

    if faltando:
        warnings.warn(
            f"[MiniRocket '{cenario_mr}'/{arquivo}] {len(faltando)} "
            f"combinacao(oes) dataset/detector ausentes (esperado -- "
            f"experimentos do MiniRocket ainda nao finalizados). "
            f"Primeiras 10: {faltando[:10]}"
        )
    return pd.DataFrame(linhas, columns=["dataset_pasta", "detector", "val"])


def correlacao_rank_por_dataset_minirocket(df_split: pd.DataFrame, cenario_a_asoc: str,
                                            cenario_mr: str, arquivo: str, metrica: str,
                                            min_detectores_comuns: int = 3,
                                            comparacoes: list = COMPARACOES) -> pd.DataFrame:
    """Igual a correlacao_rank_por_dataset, mas cruzando um cenario
    deste pipeline ASOC (cenario_a_asoc, tipicamente 'drift', o modelo
    barato de 2n features) com um cenario do PIPELINE MINIROCKET
    (cenario_mr, 'c1' ou 'c2'), lido diretamente de MINIROCKET_PASTA_RAIZ
    e convertido para os nomes de dataset deste script via
    MAPEAMENTO_DATASET_MINIROCKET.
    """
    val_a, _ = extrair_valores_cenario_split(df_split, cenario_a_asoc, comparacoes)
    val_a = val_a[val_a["detector"].isin(DETECTORES)]
    val_b = extrair_valores_minirocket_split(cenario_mr, arquivo)
    rotulo_b = f"minirocket_{cenario_mr}"
    return _correlacionar_valores_por_dataset(
        val_a, val_b, cenario_a_asoc, rotulo_b, metrica, min_detectores_comuns,
    )


def _correlacionar_valores_por_dataset(val_a: pd.DataFrame, val_b: pd.DataFrame,
                                        rotulo_a: str, rotulo_b: str, metrica: str,
                                        min_detectores_comuns: int = 3) -> pd.DataFrame:
    """Nucleo comum de correlacao por dataset: recebe DUAS tabelas ja no
    formato ['dataset_pasta', 'detector', 'val'] (de QUALQUER origem/
    pipeline -- ASOC ou MiniRocket) e calcula Spearman/Kendall entre os
    detectores em comum, separadamente para cada dataset_pasta presente
    nas duas tabelas.

    spearmanr/kendalltau operam sobre o RANK dos valores internamente,
    entao os F1/Acc brutos sao passados diretamente.

    Datasets com menos de `min_detectores_comuns` detectores em comum
    sao descartados (correlacao pouco informativa) e reportados via
    warning -- situacao esperada quando um dos dois pipelines ainda
    nao tem experimentos completos para aquele dataset/detector.
    """
    colunas_vazias = [
        "dataset_pasta", "fonte", "n_detectores_comuns",
        "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
    ]
    if val_a.empty or val_b.empty:
        warnings.warn(
            f"[Correlacao por dataset] '{rotulo_a}' ou '{rotulo_b}' sem dados "
            f"-- tabela vazia."
        )
        return pd.DataFrame(columns=colunas_vazias)

    datasets = sorted(set(val_a["dataset_pasta"]) & set(val_b["dataset_pasta"]))
    linhas = []
    datasets_descartados = []
    for ds in datasets:
        sub_a = val_a.loc[val_a["dataset_pasta"] == ds].set_index("detector")["val"]
        sub_b = val_b.loc[val_b["dataset_pasta"] == ds].set_index("detector")["val"]
        comuns = sorted(set(sub_a.index) & set(sub_b.index))
        if len(comuns) < min_detectores_comuns:
            datasets_descartados.append((ds, len(comuns)))
            continue
        arr_a = sub_a.loc[comuns].to_numpy(dtype=float)
        arr_b = sub_b.loc[comuns].to_numpy(dtype=float)
        rho, p_rho = spearmanr(arr_a, arr_b)
        tau, p_tau = kendalltau(arr_a, arr_b)
        linhas.append({
            "dataset_pasta": ds, "fonte": mapear_fonte(ds),
            "n_detectores_comuns": len(comuns),
            "spearman_rho": round(float(rho), 4), "spearman_p": round(float(p_rho), 6),
            "kendall_tau": round(float(tau), 4), "kendall_p": round(float(p_tau), 6),
        })

    if datasets_descartados:
        warnings.warn(
            f"[Correlacao por dataset '{rotulo_a}' vs '{rotulo_b}'/{metrica}] "
            f"{len(datasets_descartados)} dataset(s) descartado(s) por ter menos "
            f"de {min_detectores_comuns} detectores em comum: {datasets_descartados}"
        )

    return pd.DataFrame(linhas)


def correlacao_rank_por_dataset(df_split: pd.DataFrame, cenario_a: str, cenario_b: str,
                                 metrica: str, min_detectores_comuns: int = 3,
                                 comparacoes: list = COMPARACOES) -> pd.DataFrame:
    """Correlacao por dataset entre dois cenarios do MESMO pipeline
    ASOC (ex.: 'drift' vs 'catch24_drift'), ambos extraidos de df_split.
    Ver docstring de _correlacionar_valores_por_dataset para detalhes
    do calculo."""
    val_a, _ = extrair_valores_cenario_split(df_split, cenario_a, comparacoes)
    val_b, _ = extrair_valores_cenario_split(df_split, cenario_b, comparacoes)
    val_a = val_a[val_a["detector"].isin(DETECTORES)]
    val_b = val_b[val_b["detector"].isin(DETECTORES)]
    return _correlacionar_valores_por_dataset(
        val_a, val_b, cenario_a, cenario_b, metrica, min_detectores_comuns,
    )


def resumir_correlacao_datasets(df_corr: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Resumo agregado da tabela de correlacoes por dataset (saida de
    correlacao_rank_por_dataset). Reporta DUAS visoes lado a lado:

    1. 'todos_splits' -- os ate 16 datasets, tratados individualmente
       (justificativa: canais de sensor da mesma fonte podem ter
       dinamicas de drift proprias, ver docstring da secao acima).
    2. 'fontes_split_unico' -- restrito aos datasets que SAO, sozinhos,
       uma fonte independente (excluindo os splits de P2, W, sp/sw),
       como checagem de sensibilidade: se a conclusao muda muito entre
       as duas visoes, a inclusao dos splits multiplos esta pesando
       na media mais do que deveria.

    Para cada visao, alem de media/mediana/desvio, roda um teste de
    Wilcoxon signed-rank do coeficiente de Spearman contra 0 -- uso
    EXPLORATORIO (o teste sofre da mesma pseudorreplicacao potencial
    na visao 'todos_splits'; a visao 'fontes_split_unico' e a mais
    defensavel para esse teste especificamente).
    """
    if df_corr.empty:
        return pd.DataFrame()

    def _resumo(sub: pd.DataFrame, rotulo: str) -> dict:
        if sub.empty:
            return {"visao": rotulo, "n_datasets": 0}
        rhos = sub["spearman_rho"].to_numpy(dtype=float)
        taus = sub["kendall_tau"].to_numpy(dtype=float)
        resultado = {
            "visao": rotulo, "n_datasets": len(sub),
            "spearman_rho_media": round(float(np.mean(rhos)), 4),
            "spearman_rho_mediana": round(float(np.median(rhos)), 4),
            "spearman_rho_desvio": round(float(np.std(rhos, ddof=0)), 4),
            "kendall_tau_media": round(float(np.mean(taus)), 4),
            "pct_positivos": round(float(np.mean(rhos > 0) * 100), 1),
        }
        rhos_nao_zero = rhos[rhos != 0]
        if len(rhos_nao_zero) >= 1:
            try:
                stat, p_val = wilcoxon(rhos_nao_zero, alternative="two-sided")
                resultado["wilcoxon_p_rho_vs_0"] = round(float(p_val), 6)
                resultado["sig_rho_diferente_de_0"] = bool(p_val < alpha)
            except Exception as e:
                warnings.warn(f"[Resumo correlacao] Wilcoxon falhou: {e}")
                resultado["wilcoxon_p_rho_vs_0"] = np.nan
                resultado["sig_rho_diferente_de_0"] = False
        else:
            resultado["wilcoxon_p_rho_vs_0"] = np.nan
            resultado["sig_rho_diferente_de_0"] = False
        return resultado

    fontes_split_unico = {
        fonte for fonte, splits in GRUPOS_FONTE.items()
    }
    sub_unico = df_corr[~df_corr["fonte"].isin(fontes_split_unico)]

    linhas = [
        _resumo(df_corr, "todos_splits"),
        _resumo(sub_unico, "fontes_split_unico"),
    ]
    return pd.DataFrame(linhas)


def imprimir_correlacao_datasets(df_corr: pd.DataFrame, df_resumo: pd.DataFrame,
                                  cenario_a: str, cenario_b: str,
                                  rotulo_metrica: str) -> None:
    if df_corr.empty:
        print(f"[AVISO] Correlacao por dataset ('{cenario_a}' vs '{cenario_b}', "
              f"{rotulo_metrica}) vazia -- sem dados suficientes.")
        return
    print(f"\n[Transferencia de ranking por dataset -- '{cenario_a}' (barato) vs "
          f"'{cenario_b}' (caro), {rotulo_metrica}]")
    print("(correlacao de Spearman/Kendall entre os 9 detectores, DENTRO de "
          "cada dataset -- uso descritivo/exploratorio)")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_corr.sort_values("spearman_rho", ascending=False).to_string(index=False))
        print()
        print(df_resumo.to_string(index=False))


# ============================================================
# CORRELAÇÃO INTRA-FONTE: OS SPLITS DE UMA MESMA FONTE "VOTAM
# PARECIDO" SOBRE QUAL DETECTOR É MELHOR?
# ============================================================
# Diferente da secao anterior (que correlaciona DOIS CENARIOS para o
# MESMO dataset), esta secao correlaciona DOIS SPLITS da MESMA FONTE
# (ex.: P2-Hand vs P2-Chest) DENTRO DO MESMO cenario. Esta e a forma
# DIRETA de testar a suposicao por tras da pseudorreplicacao: se os
# splits de uma fonte tendem a concordar sobre qual detector e melhor
# (rho alto), isso sustenta trata-los como um unico "voto" repetido;
# se discordam (rho baixo/negativo), sustenta trata-los como unidades
# distinguiveis para analises descritivas como a de transferencia
# acima.
#
# Uso EXPLORATORIO/DIAGNOSTICO -- nao substitui nem altera o teste
# oficial Wilcoxon+BH (que permanece pareado por fonte independente).
# ============================================================

def _correlacionar_pares_intra_fonte(val: pd.DataFrame, rotulo_cenario: str, metrica: str,
                                      min_detectores_comuns: int = 3) -> pd.DataFrame:
    """Nucleo comum: recebe uma tabela ['dataset_pasta','detector','val']
    de UM cenario/pipeline (ASOC ou MiniRocket) e calcula, para cada
    fonte com MULTIPLOS splits (P2, W, sp_sw), a correlacao de
    Spearman/Kendall PAR A PAR entre os splits dessa fonte (nao entre
    cenarios diferentes do mesmo split, como na secao anterior).

    Retorna DataFrame ['cenario', 'fonte', 'dataset_1', 'dataset_2',
    'n_detectores_comuns', 'spearman_rho', 'spearman_p', 'kendall_tau',
    'kendall_p'] -- uma linha por PAR de splits da mesma fonte.
    """
    colunas_vazias = [
        "cenario", "fonte", "dataset_1", "dataset_2", "n_detectores_comuns",
        "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
    ]
    if val.empty:
        warnings.warn(f"[Correlacao intra-fonte] '{rotulo_cenario}' sem dados.")
        return pd.DataFrame(columns=colunas_vazias)

    val = val.copy()
    val["fonte"] = val["dataset_pasta"].map(mapear_fonte)

    linhas = []
    for fonte, grupo_fonte in val.groupby("fonte"):
        datasets = sorted(grupo_fonte["dataset_pasta"].unique())
        if len(datasets) < 2:
            continue  # fonte de split unico -- nao ha par para comparar
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                ds1, ds2 = datasets[i], datasets[j]
                s1 = grupo_fonte.loc[grupo_fonte["dataset_pasta"] == ds1].set_index("detector")["val"]
                s2 = grupo_fonte.loc[grupo_fonte["dataset_pasta"] == ds2].set_index("detector")["val"]
                comuns = sorted(set(s1.index) & set(s2.index))
                if len(comuns) < min_detectores_comuns:
                    continue
                arr1 = s1.loc[comuns].to_numpy(dtype=float)
                arr2 = s2.loc[comuns].to_numpy(dtype=float)
                rho, p_rho = spearmanr(arr1, arr2)
                tau, p_tau = kendalltau(arr1, arr2)
                linhas.append({
                    "cenario": rotulo_cenario, "fonte": fonte,
                    "dataset_1": ds1, "dataset_2": ds2,
                    "n_detectores_comuns": len(comuns),
                    "spearman_rho": round(float(rho), 4), "spearman_p": round(float(p_rho), 6),
                    "kendall_tau": round(float(tau), 4), "kendall_p": round(float(p_tau), 6),
                })
    return pd.DataFrame(linhas, columns=colunas_vazias)


def correlacao_rank_intra_fonte(df_split: pd.DataFrame, cenario_bruto: str, metrica: str,
                                 min_detectores_comuns: int = 3,
                                 comparacoes: list = COMPARACOES) -> pd.DataFrame:
    """Correlacao intra-fonte para um cenario do pipeline ASOC (ex.:
    'drift', 'catch24_drift'). Ver docstring de
    _correlacionar_pares_intra_fonte."""
    val, _ = extrair_valores_cenario_split(df_split, cenario_bruto, comparacoes)
    val = val[val["detector"].isin(DETECTORES)]
    return _correlacionar_pares_intra_fonte(val, cenario_bruto, metrica, min_detectores_comuns)


def correlacao_rank_intra_fonte_minirocket(cenario_mr: str, arquivo: str, metrica: str,
                                            min_detectores_comuns: int = 3) -> pd.DataFrame:
    """Correlacao intra-fonte para um cenario do pipeline MiniRocket
    ('c1' ou 'c2'), lido diretamente de MINIROCKET_PASTA_RAIZ. Como o
    MiniRocket ainda nao foi finalizado, fontes com poucos splits
    rodados ate agora podem nao ter nenhum par disponivel -- isso e
    esperado e nao interrompe a execucao."""
    val = extrair_valores_minirocket_split(cenario_mr, arquivo)
    return _correlacionar_pares_intra_fonte(val, f"minirocket_{cenario_mr}", metrica, min_detectores_comuns)


def resumir_correlacao_intra_fonte(df_pares: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Resume a tabela de pares intra-fonte: media/mediana/desvio do
    rho, % de pares positivos, e um teste de Wilcoxon signed-rank do
    rho contra 0 (H1: os splits de uma mesma fonte tendem a concordar
    sobre qual detector e melhor). Calculado POR CENARIO e, dentro de
    cada cenario, tambem quebrado POR FONTE (quando ha mais de um par
    disponivel naquela fonte especificamente, ex.: W com 5 splits gera
    10 pares).

    ATENCAO -- os pares dentro da mesma fonte NAO sao mutuamente
    independentes entre si (ex.: os 10 pares de W compartilham 5
    splits, entao um teste agregando todos os pares de todas as
    fontes de uma vez tem menos informacao nova do que o numero de
    pares sugere). Por isso o resumo tambem e quebrado por fonte, onde
    essa nao-independencia fica visivel e mais facil de interpretar.
    """
    if df_pares.empty:
        return pd.DataFrame()

    linhas = []
    for cenario, grupo_cen in df_pares.groupby("cenario"):
        rhos = grupo_cen["spearman_rho"].to_numpy(dtype=float)
        linha = {
            "cenario": cenario, "nivel": "todas_fontes_juntas", "fonte": "(todas)",
            "n_pares": len(rhos),
            "spearman_rho_media": round(float(np.mean(rhos)), 4),
            "spearman_rho_mediana": round(float(np.median(rhos)), 4),
            "pct_positivos": round(float(np.mean(rhos > 0) * 100), 1),
        }
        rhos_nz = rhos[rhos != 0]
        if len(rhos_nz) >= 1:
            try:
                _, p_val = wilcoxon(rhos_nz, alternative="two-sided")
                linha["wilcoxon_p"] = round(float(p_val), 6)
                linha["sig_rho_diferente_de_0"] = bool(p_val < alpha)
            except Exception as e:
                warnings.warn(f"[Resumo intra-fonte] Wilcoxon falhou: {e}")
                linha["wilcoxon_p"] = np.nan
                linha["sig_rho_diferente_de_0"] = False
        else:
            linha["wilcoxon_p"] = np.nan
            linha["sig_rho_diferente_de_0"] = False
        linhas.append(linha)

        for fonte, grupo_fonte in grupo_cen.groupby("fonte"):
            rhos_f = grupo_fonte["spearman_rho"].to_numpy(dtype=float)
            linhas.append({
                "cenario": cenario, "nivel": "por_fonte", "fonte": fonte,
                "n_pares": len(rhos_f),
                "spearman_rho_media": round(float(np.mean(rhos_f)), 4),
                "spearman_rho_mediana": round(float(np.median(rhos_f)), 4),
                "pct_positivos": round(float(np.mean(rhos_f > 0) * 100), 1),
                "wilcoxon_p": np.nan, "sig_rho_diferente_de_0": False,
            })

    return pd.DataFrame(linhas)


def imprimir_correlacao_intra_fonte(df_pares: pd.DataFrame, df_resumo: pd.DataFrame,
                                     rotulo_metrica: str) -> None:
    if df_pares.empty:
        print(f"[AVISO] Correlacao intra-fonte ({rotulo_metrica}) vazia -- "
              f"sem pares de splits disponiveis (fontes de split unico "
              f"nao geram pares).")
        return
    print(f"\n[Correlacao INTRA-FONTE (splits da mesma fonte concordam sobre "
          f"o melhor detector?) -- {rotulo_metrica}]")
    print("(rho alto = splits votam parecido, sustenta pseudorreplicacao | "
          "rho baixo/negativo = splits discordam, sustenta trata-los como "
          "unidades distinguiveis)")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_pares.sort_values(["cenario", "fonte", "spearman_rho"],
                                    ascending=[True, True, False]).to_string(index=False))
        print()
        print(df_resumo.to_string(index=False))


# ============================================================
# VISÃO MISTA: AGREGAÇÃO DIFERENCIADA POR FONTE NA ANÁLISE DE
# TRANSFERÊNCIA (motivada pela correlação intra-fonte acima)
# ============================================================
# A correlacao intra-fonte mostrou que W tem concordancia real entre
# seus 5 splits (rho mediano ~0.5, quase sempre positivo), enquanto P2
# e sp_sw tem concordancia fraca/negativa. Isso justifica agregar
# SOMENTE W antes de calcular a correlacao de transferencia (drift vs
# catch22+drift / MiniRocket), mantendo P2 e sp_sw como unidades
# distintas -- ao inves do tudo-ou-nada usado em
# resumir_correlacao_datasets ('todos_splits' vs 'fontes_split_unico').
# ============================================================

GRUPOS_FONTE_MISTO = {
    "W": GRUPOS_FONTE["W"],
    # P2 e sp_sw propositalmente NAO entram aqui -- permanecem como
    # splits individuais, ver justificativa acima.
}


def agregar_valores_minirocket_seletivo(val: pd.DataFrame, grupos_agregar: dict,
                                         meio: str = MEIO_AGREGACAO) -> pd.DataFrame:
    """Equivalente a agregar_por_fonte_seletivo, mas para tabelas no
    formato ['dataset_pasta','detector','val'] (saida de
    extrair_valores_minirocket_split), usado para manter a mesma
    agregacao diferenciada tambem nas comparacoes com o MiniRocket."""
    func = _FUNC_AGREGACAO[meio]
    df = val.copy()

    def _mapear_seletivo(nome: str) -> str:
        for fonte, splits in grupos_agregar.items():
            if nome in splits:
                return fonte
        return nome

    df["dataset_pasta"] = df["dataset_pasta"].map(_mapear_seletivo)
    agregada = (
        df.groupby(["dataset_pasta", "detector"])["val"]
        .apply(lambda s: func(s.to_numpy(dtype=float)))
        .reset_index()
    )
    agregada.columns = ["dataset_pasta", "detector", "val"]
    return agregada


def resumir_correlacao_visao_mista(df_corr: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Resumo simples para a visao mista: TODAS as linhas de df_corr ja
    representam a unidade considerada apropriada (fonte agregada onde
    ha evidencia de redundancia -- W --, split individual onde nao ha
    -- P2, sp_sw, e as fontes de split unico). Diferente de
    resumir_correlacao_datasets, NAO ha duas visoes aqui -- ja e a
    visao unica considerada mais defensavel."""
    if df_corr.empty:
        return pd.DataFrame()
    rhos = df_corr["spearman_rho"].to_numpy(dtype=float)
    linha = {
        "n_unidades": len(rhos),
        "spearman_rho_media": round(float(np.mean(rhos)), 4),
        "spearman_rho_mediana": round(float(np.median(rhos)), 4),
        "spearman_rho_desvio": round(float(np.std(rhos, ddof=0)), 4),
        "pct_positivos": round(float(np.mean(rhos > 0) * 100), 1),
    }
    rhos_nz = rhos[rhos != 0]
    if len(rhos_nz) >= 1:
        try:
            _, p_val = wilcoxon(rhos_nz, alternative="two-sided")
            linha["wilcoxon_p_rho_vs_0"] = round(float(p_val), 6)
            linha["sig_rho_diferente_de_0"] = bool(p_val < alpha)
        except Exception as e:
            warnings.warn(f"[Resumo visao mista] Wilcoxon falhou: {e}")
            linha["wilcoxon_p_rho_vs_0"] = np.nan
            linha["sig_rho_diferente_de_0"] = False
    else:
        linha["wilcoxon_p_rho_vs_0"] = np.nan
        linha["sig_rho_diferente_de_0"] = False
    return pd.DataFrame([linha])


def imprimir_correlacao_visao_mista(df_corr: pd.DataFrame, df_resumo: pd.DataFrame,
                                     rotulo_comparacao: str, rotulo_metrica: str) -> None:
    if df_corr.empty:
        print(f"[AVISO] Visao mista ('{rotulo_comparacao}', {rotulo_metrica}) "
              f"vazia -- sem dados suficientes.")
        return
    print(f"\n[Visao MISTA (W agregado, P2/sp_sw individuais) -- "
          f"'{rotulo_comparacao}', {rotulo_metrica}]")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_corr.sort_values("spearman_rho", ascending=False).to_string(index=False))
        print()
        print(df_resumo.to_string(index=False))


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

    # NOME_PARA_COL reaproveitado do nivel de modulo (definido perto de DETECTORES)

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

# ============================================================
# TABELA DO ARTIGO: delta pareado + p ajustado (BH), por comparacao
# ============================================================
TABELA_COMPARACOES = [("C0_vs_C1", "C0 vs.\\ C1"),
                      ("C1_vs_C2", "C1 vs.\\ C2"),
                      ("C2_vs_C3", "C2 vs.\\ C3"),
                      ("C2raw_vs_C3", "C2.raw vs.\\ C3")]
TABELA_ORDEM_DETECTORES = ["ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
                           "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED"]
TABELA_INCLUIR_CONTAGENS = False   # True: acrescenta "n+/n-" em cada celula
TABELA_SAIDA_TEX = os.path.join(PASTA_RAIZ, "tabela_delta_pareado.tex")
TABELA_SAIDA_CSV = os.path.join(PASTA_RAIZ, "tabela_delta_pareado.csv")


def _fmt_num(v: float, casas: int = 2) -> str:
    txt = f"{v:.{casas}f}"
    return txt.replace("-", "$-$") if v < 0 else txt


def _celula_latex(row) -> str:
    delta = _fmt_num(row["cliff_delta"])
    if row["p_bh"] <= ALPHA_FDR:
        delta = f"\\textbf{{{delta}}}"
    cel = f"{delta}~{{\\scriptsize({row['p_bh']:.3f})}}"
    if TABELA_INCLUIR_CONTAGENS:
        cel += f"~{{\\scriptsize[{int(row['n_pos'])}/{int(row['n_neg'])}]}}"
    return cel


def gerar_tabela_delta(resultados_fonte: dict, caminho_tex: str = TABELA_SAIDA_TEX,
                       caminho_csv: str = TABELA_SAIDA_CSV) -> None:
    """Gera a tabela (LaTeX) com delta pareado e p ajustado por BH para as
    comparacoes C0 vs C1, C1 vs C2 e C2 vs C3, em F1 e acuracia, e um CSV
    completo (Tabela S-X: p bruto, p ajustado, n+/n-)."""
    linhas_csv = []
    for comp, _ in TABELA_COMPARACOES:
        for sufixo in ["f1", "acc"]:
            df = resultados_fonte.get(comp, {}).get(sufixo)
            if df is None:
                continue
            for _, r in df.iterrows():
                linhas_csv.append({
                    "comparacao": comp, "metrica": sufixo,
                    "detector": NOME_PARA_COL.get(r["detector"], r["detector"]),
                    "n_fontes": int(r["n_unidades"]), "n_pos": int(r["n_pos"]),
                    "n_neg": int(r["n_neg"]), "delta": r["cliff_delta"],
                    "p_bruto": r["p_value"], "p_bh": r["p_bh"],
                    "significativo_bh": bool(r["sig_bh"]),
                })
    df_csv = pd.DataFrame(linhas_csv)
    df_csv.to_csv(caminho_csv, index=False)

    n_fontes = sorted(df_csv["n_fontes"].unique()) if not df_csv.empty else []
    n_txt = str(n_fontes[0]) if len(n_fontes) == 1 else "/".join(map(str, n_fontes))

    ncomp = len(TABELA_COMPARACOES)
    out = []
    out.append("% Gerado por teste_reais_agr_datasets.py -- nao editar a mao")
    out.append("\\begin{table}[!t]")
    out.append("\\centering")
    out.append("\\caption{Paired dominance statistic $\\delta = (n_{+} - n_{-})/N$ between "
               f"scenarios ($N={n_txt}$ independent real-world sources), with "
               "Benjamini--Hochberg-adjusted $p$-values in parentheses. "
               "\\textbf{Bold}: $p_{\\mathrm{adj}} \\le 0.05$. Positive $\\delta$ "
               "favours the second scenario"
               + ("; $[n_{+}/n_{-}]$: number of sources favouring each scenario" if TABELA_INCLUIR_CONTAGENS else "")
               + ".}")
    out.append("\\label{tab:delta_real}")
    out.append("\\setlength{\\tabcolsep}{2.3pt}")
    out.append("\\begin{tabular}{l" + "rr" * ncomp + "}")
    out.append("\\hline")
    out.append(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{rot}}}" for _, rot in TABELA_COMPARACOES) + " \\\\")
    out.append("".join(f"\\cline{{{2 + 2*i}-{3 + 2*i}}}" for i in range(ncomp)))
    out.append("Det. & " + " & ".join(["\\multicolumn{1}{c}{F1} & \\multicolumn{1}{c}{Acc}"] * ncomp) + " \\\\")
    out.append("\\hline")
    for det in TABELA_ORDEM_DETECTORES:
        cels = []
        for comp, _ in TABELA_COMPARACOES:
            for sufixo in ["f1", "acc"]:
                df = resultados_fonte.get(comp, {}).get(sufixo)
                linha = df[df["detector"] == det] if df is not None else pd.DataFrame()
                cels.append(_celula_latex(linha.iloc[0]) if not linha.empty else "--")
        out.append(f"{NOME_PARA_COL.get(det, det)} & " + " & ".join(cels) + " \\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    with open(caminho_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n[OK] Tabela LaTeX salva em: {caminho_tex}")
    print(f"[OK] Tabela completa (Tabela S-X) salva em: {caminho_csv}")


# ============================================================
# TABELA S-4 (material suplementar): C2.raw com a configuração do C2 vs. C3
#   Mesmo formato da tab:delta_real (9 detectores, F1 e acurácia), com as
#   contagens n+/n- em cada célula. Complementa a coluna C2.raw vs. C3 do
#   artigo (configuração do C3) como verificação de robustez.
# ============================================================
TAB_C2RAW_C2_NUMERO = "S-4"
TAB_C2RAW_C2_COMP = "C2rawC2_vs_C3"
TAB_C2RAW_C2_SAIDA_TEX = os.path.join(PASTA_RAIZ, "tabela_c2raw_conf_c2.tex")
TAB_C2RAW_C2_SAIDA_CSV = os.path.join(PASTA_RAIZ, "tabela_c2raw_conf_c2.csv")


def gerar_tabela_c2raw_c2(resultados_fonte: dict, caminho_tex: str = TAB_C2RAW_C2_SAIDA_TEX,
                          caminho_csv: str = TAB_C2RAW_C2_SAIDA_CSV) -> None:
    res = resultados_fonte.get(TAB_C2RAW_C2_COMP, {})
    if not res:
        print(f"[AVISO] Tabela {TAB_C2RAW_C2_NUMERO} não gerada: sem resultados de {TAB_C2RAW_C2_COMP}")
        return
    linhas_csv, corpo, ns = [], [], set()
    for det in TABELA_ORDEM_DETECTORES:
        cels = []
        for sufixo in ["f1", "acc"]:
            df = res.get(sufixo)
            linha = df[df["detector"] == det] if df is not None else pd.DataFrame()
            if linha.empty:
                cels.append("--")
                continue
            r = linha.iloc[0]
            ns.add(int(r["n_unidades"]))
            delta = _fmt_num(r["cliff_delta"])
            if r["p_bh"] <= ALPHA_FDR:
                delta = f"\\textbf{{{delta}}}"
            cels.append(f"{delta}~{{\\scriptsize({r['p_bh']:.3f})}}"
                        f"~{{\\scriptsize[{int(r['n_pos'])}/{int(r['n_neg'])}]}}")
            linhas_csv.append({"metrica": sufixo, "detector": NOME_PARA_COL.get(det, det),
                               "n_fontes": int(r["n_unidades"]), "n_pos": int(r["n_pos"]),
                               "n_neg": int(r["n_neg"]), "delta": r["cliff_delta"],
                               "p_bruto": r["p_value"], "p_bh": r["p_bh"],
                               "significativo_bh": bool(r["sig_bh"])})
        corpo.append(f"{NOME_PARA_COL.get(det, det)} & " + " & ".join(cels) + " \\\\")
    pd.DataFrame(linhas_csv).to_csv(caminho_csv, index=False)

    n_txt = "/".join(map(str, sorted(ns)))
    legenda = ("Paired dominance statistic $\\delta = (n_{+} - n_{-})/N$ between C2.raw, evaluated with "
               f"the configuration selected for C2, and C3 ($N={n_txt}$ independent real-world sources), "
               "with Benjamini--Hochberg-adjusted $p$-values in parentheses and $[n_{+}/n_{-}]$ in "
               "brackets. \\textbf{Bold}: $p_{\\mathrm{adj}} \\le 0.05$. Positive $\\delta$ favours C3.")
    tex = "\n".join([
        "% Gerada por teste_reais_agr_datasets.py -- nao editar a mao",
        "\\begin{minipage}{\\linewidth}",
        f"\\noindent\\sffamily\\small\\textbf{{\\color{{scolor}}Table {TAB_C2RAW_C2_NUMERO}.}}\\\\ " + legenda,
        "\\label{tab:c2raw_conf_c2_supp}",
        "\\par\\normalfont\\normalsize",
        "\\begin{center}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "\\textbf{Det.} & \\multicolumn{1}{c}{\\textbf{F1}} & \\multicolumn{1}{c}{\\textbf{Acc}} \\\\",
        "\\hline",
        *corpo,
        "\\hline",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{minipage}",
    ])
    with open(caminho_tex, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"[OK] Tabela {TAB_C2RAW_C2_NUMERO} salva em: {caminho_tex}")
    print(f"[OK] Valores da Tabela {TAB_C2RAW_C2_NUMERO} salvos em: {caminho_csv}")


# ============================================================
# FIGURA DE GANHOS POR DATASET (estilo da Figura 3 do PAKDD)
#   (a) ganho sobre C0: C2 (catch22) e C1 (pior/mediana/melhor detector)
#   (b) ganho de C3 sobre C2 (pior/mediana/melhor detector)
# ============================================================
FIG_GANHOS_SAIDA = os.path.join(PASTA_RAIZ, "figura_ganhos_por_dataset")   # .png e .pdf
FIG_GANHOS_CSV = os.path.join(PASTA_RAIZ, "ganhos_por_dataset_f1.csv")

# pasta do dataset -> (rotulo no artigo, fonte). A ordem define a ordem no eixo;
# datasets da mesma fonte ficam juntos; o contraexemplo (RO) fica no fim.
FIG_GANHOS_DATASETS = [
    ("pamap2_ankle", "P2-Ankle", "P2"), ("pamap2_hand", "P2-Hand", "P2"),
    ("pamap2_chest", "P2-Chest", "P2"), ("usc_had", "USC", "USC"),
    ("ward_left_wrist", "W-LWrist", "W"), ("ward_waist", "W-Waist", "W"),
    ("ward_right_ankle", "W-RAnkle", "W"), ("ward_right_wrist", "W-RWrist", "W"),
    ("ward_left_ankle", "W-LAnkle", "W"), ("mhealth", "MH", "MH"),
    ("rs", "DR", "DR"), ("smartphone", "SmPh", "SmPh"), ("gait", "Gait", "Gait"),
    ("sp_har", "Sp", "Sp/Sw"), ("sw_har", "Sw", "Sp/Sw"), ("occ", "RO", "RO"),
]
FIG_GANHOS_CONTRAEXEMPLO = "occ"      # separado por linha tracejada (None para desligar)

COR_C1, COR_C2, COR_C3 = "#2a9d78", "#d97a2b", "#3767b8"   # paleta validada (CVD)
COR_C2RAW = "#59636e"            # neutro: controle, fora da paleta categórica
# controle C2.raw mostrado no painel (b): configuração do melhor detector do C3
# (a mesma comparação da tabela do artigo); None para não mostrar
FIG_GANHOS_COMP_C2RAW = "C2raw_vs_C3"


def gerar_figura_ganhos(df_split_f1: pd.DataFrame, caminho_base: str = FIG_GANHOS_SAIDA,
                        caminho_csv: str = FIG_GANHOS_CSV) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = df_split_f1.copy()
    d["ganho_pp"] = 100 * (d["val_B"] - d["val_A"])

    def faixa(comp):
        x = d[d["comparacao"] == comp]
        r = x.groupby("dataset_pasta")["ganho_pp"].agg(pior="min", mediana="median",
                                                        melhor="max", n_detectores="count")
        r["det_pior"] = x.loc[x.groupby("dataset_pasta")["ganho_pp"].idxmin()].set_index("dataset_pasta")["detector"]
        r["det_melhor"] = x.loc[x.groupby("dataset_pasta")["ganho_pp"].idxmax()].set_index("dataset_pasta")["detector"]
        return r

    c1 = faixa("C0_vs_C1")
    c3 = faixa("C2_vs_C3")
    c2 = d[d["comparacao"] == "C0_vs_C2"].groupby("dataset_pasta")["ganho_pp"].mean()

    # C2.raw em relação ao C2: parte do ganho do C3 que as variáveis originais
    # explicam sozinhas (C2.raw é o lado A da comparação C2raw_vs_C3)
    f1_c2 = d[d["comparacao"] == "C0_vs_C2"].groupby("dataset_pasta")["val_B"].mean()
    xr = d[d["comparacao"] == FIG_GANHOS_COMP_C2RAW] if FIG_GANHOS_COMP_C2RAW else d.iloc[0:0]
    c2raw = (100 * (xr.groupby("dataset_pasta")["val_A"].mean() - f1_c2)).dropna()
    if FIG_GANHOS_COMP_C2RAW and c2raw.empty:
        warnings.warn(f"Sem pares de {FIG_GANHOS_COMP_C2RAW}: C2.raw fora da figura")

    # C3 em relação a C0 (C3 absoluto do par C2_vs_C3 menos o C0 do par C0_vs_C2),
    # para indicar no painel (b) os datasets em que nenhum detector do C3 supera o C0
    f1_c0 = d[d["comparacao"] == "C0_vs_C2"].groupby("dataset_pasta")["val_A"].mean()
    x3 = d[d["comparacao"] == "C2_vs_C3"]
    c3_c0 = (100 * (x3["val_B"] - x3["dataset_pasta"].map(f1_c0))).groupby(x3["dataset_pasta"]) \
        .agg(pior="min", mediana="median", melhor="max")

    ordem = [x for x in FIG_GANHOS_DATASETS if x[0] in set(c1.index) | set(c3.index)]
    faltando = (set(c1.index) | set(c3.index)) - {x[0] for x in ordem}
    if faltando:
        warnings.warn(f"Datasets sem rotulo em FIG_GANHOS_DATASETS (fora da figura): {sorted(faltando)}")

    pd.concat([c1.add_prefix("C1_menos_C0_"), c3.add_prefix("C3_menos_C2_"),
               c2.rename("C2_menos_C0"), c3_c0.add_prefix("C3_menos_C0_"),
               c2raw.rename("C2raw_menos_C2")], axis=1).reindex([o[0] for o in ordem]) \
      .rename_axis("dataset_pasta").to_csv(caminho_csv)

    x = np.arange(len(ordem))
    TINTA, MUTED, GRADE, FAIXA = "#1f2328", "#59636e", "#d0d7de", "#eef0f2"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA, "ytick.color": MUTED})
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    # grupos de fontes com mais de um dataset: faixa cinza + rotulo
    grupos = []
    for i, (_, _, fonte) in enumerate(ordem):
        if grupos and grupos[-1][0] == fonte:
            grupos[-1][2] = i
        else:
            grupos.append([fonte, i, i])

    def moldura(ax, rotular):
        for fonte, ini, fim in grupos:
            if fim > ini:
                ax.axvspan(ini - 0.5, fim + 0.5, color=FAIXA, zorder=0, lw=0)
                if rotular:
                    ax.text((ini + fim) / 2, 1.0, fonte, transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8.5, style="italic", color=MUTED)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        if FIG_GANHOS_CONTRAEXEMPLO:
            idx = [i for i, o in enumerate(ordem) if o[0] == FIG_GANHOS_CONTRAEXEMPLO]
            if idx:
                ax.axvline(idx[0] - 0.5, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.grid(axis="y", color=GRADE, lw=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)

    def faixa_plot(ax, r, pos, cor, marcador):
        for xi, (pasta, _, _) in zip(pos, ordem):
            if pasta not in r.index:
                continue
            row = r.loc[pasta]
            ax.plot([xi, xi], [row["pior"], row["melhor"]], color=cor, lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(xi, row["mediana"], marcador, ms=7, color=cor, mec="white", mew=1.1, zorder=3)

    # (a) ganho sobre C0
    moldura(axa, rotular=True)
    faixa_plot(axa, c1, x + 0.13, COR_C1, "^")
    axa.plot(x[[o[0] in c2.index for o in ordem]] - 0.13,
             [c2[o[0]] for o in ordem if o[0] in c2.index],
             "s", ms=6.5, color=COR_C2, mec="white", mew=1.1, zorder=3)
    axa.set_ylabel("$\\Delta$F1 vs. C0 (p.p.)")
    axa.text(0.0, 1.06, "(a)", transform=axa.transAxes, fontsize=9.5, fontweight="bold", color=TINTA)

    # (b) ganho de C3 sobre C2
    moldura(axb, rotular=False)
    desl = 0.13 if not c2raw.empty else 0.0
    faixa_plot(axb, c3, x + desl, COR_C3, "o")
    if not c2raw.empty:
        axb.plot(x[[o[0] in c2raw.index for o in ordem]] - desl,
                 [c2raw[o[0]] for o in ordem if o[0] in c2raw.index],
                 "D", ms=5.5, color=COR_C2RAW, mec="white", mew=1.0, zorder=3)
    # datasets em que o melhor detector do C3 continua abaixo do C0: anotar C3 - C0
    for xi, (pasta, _, _) in zip(x, ordem):
        if pasta in c3_c0.index and pasta in c3.index and c3_c0.loc[pasta, "melhor"] < 0:
            r = c3_c0.loc[pasta]
            axb.annotate(f"vs. C0:\n{r['pior']:.1f} to\n{r['melhor']:.1f}".replace("-", "\u2212"),
                         (xi, max(c3.loc[pasta, "melhor"], c2raw.get(pasta, -np.inf))), xytext=(0, 6), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, color=MUTED, linespacing=1.1)
    axb.set_ylabel("$\\Delta$F1 vs. C2 (p.p.)")
    axb.text(0.0, 1.02, "(b)", transform=axb.transAxes, fontsize=9.5, fontweight="bold", color=TINTA)
    axb.set_xticks(x)
    axb.set_xticklabels([o[1] for o in ordem], rotation=45, ha="right")
    axb.set_xlim(-0.6, len(ordem) - 0.4)

    # com duas colunas, a primeira traz as configurações únicas (C2, C2.raw)
    # e a segunda os cenários com detector (C1, C3)
    legenda = [Line2D([0], [0], marker="s", ls="none", color=COR_C2, ms=7, label="C2 (catch22)")]
    if not c2raw.empty:
        legenda.append(Line2D([0], [0], marker="D", ls="none", color=COR_C2RAW, ms=6,
                              label="C2.raw (catch22 + original variables)"))
    legenda += [
        Line2D([0], [0], marker="^", color=COR_C1, lw=2, ms=7, label="C1 (median, bar: range across detectors)"),
        Line2D([0], [0], marker="o", color=COR_C3, lw=2, ms=7, label="C3 (median, bar: range across detectors)"),
    ]
    fig.legend(handles=legenda, loc="upper center", ncol=2 if len(legenda) > 3 else 3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8.5, handlelength=1.8)
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.09, right=0.99)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{caminho_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura de ganhos salva em: {caminho_base}.png/.pdf")
    print(f"[OK] Valores da figura salvos em: {caminho_csv}")


# ============================================================
# TABELA S-9 (material suplementar): F1 macro por dataset
#   C0, C2.raw (duas variantes) e C2: média entre os folds de teste (sem DP: C1 e C3 não têm um DP comparável)
#   C1 e C3: mediana entre os nove detectores [pior, melhor]
#   Melhor detector de C1 e de C3 em cada dataset (a posteriori); o valor
#   desse detector é o "melhor" entre colchetes
# Mesma fonte dos testes: df_f1_all.csv, coluna COLUNA_METRICA (RF), média
# entre os folds de cada dataset (splits de P2, W e Sp/Sw em linhas próprias).
# ============================================================
TAB_F1_NUMERO = "S-15"
# controle C2.raw nas duas variantes: configuração do melhor detector do C3 e do C2
TAB_F1_CENARIOS_C2RAW = [("C2rawC3", "catch24_raw_c3"), ("C2rawC2", "catch24_raw_c2")]
TAB_F1_SAIDA_TEX = os.path.join(PASTA_RAIZ, "tabela_f1_por_dataset.tex")
TAB_F1_SAIDA_CSV = os.path.join(PASTA_RAIZ, "tabela_f1_por_dataset.csv")


def _f1_por_fold(pasta: str, cenario: str, detector: str):
    """F1 (%) por fold de um cenário, ou None se o arquivo não existir."""
    df = load_csv_flexivel(pasta, cenario, detector, "df_f1_all.csv")
    if df.empty:
        return None
    return 100 * df.groupby("fold")[COLUNA_METRICA].mean()


def gerar_tabela_f1_dataset(pastas_datasets: list, df_split_f1: pd.DataFrame = None,
                            caminho_tex: str = TAB_F1_SAIDA_TEX,
                            caminho_csv: str = TAB_F1_SAIDA_CSV) -> None:
    por_nome = {Path(p).name: p for p in pastas_datasets}
    linhas = []
    for pasta_nome, rotulo, _ in FIG_GANHOS_DATASETS:
        pasta = por_nome.get(pasta_nome)
        if pasta is None:
            warnings.warn(f"[Tabela {TAB_F1_NUMERO}] dataset não encontrado: {pasta_nome}")
            continue
        r = {"dataset_pasta": pasta_nome, "Dataset": rotulo}
        folds_c0 = None
        for chave, cen in [("C0", "baseline"), *TAB_F1_CENARIOS_C2RAW, ("C2", "catch24")]:
            v = _f1_por_fold(pasta, cen, "(none)")
            if v is None:
                if not chave.startswith("C2raw"):
                    warnings.warn(f"[Tabela {TAB_F1_NUMERO}] {pasta_nome}: sem {cen}")
                r[f"{chave}_media"] = r[f"{chave}_dp"] = np.nan
                continue
            if chave == "C0":
                folds_c0 = set(v.index)
            elif folds_c0 is not None and set(v.index) != folds_c0:
                warnings.warn(f"[Tabela {TAB_F1_NUMERO}] {pasta_nome}: folds de {cen} diferem dos de C0")
            r[f"{chave}_media"], r[f"{chave}_dp"] = v.mean(), v.std(ddof=1)
        for chave, cen in [("C1", "drift"), ("C3", "catch24_drift")]:
            medias, dps = {}, {}
            for det in DETECTORES:
                v = _f1_por_fold(pasta, cen, det)
                if v is not None:
                    medias[det], dps[det] = v.mean(), v.std(ddof=1)
            if len(medias) < len(DETECTORES):
                warnings.warn(f"[Tabela {TAB_F1_NUMERO}] {pasta_nome}: {cen} com "
                              f"{len(medias)} de {len(DETECTORES)} detectores")
            m = pd.Series(medias, dtype=float)
            r[f"{chave}_mediana"] = m.median() if len(m) else np.nan
            r[f"{chave}_pior"] = m.min() if len(m) else np.nan
            r[f"{chave}_melhor"] = m.max() if len(m) else np.nan
            r[f"{chave}_n_detectores"] = len(m)
            r[f"{chave}_det_melhor"] = m.idxmax() if len(m) else None
        linhas.append(r)
    tab = pd.DataFrame(linhas)

    # conferência: as médias devem ser as mesmas usadas nos testes e na fig:ganhos
    if df_split_f1 is not None and not tab.empty:
        ref = df_split_f1.assign(A=100 * df_split_f1["val_A"], B=100 * df_split_f1["val_B"])
        checks = [
            ("C0_media", ref[ref["comparacao"] == "C0_vs_C2"].groupby("dataset_pasta")["A"].mean()),
            ("C2_media", ref[ref["comparacao"] == "C0_vs_C2"].groupby("dataset_pasta")["B"].mean()),
            ("C1_mediana", ref[ref["comparacao"] == "C0_vs_C1"].groupby("dataset_pasta")["B"].median()),
            ("C3_mediana", ref[ref["comparacao"] == "C2_vs_C3"].groupby("dataset_pasta")["B"].median()),
        ]
        t = tab.set_index("dataset_pasta")
        for col, serie in checks:
            dif = (t[col] - serie.reindex(t.index)).abs()
            ruins = dif[dif > 1e-6]
            if len(ruins):
                warnings.warn(f"[Tabela {TAB_F1_NUMERO}] {col} difere do teste estatístico em: "
                              f"{ {k: round(v, 4) for k, v in ruins.items()} }")
        print(f"[OK] Tabela {TAB_F1_NUMERO}: médias conferidas com as do teste estatístico")

    tab.to_csv(caminho_csv, index=False)

    def num(v):
        return "--" if pd.isna(v) else f"{v:.2f}"

    def valor(m, negrito=False):
        if m is None or pd.isna(m):
            return "--"
        return f"\\textbf{{{m:.2f}}}" if negrito else f"{m:.2f}"

    def com_faixa(med, lo, hi, negrito=False):
        if pd.isna(med):
            return "--"
        txt = f"\\textbf{{{med:.2f}}}" if negrito else f"{med:.2f}"
        return txt + f"~{{\\scriptsize[{lo:.2f}, {hi:.2f}]}}"

    corpo = []
    for _, r in tab.iterrows():
        # negrito: maior valor exibido entre C0, as duas variantes de C2.raw, C2 e as medianas de C1 e C3
        cols_negrito = ["C0_media", "C2rawC3_media", "C2rawC2_media", "C2_media", "C1_mediana", "C3_mediana"]
        cand = {k: round(r[k], 2) for k in cols_negrito if not pd.isna(r[k])}
        topo = max(cand.values()) if cand else None
        b = {k: (cand.get(k) == topo) for k in cols_negrito}
        def nome_det(d):
            return NOME_PARA_COL.get(d, d) if isinstance(d, str) else "--"
        corpo.append(" & ".join([
            r["Dataset"],
            valor(r["C0_media"], b["C0_media"]),
            valor(r["C2rawC3_media"], b["C2rawC3_media"]),
            valor(r["C2rawC2_media"], b["C2rawC2_media"]),
            valor(r["C2_media"], b["C2_media"]),
            com_faixa(r["C1_mediana"], r["C1_pior"], r["C1_melhor"], b["C1_mediana"]),
            com_faixa(r["C3_mediana"], r["C3_pior"], r["C3_melhor"], b["C3_mediana"]),
            nome_det(r.get("C1_det_melhor")),
            nome_det(r.get("C3_det_melhor")),
        ]) + " \\\\")
        if r["dataset_pasta"] == FIG_GANHOS_CONTRAEXEMPLO:
            corpo.insert(len(corpo) - 1, "\\hline")

    legenda = ("Macro F1-score (\\%) of RF per real-world dataset, averaged over the test folds. "
               "C1 and C3: median over the nine detectors, with the lowest and the highest value across detectors in brackets. "
               "C2.raw: catch22 and the original variables, evaluated with the configuration selected for "
               "the best C3 detector (C3 conf.) or for C2 (C2 conf.). \\textbf{Bold}: largest value among "
               "C0, C2.raw, C2 and the medians of C1 and C3. $^{\\dagger}$Detector with which the RF "
               "reached the largest macro F1-score on each dataset (best value in brackets), identified "
               "\\emph{a posteriori}.")
    tex = "\n".join([
        "% Gerada por teste_reais_agr_datasets.py -- nao editar a mao",
        "\\begin{minipage}{\\linewidth}",
        f"\\noindent\\sffamily\\small\\textbf{{\\color{{scolor}}Table {TAB_F1_NUMERO}.}}\\\\ " + legenda,
        "\\label{tab:f1_dataset_supp}",
        "\\par\\normalfont\\normalsize",
        "\\begin{center}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{C0} & \\multicolumn{2}{c}{\\textbf{C2.raw}} & \\textbf{C2} & "
        "\\textbf{C1} & \\textbf{C3} & \\multicolumn{2}{c}{\\textbf{Best detector}$^{\\dagger}$} \\\\",
        " & & {\\scriptsize C3 conf.} & {\\scriptsize C2 conf.} & & {\\scriptsize median [worst, best]} & "
        "{\\scriptsize median [worst, best]} & {\\scriptsize C1} & {\\scriptsize C3} \\\\",
        "\\hline",
        *corpo,
        "\\hline",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{minipage}",
    ])
    with open(caminho_tex, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    for chave, cen in TAB_F1_CENARIOS_C2RAW:
        faltam = tab.loc[tab[f"{chave}_media"].isna(), "Dataset"].tolist()
        if faltam:
            print(f"[AVISO] Tabela {TAB_F1_NUMERO}: sem {cen} em {faltam} ('--' na tabela)")
    print(f"[OK] Tabela {TAB_F1_NUMERO} salva em: {caminho_tex}")
    print(f"[OK] Valores da Tabela {TAB_F1_NUMERO} salvos em: {caminho_csv}")


# ============================================================
# CONTROLES EWMA (material suplementar)
#   EWMA_ALARM: media movel exponencial dos alarmes do melhor detector no lugar
#               do Algorithm 1 (caso particular com phi e beta constantes)
#   EWMA_VAR:   variancia exponencial do sinal no lugar dos alarmes
#   Cada controle e avaliado uma vez por dataset e cenario (controle_ewma.py),
#   com a configuracao do melhor detector (RF, normalizacoes, janela do catch22)
#   e lambda pelo horizonte de memoria do phi_b dele, sem busca propria. A
#   comparacao justa e com o Algorithm 1 desse mesmo detector, com a mesma
#   configuracao e o mesmo horizonte: so o acumulador (EWMA_ALARM) ou so o sinal
#   acumulado (EWMA_VAR) muda.
#   Estrutura em cada pasta de dataset (exportar_controles_ewma.sh):
#     <dataset>/controle_ewma/<C1|C3>_<EWMA_VAR|EWMA_ALARM>/
#         df_f1_all.csv, df_acc_all.csv, resultado_controle.json
#   Pareamento por fold, agregacao fold -> split -> fonte e Wilcoxon iguais aos
#   da tab:delta_real; BH entre os dois controles em cada cenario e metrica.
#   A = controle, B = Algorithm 1: delta positivo favorece o Algorithm 1.
# ============================================================
CTRL_PASTA = "controle_ewma"
CTRL_CENARIOS = {"C1": "drift", "C3": "catch24_drift"}
CTRL_NOMES = (("EWMA_ALARM", "EWMAalarm"), ("EWMA_VAR", "EWMAvar"))      # ordem das linhas
CTRL_ROTULOS = {"EWMAalarm": "EWMA$_{alarm}$ vs.\\ Algorithm~1",
                "EWMAvar": "EWMA$_{var}$ vs.\\ Algorithm~1"}
CTRL_TOLERANCIA_PP = 0.05     # aviso se o Algorithm 1 local diferir do usado no controle
CTRL_TABELA_NUMERO = "S-x"
CTRL_TABELA_TEX = os.path.join(PASTA_RAIZ, "tabela_controles_ewma.tex")
CTRL_TABELA_CSV = os.path.join(PASTA_RAIZ, "tabela_controles_ewma.csv")
CTRL_DATASET_TEX = os.path.join(PASTA_RAIZ, "tabela_controles_ewma_por_dataset.tex")
CTRL_DATASET_CSV = os.path.join(PASTA_RAIZ, "tabela_controles_ewma_por_dataset.csv")


def _ler_controle(pasta: str, nome_pasta: str, arquivo: str):
    """(df_metrica, json) do controle; (None, None) se ausente."""
    import json
    base = os.path.join(pasta, CTRL_PASTA, nome_pasta)
    arq, js = os.path.join(base, arquivo), os.path.join(base, "resultado_controle.json")
    if not (os.path.exists(arq) and os.path.exists(js)):
        return None, None
    df = pd.read_csv(arq)
    if not {"fold", COLUNA_METRICA}.issubset(df.columns):
        warnings.warn(f"Arquivo sem colunas esperadas: {arq}")
        return None, None
    with open(js) as fh:
        info = json.load(fh)
    return df[["fold", COLUNA_METRICA]].copy(), info


def coletar_pares_controles(pastas_datasets: list, arquivo: str, checar: bool = False) -> pd.DataFrame:
    """Pares (controle, Algorithm 1 do melhor detector do mesmo dataset e cenario) por fold."""
    rows, avisos = [], []
    for pasta in pastas_datasets:
        ds = Path(pasta).name
        for rotulo, cen_alg in CTRL_CENARIOS.items():
            for ctl, nome_ctl in CTRL_NOMES:
                df_a, info = _ler_controle(pasta, f"{rotulo}_{ctl}", arquivo)
                if df_a is None:
                    continue
                ref = info["detector_referencia"]
                df_b = load_csv_flexivel(pasta, cen_alg, ref, arquivo)
                if df_b.empty:
                    warnings.warn(f"Sem Algorithm 1: {ds} {cen_alg}/{ref}")
                    continue
                if checar:
                    # o detector de referencia do controle deve ser o melhor detector local
                    medias = {d: load_csv_flexivel(pasta, cen_alg, d, arquivo)[COLUNA_METRICA].mean()
                              for d in DETECTORES if not load_csv_flexivel(pasta, cen_alg, d, arquivo).empty}
                    melhor = max(medias, key=medias.get) if medias else ref
                    if melhor != ref and 100 * (medias[melhor] - medias.get(ref, np.nan)) > CTRL_TOLERANCIA_PP:
                        avisos.append(f"{ds:18s} {rotulo} controle com {ref} ({100 * medias.get(ref, np.nan):.2f}), "
                                      f"mas o melhor local e {melhor} ({100 * medias[melhor]:.2f})")
                a = df_a.rename(columns={COLUNA_METRICA: "val_A"})
                b = df_b.rename(columns={COLUNA_METRICA: "val_B"})[["fold", "val_B"]]
                m = pd.merge(a, b, on="fold", how="inner")
                if len(m) != len(a):
                    warnings.warn(f"Folds diferentes entre controle e Algorithm 1: {ds} {rotulo} {ctl}")
                m["dataset_pasta"], m["comparacao"], m["detector"] = ds, f"{nome_ctl}_vs_{rotulo}", "BEST"
                m["pasta"], m["run_A"], m["run_B"] = pasta, ctl, ref
                rows.append(m[["dataset_pasta", "pasta", "comparacao", "detector",
                               "fold", "run_A", "run_B", "val_A", "val_B"]])
    if avisos:
        print("\n[AVISO] O detector usado para construir o controle nao e o melhor detector local "
              "(o melhor foi escolhido pelos JSONs do cluster). Confira antes de usar a tabela:")
        for av in sorted(set(avisos)):
            print("   " + av)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def gerar_tabela_controles(res: dict, caminho_tex: str = CTRL_TABELA_TEX,
                           caminho_csv: str = CTRL_TABELA_CSV) -> None:
    """Uma tabela: linhas = EWMA_ALARM e EWMA_VAR contra o Algorithm 1 do melhor
    detector; colunas = C1 e C3 (F1, acuracia). BH entre os dois controles em
    cada cenario e metrica."""
    vals = {}
    for cen in ("C1", "C3"):
        for suf in ("f1", "acc"):
            linhas = [(nome, res.get(f"{nome}_vs_{cen}", {}).get(suf)) for _, nome in CTRL_NOMES]
            linhas = [(n, df.iloc[0]) for n, df in linhas if df is not None and not df.empty]
            if not linhas:
                continue
            p_bh = _bh_ajustado(np.array([r["p_value"] for _, r in linhas], dtype=float))
            for (n, r), pa in zip(linhas, p_bh):
                vals[(n, cen, suf)] = (r, pa)
    if not vals:
        print("[AVISO] Tabela dos controles nao gerada: sem resultados")
        return
    corpo, csv, ns = [], [], set()
    for _, nome in CTRL_NOMES:
        cels = []
        for cen in ("C1", "C3"):
            for suf in ("f1", "acc"):
                if (nome, cen, suf) not in vals:
                    cels.append("--")
                    continue
                r, pa = vals[(nome, cen, suf)]
                ns.add(int(r["n_unidades"]))
                delta = _fmt_num(r["cliff_delta"])
                if pa <= ALPHA_FDR:
                    delta = f"\\textbf{{{delta}}}"
                cels.append(f"{delta}~{{\\scriptsize({pa:.3f})}}~{{\\scriptsize[{int(r['n_pos'])}/{int(r['n_neg'])}]}}")
                csv.append({"controle": nome, "cenario": cen, "metrica": suf,
                            "n_fontes": int(r["n_unidades"]), "n_pos": int(r["n_pos"]),
                            "n_neg": int(r["n_neg"]), "delta": r["cliff_delta"],
                            "p_bruto": r["p_value"], "p_bh": pa, "significativo_bh": bool(pa <= ALPHA_FDR)})
        corpo.append(f"{CTRL_ROTULOS[nome]} & " + " & ".join(cels) + " \\\\")
    pd.DataFrame(csv).to_csv(caminho_csv, index=False)
    n_txt = "/".join(map(str, sorted(ns)))
    legenda = ("Paired dominance statistic $\\delta = (n_{+} - n_{-})/N$ between each control and Algorithm~1 "
               f"with the best detector of each dataset ($N={n_txt}$ independent real-world sources), with "
               "Benjamini--Hochberg-adjusted $p$-values (over the two controls) in parentheses and "
               "$[n_{+}/n_{-}]$ in brackets. \\textbf{Bold}: $p_{\\mathrm{adj}} \\le 0.05$. Positive "
               "$\\delta$ favours Algorithm~1. Both controls use the configuration selected for that detector "
               "(classifier hyperparameters, normalizations, and \\textit{catch22} window) and the memory "
               "horizon of its $\\phi_b$, without a search of their own. EWMA$_{alarm}$ replaces Algorithm~1 by "
               "an exponential moving average of the same alarms, its particular case with constant forgetting "
               "and weight; EWMA$_{var}$ replaces the alarms by the exponentially weighted variance of the signal.")
    tex = "\n".join([
        "% Gerada por teste_reais_agr_datasets.py -- nao editar a mao",
        "\\begin{minipage}{\\linewidth}",
        f"\\noindent\\sffamily\\small\\textbf{{\\color{{scolor}}Table {CTRL_TABELA_NUMERO}.}}\\\\ " + legenda,
        "\\label{tab:controles_ewma_supp}",
        "\\par\\normalfont\\normalsize",
        "\\begin{center}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        " & \\multicolumn{2}{c}{\\textbf{C1}} & \\multicolumn{2}{c}{\\textbf{C3}} \\\\",
        "\\cline{2-3}\\cline{4-5}",
        "\\textbf{Comparison} & \\multicolumn{1}{c}{\\textbf{F1}} & \\multicolumn{1}{c}{\\textbf{Acc}} & "
        "\\multicolumn{1}{c}{\\textbf{F1}} & \\multicolumn{1}{c}{\\textbf{Acc}} \\\\",
        "\\hline",
        *corpo,
        "\\hline",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{minipage}",
    ])
    with open(caminho_tex, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"[OK] Tabela {CTRL_TABELA_NUMERO} (controles EWMA) salva em: {caminho_tex}")
    print(f"[OK] Valores salvos em: {caminho_csv}")


def gerar_tabela_controles_por_dataset(pastas_datasets: list, caminho_tex: str = CTRL_DATASET_TEX,
                                       caminho_csv: str = CTRL_DATASET_CSV) -> None:
    """Macro F1 (%) por dataset: Algorithm 1 com o melhor detector, EWMA_ALARM e
    EWMA_VAR, em C1 e C3, com o detector de referencia (h fica so no CSV)."""
    rotulos = {p: r for p, r, _ in FIG_GANHOS_DATASETS}
    linhas = []
    for pasta in pastas_datasets:
        ds = Path(pasta).name
        for rotulo, cen_alg in CTRL_CENARIOS.items():
            df_al, info_al = _ler_controle(pasta, f"{rotulo}_EWMA_ALARM", "df_f1_all.csv")
            df_v, info_v = _ler_controle(pasta, f"{rotulo}_EWMA_VAR", "df_f1_all.csv")
            info = info_al or info_v
            if info is None:
                continue
            ref = info["detector_referencia"]
            df_b = load_csv_flexivel(pasta, cen_alg, ref, "df_f1_all.csv")
            linhas.append({
                "dataset_pasta": ds, "dataset": rotulos.get(ds, ds), "cenario": rotulo,
                "detector": NOME_PARA_COL.get(ref, ref), "h": info.get("horizonte_h"),
                "alg1": 100 * df_b[COLUNA_METRICA].mean() if not df_b.empty else np.nan,
                "ewma_alarm": 100 * df_al[COLUNA_METRICA].mean() if df_al is not None else np.nan,
                "ewma_var": 100 * df_v[COLUNA_METRICA].mean() if df_v is not None else np.nan,
            })
    t = pd.DataFrame(linhas)
    if t.empty:
        print("[AVISO] Tabela dos controles por dataset nao gerada: sem resultados")
        return
    t.to_csv(caminho_csv, index=False)
    ordem = [p for p, _, _ in FIG_GANHOS_DATASETS]
    corpo = []
    for p in ordem + sorted(set(t.dataset_pasta) - set(ordem)):
        g = t[t.dataset_pasta == p].set_index("cenario")
        if g.empty:
            continue
        cels = [g["dataset"].iloc[0]]
        for cen in ("C1", "C3"):
            if cen in g.index:
                r = g.loc[cen]
                vals = {c: round(float(r[c]), 2) for c in ("alg1", "ewma_alarm", "ewma_var") if pd.notna(r[c])}
                maior = max(vals.values()) if vals else None
                cels += [r["detector"],
                         *(("--" if c not in vals else
                            (f"\\textbf{{{vals[c]:.2f}}}" if vals[c] == maior else f"{vals[c]:.2f}"))
                           for c in ("alg1", "ewma_alarm", "ewma_var"))]
            else:
                cels += ["--"] * 4
        corpo.append(" & ".join(cels) + " \\\\")
    legenda = ("Macro F1-score (\\%) on the real datasets, averaged over the folds, of Algorithm~1 with the "
               "best detector of each scenario and of the two controls built with its configuration. "
               "\\textbf{Bold}: highest value in each scenario.")
    tex = "\n".join([
        "% Gerada por teste_reais_agr_datasets.py -- nao editar a mao",
        "\\begin{minipage}{\\linewidth}",
        "\\noindent\\sffamily\\small\\textbf{\\color{scolor}Table S-y.}\\\\ " + legenda,
        "\\label{tab:controles_ewma_dataset_supp}",
        "\\par\\normalfont\\normalsize",
        "\\begin{center}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "lrrr" * 2 + "}",
        "\\hline",
        " & \\multicolumn{4}{c}{\\textbf{C1}} & \\multicolumn{4}{c}{\\textbf{C3}} \\\\",
        "\\cline{2-5}\\cline{6-9}",
        "\\textbf{Dataset} & Det. & Alg.~1 & EWMA$_{alarm}$ & EWMA$_{var}$ & "
        "Det. & Alg.~1 & EWMA$_{alarm}$ & EWMA$_{var}$ \\\\",
        "\\hline",
        *corpo,
        "\\hline",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{minipage}",
    ])
    with open(caminho_tex, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"[OK] Tabela S-y (controles por dataset) salva em: {caminho_tex}")


def analisar_controles_ewma(pastas_datasets: list) -> None:
    res = {}
    for arquivo, rotulo_metrica, suf in METRICAS:
        df_pares = coletar_pares_controles(pastas_datasets, arquivo, checar=(suf == "f1"))
        if df_pares.empty:
            print(f"\n[AVISO] Controles EWMA: nenhum resultado em <dataset>/{CTRL_PASTA}/ ({arquivo})")
            return
        df_fonte = agregar_por_fonte(agregar_por_split(df_pares), meio=MEIO_AGREGACAO)
        df_res = analisar_fonte(df_fonte)
        df_res.to_csv(os.path.join(PASTA_RAIZ, f"analise_controles_ewma_fonte_{suf}.csv"), index=False)
        df_fonte.assign(dif=df_fonte["val_B"] - df_fonte["val_A"]).to_csv(
            os.path.join(PASTA_RAIZ, f"valores_controles_ewma_fonte_{suf}.csv"), index=False)
        print(f"\n=== Controles EWMA ({rotulo_metrica}); A = controle, B = Algorithm 1 do melhor detector ===")
        with pd.option_context("display.width", 200):
            print(df_res[["comparacao", "n_unidades", "n_pos", "n_neg", "cliff_delta", "p_value"]]
                  .to_string(index=False))
        for comp, grupo in df_res.groupby("comparacao"):
            res.setdefault(comp, {})[suf] = grupo.reset_index(drop=True)
    gerar_tabela_controles(res)
    gerar_tabela_controles_por_dataset(pastas_datasets)


# ============================================================
# F1 POR CLASSE de um dataset (ex.: MH, para ver se o ganho se concentra
# na classe 0/Idle). Fonte: df_f1_all_classe.csv (colunas "RF__class_<rótulo>",
# média entre os folds do melhor trial de cada run). Para C1 e C3, usa a pasta
# original, a mesma de load_csv_flexivel (Tabelas S-11 e S-15 e testes); a
# pasta de rerun só entra se o arquivo não existir na original. Reporta o
# melhor detector (maior macro F1 em df_f1_all.csv) e a mediana dos nove.
# Uso isolado:  python teste_reais_agr_datasets.py --f1-classe mhealth
# ============================================================
RERUN_DIR = "resultados_catch24_drift_rerun"


def _pasta_cenario(pasta: str, cenario: str, detector: str = None) -> str:
    # mesma ordem de load_csv_flexivel: <cenario>/<detector>, depois <cenario>
    cands = []
    if detector:
        cands.append(os.path.join(pasta, cenario, detector))
    cands.append(os.path.join(pasta, cenario))
    for c in cands:
        if os.path.exists(os.path.join(c, "df_f1_all.csv")):
            return c
    if detector:
        rr = os.path.join(pasta, RERUN_DIR, cenario, detector)
        if os.path.exists(os.path.join(rr, "df_f1_all.csv")):
            warnings.warn(f"{pasta}/{cenario}/{detector}: usando o rerun "
                          f"(df_f1_all.csv ausente na pasta original)")
            return rr
    return None


def _f1_classe(pasta_cen: str) -> pd.Series:
    arq = os.path.join(pasta_cen, "df_f1_all_classe.csv")
    if not os.path.exists(arq):
        return None
    df = pd.read_csv(arq)
    cols = [c for c in df.columns if c.startswith(f"{COLUNA_METRICA}__class_")]
    s = 100 * df[cols].mean()
    s.index = [c.split("__class_")[1] for c in cols]
    return s


def _f1_macro(pasta_cen: str) -> float:
    df = pd.read_csv(os.path.join(pasta_cen, "df_f1_all.csv"))
    return float(100 * df.groupby("fold")[COLUNA_METRICA].mean().mean())


def gerar_tabela_f1_por_classe(pastas_datasets: list, dataset_pasta: str = "mhealth") -> pd.DataFrame:
    por_nome = {Path(p).name: p for p in pastas_datasets}
    pasta = por_nome.get(dataset_pasta)
    if pasta is None:
        warnings.warn(f"[F1 por classe] dataset não encontrado: {dataset_pasta}")
        return pd.DataFrame()
    cols, info = {}, {}
    for chave, cen in [("C0", "baseline"), ("C2", "catch24")]:
        pc = _pasta_cenario(pasta, cen, "(none)")
        if pc is not None:
            cols[chave] = _f1_classe(pc)
            info[chave] = pc
    for chave, cen in [("C1", "drift"), ("C3", "catch24_drift")]:
        por_det, macro = {}, {}
        for det in DETECTORES:
            pc = _pasta_cenario(pasta, cen, det)
            if pc is None:
                continue
            f = _f1_classe(pc)
            if f is None:
                warnings.warn(f"[F1 por classe] {pc}: sem df_f1_all_classe.csv, "
                              f"{det} fica fora da tabela")
                continue
            por_det[det], macro[det] = f, _f1_macro(pc)
            # confere com a Tabela S-15: a média das classes deve ser o macro F1
            if abs(f.mean() - macro[det]) > 0.05:
                warnings.warn(f"[F1 por classe] {chave} {det}: média das classes "
                              f"{f.mean():.2f} difere do macro F1 {macro[det]:.2f}")
        if not por_det:
            continue
        melhor = max(macro, key=macro.get)
        cols[f"{chave}_melhor"] = por_det[melhor]
        cols[f"{chave}_mediana"] = pd.DataFrame(por_det).median(axis=1)
        info[chave] = f"melhor = {melhor} ({macro[melhor]:.2f}), {len(por_det)} detectores, pasta {_pasta_cenario(pasta, cen, melhor)}"
    tab = pd.DataFrame(cols)
    if {"C0", "C1_melhor"}.issubset(tab.columns):
        tab["C1_melhor-C0"] = tab["C1_melhor"] - tab["C0"]
        tab["C1_mediana-C0"] = tab["C1_mediana"] - tab["C0"]
    if {"C2", "C3_melhor"}.issubset(tab.columns):
        tab["C3_melhor-C2"] = tab["C3_melhor"] - tab["C2"]
        tab["C3_mediana-C2"] = tab["C3_mediana"] - tab["C2"]
    tab.index.name = "classe"
    tab = tab.sort_index(key=lambda i: i.astype(float))
    print(f"\n{'=' * 70}\nF1 POR CLASSE (%) - {dataset_pasta}\n{'=' * 70}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(tab.round(2).to_string())
    linha_media = tab.mean()
    print("\n  média das classes (= macro F1):")
    print("  " + ", ".join(f"{c}={v:.2f}" for c, v in linha_media.items()))
    if "0" in tab.index:
        sem0 = tab.drop(index="0").mean()
        print("  média sem a classe 0:")
        print("  " + ", ".join(f"{c}={v:.2f}" for c, v in sem0.items()))
    caminho = os.path.join(PASTA_RAIZ, f"tabela_f1_classe_{dataset_pasta}.csv")
    tab.to_csv(caminho)
    print(f"[OK] {caminho}")
    return tab


def main():
    pastas_datasets = descobrir_pastas_datasets(PASTA_RAIZ)
    imprimir_mapeamento_fonte(pastas_datasets)
    imprimir_mapeamento_minirocket()

    resultados_fonte = {comp: {} for comp in [c[2] for c in COMPARACOES]}

    for arquivo, rotulo_metrica, sufixo in METRICAS:

        print(f"\n{'#'*70}")
        print(f"  MÉTRICA: {rotulo_metrica}  ({arquivo})")
        print(f"{'#'*70}")

        print(f"\nColetando pares — {rotulo_metrica}...", flush=True)
        df_pares = coletar_pares(pastas_datasets, DETECTORES, COMPARACOES, arquivo)
        for _, _, rot, _ in COMPARACOES:
            n_ds = df_pares.loc[df_pares["comparacao"] == rot, "dataset_pasta"].nunique()
            if n_ds < len(pastas_datasets):
                print(f"  [AVISO] {rot}: pares em {n_ds} de {len(pastas_datasets)} datasets")
        print(f"  {len(df_pares)} pares fold-level | "
              f"{df_pares['dataset_pasta'].nunique()} splits de sensor")

        df_split = agregar_por_split(df_pares)
        if sufixo == "f1":
            df_split_f1 = df_split.copy()
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

        # Ranking de priorizacao + Friedman/Nemenyi para os cenarios
        # que variam por detector (ver CENARIOS_RANKING). Uso
        # exploratorio/descritivo, complementar a familia oficial
        # Wilcoxon+BH acima.
        for rotulo_curto, cenario_bruto, familia_delta in CENARIOS_RANKING:
            df_ranking = montar_ranking_priorizacao(
                df_fonte, df_res_fonte, sufixo, cenario_bruto, familia_delta,
            )
            imprimir_ranking_priorizacao(df_ranking, cenario_bruto, rotulo_metrica)
            caminho_ranking = os.path.join(
                PASTA_RAIZ, f"ranking_priorizacao_{rotulo_curto}_{sufixo}.csv"
            )
            df_ranking.to_csv(caminho_ranking, index=False)
            print(f"[OK] {caminho_ranking}")

            df_friedman, df_nemenyi = teste_friedman_nemenyi(
                df_fonte, sufixo, cenario_bruto,
            )
            imprimir_friedman_nemenyi(df_friedman, df_nemenyi, cenario_bruto, rotulo_metrica)
            caminho_friedman = os.path.join(
                PASTA_RAIZ, f"teste_friedman_{rotulo_curto}_{sufixo}.csv"
            )
            df_friedman.to_csv(caminho_friedman, index=False)
            print(f"[OK] {caminho_friedman}")
            caminho_nemenyi = os.path.join(
                PASTA_RAIZ, f"teste_nemenyi_{rotulo_curto}_{sufixo}.csv"
            )
            df_nemenyi.to_csv(caminho_nemenyi, index=False)
            print(f"[OK] {caminho_nemenyi}")

        # Transferencia de ranking entre cenarios, POR DATASET (uso
        # exploratorio -- ver docstrings de correlacao_rank_por_dataset
        # e correlacao_rank_por_dataset_minirocket). Pergunta comum as
        # tres comparacoes: "o detector bom no modelo barato (features
        # originais + drift, cenario 'drift') tambem e bom no modelo
        # caro (catch22+drift, ou MiniRocket early/late fusion) NO
        # MESMO DATASET?"
        #
        # As tres comparacoes (ver COMPARACOES_TRANSFERENCIA) sao
        # empilhadas na MESMA planilha, identificadas pela coluna
        # 'comparacao_transferencia'. Os cenarios do MiniRocket usam
        # dados de um pipeline separado (MINIROCKET_PASTA_RAIZ) ainda
        # nao finalizado -- datasets/detectores faltantes sao omitidos
        # automaticamente (ver warnings no console), NAO interrompem
        # a execucao das demais comparacoes.
        lista_corr, lista_resumo = [], []
        for rotulo_transf, origem, cenario_b in COMPARACOES_TRANSFERENCIA:
            if origem == "asoc":
                df_corr_i = correlacao_rank_por_dataset(df_split, "drift", cenario_b, sufixo)
            else:
                df_corr_i = correlacao_rank_por_dataset_minirocket(
                    df_split, "drift", cenario_b, arquivo, sufixo,
                )
            df_resumo_i = resumir_correlacao_datasets(df_corr_i)
            imprimir_correlacao_datasets(df_corr_i, df_resumo_i, "drift", rotulo_transf, rotulo_metrica)

            df_corr_i = df_corr_i.copy()
            df_corr_i.insert(0, "comparacao_transferencia", rotulo_transf)
            df_resumo_i = df_resumo_i.copy()
            df_resumo_i.insert(0, "comparacao_transferencia", rotulo_transf)
            lista_corr.append(df_corr_i)
            lista_resumo.append(df_resumo_i)

        df_corr_todas = pd.concat(lista_corr, ignore_index=True) if lista_corr else pd.DataFrame()
        df_resumo_todas = pd.concat(lista_resumo, ignore_index=True) if lista_resumo else pd.DataFrame()

        caminho_corr = os.path.join(
            PASTA_RAIZ, f"correlacao_rank_dataset_transferencia_{sufixo}.csv"
        )
        df_corr_todas.to_csv(caminho_corr, index=False)
        print(f"\n[OK] {caminho_corr}  (3 comparacoes empilhadas, ver coluna "
              f"'comparacao_transferencia')")
        caminho_resumo_corr = os.path.join(
            PASTA_RAIZ, f"correlacao_rank_dataset_resumo_{sufixo}.csv"
        )
        df_resumo_todas.to_csv(caminho_resumo_corr, index=False)
        print(f"[OK] {caminho_resumo_corr}")

        # Correlacao INTRA-FONTE: os splits da mesma fonte concordam
        # sobre qual detector e melhor? Teste direto da suposicao de
        # pseudorreplicacao (ver docstrings da secao acima). Calculado
        # para os cenarios com dados suficientes nas duas pontas
        # (ASOC 'drift'/'catch24_drift' sempre disponiveis; MiniRocket
        # 'c1'/'c2' apenas onde ja houver 2+ splits de uma mesma fonte
        # rodados -- experimento ainda incompleto).
        lista_pares_intra, lista_resumo_intra = [], []
        for cenario_bruto in ["drift", "catch24_drift"]:
            df_pares_i = correlacao_rank_intra_fonte(df_split, cenario_bruto, sufixo)
            lista_pares_intra.append(df_pares_i)
        for cenario_mr in ["c1", "c2"]:
            df_pares_i = correlacao_rank_intra_fonte_minirocket(cenario_mr, arquivo, sufixo)
            lista_pares_intra.append(df_pares_i)

        df_pares_intra_todas = pd.concat(
            [d for d in lista_pares_intra if not d.empty], ignore_index=True,
        ) if any(not d.empty for d in lista_pares_intra) else pd.DataFrame()
        df_resumo_intra = resumir_correlacao_intra_fonte(df_pares_intra_todas)
        imprimir_correlacao_intra_fonte(df_pares_intra_todas, df_resumo_intra, rotulo_metrica)

        caminho_pares_intra = os.path.join(
            PASTA_RAIZ, f"correlacao_rank_intra_fonte_{sufixo}.csv"
        )
        df_pares_intra_todas.to_csv(caminho_pares_intra, index=False)
        print(f"[OK] {caminho_pares_intra}")
        caminho_resumo_intra = os.path.join(
            PASTA_RAIZ, f"correlacao_rank_intra_fonte_resumo_{sufixo}.csv"
        )
        df_resumo_intra.to_csv(caminho_resumo_intra, index=False)
        print(f"[OK] {caminho_resumo_intra}")

        # Visao MISTA: W agregado (evidencia de concordancia real entre
        # seus 5 splits, ver correlacao intra-fonte acima), P2 e sp_sw
        # mantidos como splits individuais (evidencia de concordancia
        # fraca/negativa). Recalcula a transferencia de ranking nas
        # tres comparacoes usando essa agregacao diferenciada.
        df_split_misto = agregar_por_fonte_seletivo(df_split, GRUPOS_FONTE_MISTO)
        lista_corr_mista = []
        for rotulo_transf, origem, cenario_b in COMPARACOES_TRANSFERENCIA:
            if origem == "asoc":
                df_corr_m = correlacao_rank_por_dataset(df_split_misto, "drift", cenario_b, sufixo)
            else:
                val_a_m, _ = extrair_valores_cenario_split(df_split_misto, "drift")
                val_a_m = val_a_m[val_a_m["detector"].isin(DETECTORES)]
                val_b_m = agregar_valores_minirocket_seletivo(
                    extrair_valores_minirocket_split(cenario_b, arquivo), GRUPOS_FONTE_MISTO,
                )
                df_corr_m = _correlacionar_valores_por_dataset(
                    val_a_m, val_b_m, "drift", f"minirocket_{cenario_b}", sufixo,
                )
            df_resumo_m = resumir_correlacao_visao_mista(df_corr_m)
            imprimir_correlacao_visao_mista(df_corr_m, df_resumo_m, rotulo_transf, rotulo_metrica)
            df_corr_m = df_corr_m.copy()
            df_corr_m.insert(0, "comparacao_transferencia", rotulo_transf)
            lista_corr_mista.append(df_corr_m)

        df_corr_mista_todas = pd.concat(lista_corr_mista, ignore_index=True) if lista_corr_mista else pd.DataFrame()
        caminho_mista = os.path.join(
            PASTA_RAIZ, f"correlacao_rank_dataset_visao_mista_{sufixo}.csv"
        )
        df_corr_mista_todas.to_csv(caminho_mista, index=False)
        print(f"[OK] {caminho_mista}  (W agregado, P2/sp_sw individuais)")

    gerar_tabela_delta(resultados_fonte)
    gerar_tabela_c2raw_c2(resultados_fonte)
    gerar_figura_ganhos(df_split_f1)
    gerar_tabela_f1_dataset(pastas_datasets, df_split_f1)
    gerar_tabela_f1_por_classe(pastas_datasets, "mhealth")
    analisar_controles_ewma(pastas_datasets)

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--f1-classe":
        # só a tabela de F1 por classe, sem os testes estatísticos
        gerar_tabela_f1_por_classe(descobrir_pastas_datasets(PASTA_RAIZ),
                                   sys.argv[2] if len(sys.argv) > 2 else "mhealth")
    else:
        main()