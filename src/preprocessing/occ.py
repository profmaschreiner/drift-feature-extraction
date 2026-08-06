import os
import pandas as pd
import numpy as np

# ==============================
# CONFIGURAÇÕES
# ==============================

PASTA_ENTRADA = "datasetsNew/occ"
PASTA_SAIDA = "datasetsNew/occ_pre_processados"


#COL_CLASSE = "label"
COL_CLASSE = "Room_Occupancy_Count"
# NOVA OPÇÃO:
# se True, remove de todos os dataframes as classes que não aparecem em todos os arquivos
REMOVER_CLASSES_NAO_COMUNS = False

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ==============================
# FUNÇÃO: ANALISAR MISSING
# ==============================
def analisar_missing(df, nome_arquivo):
    total = len(df)
    missing_por_coluna = df.isna().sum()
    perc_missing = (missing_por_coluna / total) * 100 if total > 0 else 0

    resumo = pd.DataFrame({
        "coluna": df.columns,
        "missing_count": missing_por_coluna.values,
        "missing_percent": perc_missing.values if hasattr(perc_missing, "values") else [perc_missing] * len(df.columns)
    })

    resumo["arquivo"] = nome_arquivo

    return resumo

# ==============================
# FUNÇÃO: ANÁLISE DE DISTRIBUIÇÃO
# ==============================
def analise(dfs, titulo="Distribuição de classes"):
    nomes = [f"df_{i}" for i in range(len(dfs))]
    col_classe = COL_CLASSE

    classes_validas = []
    for df in dfs:
        if col_classe in df.columns and not df.empty:
            classes_validas.append(set(df[col_classe].unique()))

    if not classes_validas:
        print(f"\n{titulo}")
        print("Nenhum dataframe válido para análise.")
        return

    todas_classes = sorted(set().union(*classes_validas))

    resultado = []

    for nome, df in zip(nomes, dfs):
        if df.empty:
            freq = pd.Series(0, index=todas_classes, dtype=int)
        else:
            freq = df[col_classe].value_counts(normalize=False)
            freq = freq.reindex(todas_classes, fill_value=0)

        freq["dataframe"] = nome
        resultado.append(freq)

    tabela = pd.DataFrame(resultado).set_index("dataframe")
    tabela = tabela[todas_classes]

    print(f"\n{titulo}")
    print(tabela)

# ==============================
# FUNÇÃO: SUBAMOSTRAGEM PROPORCIONAL TEMPORAL
# ==============================
def subamostragem_proporcional_temporal(
    df,
    col_classe="label",
    frac_manter=0.30,
    classes_preservadas=None
):
    """
    Reduz o dataframe mantendo aproximadamente a proporção de classes
    dentro do próprio arquivo e preservando a ordem temporal final.

    Estratégia:
    - Para cada classe, mantém uma fração fixa das amostras.
    - Para classes preservadas, mantém 100%.
    - A seleção é distribuída ao longo do tempo usando índices
      igualmente espaçados, evitando amostragem aleatória pura.
    """
    if classes_preservadas is None:
        classes_preservadas = set()

    if not (0 < frac_manter <= 1):
        raise ValueError("frac_manter deve estar no intervalo (0, 1].")

    if df.empty:
        return df.copy()

    indices_selecionados = []

    for classe, grupo in df.groupby(col_classe, sort=False):
        idx = grupo.index.to_numpy()
        n = len(idx)

        if n == 0:
            continue

        if classe in classes_preservadas:
            keep_idx = idx
        else:
            n_keep = max(1, int(round(n * frac_manter)))

            # seleciona pontos distribuídos ao longo do tempo
            posicoes = np.linspace(0, n - 1, n_keep, dtype=int)
            keep_idx = idx[posicoes]

        indices_selecionados.extend(keep_idx.tolist())

    indices_selecionados = sorted(indices_selecionados)

    df_out = df.loc[indices_selecionados].reset_index(drop=True)
    return df_out

# ==============================
# FUNÇÃO: DESCOBRIR CLASSES COMUNS
# ==============================
def obter_classes_comuns(arquivos, pasta_entrada, col_classe):
    """
    Lê todos os arquivos, remove NA e calcula a interseção das classes
    presentes em todos os dataframes.
    """
    classes_por_arquivo = {}

    for arquivo in arquivos:
        caminho = os.path.join(pasta_entrada, arquivo)
        df = pd.read_csv(caminho)
        df = df.replace(['?', 'N/A', ''], np.nan)
        df = df.dropna().reset_index(drop=True)

        if col_classe not in df.columns:
            raise ValueError(f"A coluna de classe '{col_classe}' não existe em {arquivo}.")

        classes_por_arquivo[arquivo] = set(df[col_classe].unique())

    classes_comuns = set.intersection(*classes_por_arquivo.values()) if classes_por_arquivo else set()
    return classes_comuns, classes_por_arquivo

# ==============================
# PROCESSAMENTO PRINCIPAL
# ==============================
relatorio_missing_total = []

arquivos = sorted([f for f in os.listdir(PASTA_ENTRADA) if f.endswith(".csv")])

print(f"Total de indivíduos encontrados: {len(arquivos)}")

# ==============================
# ETAPA OPCIONAL: IDENTIFICAR CLASSES COMUNS
# ==============================
classes_comuns = None
classes_por_arquivo = None

if REMOVER_CLASSES_NAO_COMUNS:
    classes_comuns, classes_por_arquivo = obter_classes_comuns(
        arquivos,
        PASTA_ENTRADA,
        COL_CLASSE
    )

    print("\nClasses por arquivo:")
    for arq, classes in classes_por_arquivo.items():
        print(f"{arq}: {sorted(classes)}")

    print(f"\nClasses presentes em todos os arquivos: {sorted(classes_comuns)}")

dfs_originais = []
dfs_processados = []
i = 0

for idx, arquivo in enumerate(arquivos):
    caminho = os.path.join(PASTA_ENTRADA, arquivo)

    print(f"\nProcessando: {arquivo}")

    # ==========================
    # 1) LEITURA
    # ==========================
    df = pd.read_csv(caminho)
    print(f"Tamanho original: {df.shape}")

    # ==========================
    # 2) PADRONIZA VALORES AUSENTES
    # ==========================
    df = df.replace(['?', 'N/A', ''], np.nan)

    # ==========================
    # 3) ANÁLISE DE MISSING
    # ==========================
    relatorio = analisar_missing(df, arquivo)
    relatorio_missing_total.append(relatorio)

    # ==========================
    # 4) REMOÇÃO DE LINHAS COM NA
    # ==========================
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Removidas {before - after} linhas com NA")

    # ==========================
    # 5) REMOÇÃO OPCIONAL DE CLASSES NÃO COMUNS
    # ==========================
    if REMOVER_CLASSES_NAO_COMUNS:
        before_classes = len(df)
        df = df[df[COL_CLASSE].isin(classes_comuns)].reset_index(drop=True)
        after_classes = len(df)
        print(f"Removidas {before_classes - after_classes} linhas de classes não comuns")

    dfs_originais.append(df.copy())
    """
    # ==========================
    # 6) SUBAMOSTRAGEM PROPORCIONAL POR CLASSE
    # ==========================
    if df.empty:
        print("Dataframe ficou vazio após os filtros. Arquivo será ignorado.")
        continue

    dist_antes = df[COL_CLASSE].value_counts(normalize=True).sort_index() * 100
    
    # preserva apenas classes comuns também nas preservadas
    classes_preservadas_ativas = set(CLASSES_PRESERVADAS)
    if REMOVER_CLASSES_NAO_COMUNS:
        classes_preservadas_ativas = classes_preservadas_ativas.intersection(classes_comuns)

    df_reduzido = subamostragem_proporcional_temporal(
        df,
        col_classe=COL_CLASSE,
        frac_manter=FRAC_MANter,
        classes_preservadas=classes_preservadas_ativas
    )

    dist_depois = df_reduzido[COL_CLASSE].value_counts(normalize=True).sort_index() * 100

    print(f"Tamanho após subamostragem: {df_reduzido.shape}")
    print("\nPercentual por classe antes:")
    print(dist_antes.round(2))
    print("\nPercentual por classe depois:")
    print(dist_depois.round(2))

    dfs_processados.append(df_reduzido.copy())
    """
    df_reduzido = df
    # ==========================
    # 7) SALVAR ARQUIVO PROCESSADO
    # ==========================
    #if idx in [0, 3, 5, 6, 7, 8, 9, 10, 11]:
    caminho_saida = os.path.join(PASTA_SAIDA, "base" + str(i) + ".csv")
    df_reduzido.to_csv(caminho_saida, index=False)
    i += 1

# ==============================
# RELATÓRIOS FINAIS
# ==============================
analise(dfs_originais, titulo="Distribuição original de classes")
analise(dfs_processados, titulo="Distribuição após subamostragem")

df_missing_final = pd.concat(relatorio_missing_total, ignore_index=True)
df_missing_final.to_csv(
    os.path.join(PASTA_SAIDA, "relatorio_missing.csv"),
    index=False
)

print("\nPré-processamento concluído!")
