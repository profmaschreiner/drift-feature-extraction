"""
heatmap_default_otimizado_grupos.py
=====================================
Gera heatmaps comparando hiperparâmetros default vs. otimizados nos
cenários C1 (drift) e C3 (catch24_drift), separando os 20 datasets
sintéticos em dois grupos conforme o sufixo do nome da pasta:

    _D_  : feature afetada pelo drift é Diferente entre as classes
    _I_  : feature afetada pelo drift é Igual (a mesma) entre as classes

Para cada métrica (F1-score macro, Acurácia) é gerada uma figura com
dois heatmaps lado a lado (grupo _D_ | grupo _I_):

    Linhas   : 9 detectores de drift
    Colunas  : C1_default | C1_otimizado | C3_default | C3_otimizado
    Célula   : média por dataset (média por fold, depois média entre folds),
               seguida de média simples entre os datasets do grupo — MESMO
               pipeline de agregação usado nos scripts de teste estatístico
               (test_estat_ot_default_sintetico_agr.py e
               teste_estatistico_cenarios_sintetico_agr.py), garantindo que
               heatmap e teste estatístico sejam diretamente comparáveis.
               Cada dataset tem peso igual, independente do número de
               folds/runs que possui.

C0 (baseline, sem detector de drift e sem catch22/catch24) e C2
(catch22/catch24 puro, sem detector) não compõem a matriz principal,
pois não admitem detector de drift nem distinção default/otimizado.
Seus valores são exibidos como linhas extras mescladas no topo de cada
heatmap (C0 acima de C2), com a média geral do grupo (_D_ ou _I_).

Estrutura de pastas esperada (idêntica aos scripts de teste estatístico):
    PASTA_RAIZ/<dataset>/resultados_catch24_drift/<cenario>/<detector>/df_f1_all.csv
    cenários: drift, drift_ot, catch24_drift, catch24_drift_ot

Saídas
------
* PNG : heatmap_f1_default_otimizado_grupos.png
* PNG : heatmap_acc_default_otimizado_grupos.png
* CSV : heatmap_f1_default_otimizado_grupos_dados.csv  (matriz usada na figura)
* CSV : heatmap_acc_default_otimizado_grupos_dados.csv

Configuração: ajuste apenas PASTA_RAIZ.
"""

import os
import warnings
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÃO — ajuste apenas esta seção
# ============================================================

PASTA_RAIZ = "exp_otimizacao/result_sintetico_completo"

SUBPASTA_EXPERIMENTO = "resultados_catch24_drift"

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

NOME_PARA_LINHA = {
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

# (cenário_default, cenário_otimizado, rótulo_base)
PARES_CENARIO = [
    ("drift",         "drift_ot",         "C1"),
    ("catch24_drift", "catch24_drift_ot", "C3"),
]

COLUNAS_HEATMAP = ["C1_default", "C3_default"]

# Detectores para os quais a célula da matriz principal deve exibir,
# além do valor default, o valor com hiperparâmetros otimizados (formato
# "default → otimizado"). Atualmente aplicado a todos os detectores.
DETECTORES_COM_OTIMIZACAO_EXIBIDA = list(DETECTORES)

# Rotulo de exibicao das colunas (notacao C1.A / C3.A ja usada na Secao
# V-C do artigo para a configuracao default dos detectores).
ROTULO_COLUNA_HEATMAP = {
    "C1_default": "C1.A",
    "C3_default": "C3.A",
}

# Cenário C2: catch22/catch24 puro, sem detector de drift (pasta sem
# subpasta por detector — os CSVs ficam direto dentro dela).
CENARIO_C2 = "catch24"

# Cenário C0: baseline, sem detector de drift e sem catch22/catch24 (pasta
# sem subpasta por detector, mesma estrutura de CENARIO_C2).
CENARIO_C0 = "baseline"

GRUPOS = ["_D_", "_I_"]
GRUPO_ROTULO = {
    "_D_": "Feature diferente entre classes (_D_)",
    "_I_": "Feature igual entre classes (_I_)",
}

# Linha ORACLE: valores fixos fornecidos diretamente (cenario de deteccao
# perfeita dos drifts, nao calculado a partir dos CSVs de experimento).
# Mesma estrutura de colunas de COLUNAS_HEATMAP (C1.A, C3.A) por grupo.
VALORES_ORACLE = {
    "_D_": {"C1_default": 97.90, "C3_default": 98.01},
    "_I_": {"C1_default": 40.73, "C3_default": 42.15},
}

METRICAS = [
    ("df_f1_all.csv",  "F1-score macro", "f1",  "heatmap_f1_default_otimizado_grupos"),
    ("df_acc_all.csv", "Acurácia",       "acc", "heatmap_acc_default_otimizado_grupos"),
]

COLUNA_METRICA = "RF"

COR_BASE = "#355C7D"

# ============================================================
# DESCOBERTA E AGRUPAMENTO DE PASTAS
# ============================================================

def descobrir_pastas_datasets(pasta_raiz: str, subpasta_experimento: str = "") -> list:
    raiz = Path(pasta_raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Pasta raiz não encontrada: {pasta_raiz}\n"
            "Ajuste a variável PASTA_RAIZ no início do script."
        )
    candidatos = sorted([p for p in raiz.iterdir() if p.is_dir()])

    pastas = []
    for p in candidatos:
        base = (p / subpasta_experimento) if subpasta_experimento else p
        if base.is_dir() and (base / "drift").is_dir():
            pastas.append(str(base))

    if not pastas:
        raise RuntimeError(
            f"Nenhuma subpasta válida encontrada em '{pasta_raiz}'.\n"
            f"Verifique a subpasta '{subpasta_experimento}' e a presença de 'drift/'."
        )
    print(f"Datasets sintéticos encontrados ({len(pastas)}): "
          f"{[Path(p).parent.name for p in pastas]}")
    return pastas


def classificar_grupo(nome_dataset: str) -> str:
    """Retorna '_D_', '_I_' ou None se o dataset não pertencer a nenhum grupo."""
    for sufixo in GRUPOS:
        if nome_dataset.endswith(sufixo):
            return sufixo
    return None


def agrupar_pastas(pastas: list) -> dict:
    """Retorna {'_D_': [pastas...], '_I_': [pastas...]}, ignorando datasets
    cujo nome não termina em nenhum dos sufixos de GRUPOS."""
    grupos = {g: [] for g in GRUPOS}
    ignorados = []
    for pasta in pastas:
        nome_dataset = Path(pasta).parent.name
        grupo = classificar_grupo(nome_dataset)
        if grupo is None:
            ignorados.append(nome_dataset)
            continue
        grupos[grupo].append(pasta)

    for g in GRUPOS:
        print(f"  Grupo {g}: {len(grupos[g])} datasets -> "
              f"{[Path(p).parent.name for p in grupos[g]]}")
    if ignorados:
        warnings.warn(
            f"{len(ignorados)} dataset(s) não terminam em '_D_' nem '_I_' "
            f"e foram ignorados: {ignorados}"
        )
    return grupos


# ============================================================
# LEITURA DOS CSVS E CÁLCULO DA MÉDIA
# ============================================================

def load_media_por_fold(pasta_dataset: str, cenario: str, detector: str,
                         arquivo: str) -> float:
    """
    Lê o CSV de um dataset/cenário/detector e retorna a média do dataset,
    seguindo o MESMO pipeline de agregação dos scripts de teste estatístico:
      1. média de COLUNA_METRICA por fold (agrupando 'run' dentro de cada fold)
      2. média dessas médias entre os folds -> 1 valor representando o dataset

    Retorna np.nan se o arquivo não existir ou não tiver as colunas esperadas.
    """
    path = os.path.join(pasta_dataset, cenario, detector, arquivo)
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    if not {"fold", COLUNA_METRICA}.issubset(df.columns):
        warnings.warn(f"Colunas esperadas ausentes em: {path}")
        return np.nan
    media_por_fold = df.groupby("fold")[COLUNA_METRICA].mean()
    if media_por_fold.empty:
        return np.nan
    return float(media_por_fold.mean())


def calcular_matriz(grupos: dict, arquivo: str) -> dict:
    """
    Para cada grupo (_D_, _I_), calcula a matriz [detector x coluna_heatmap]
    com a MÉDIA POR DATASET (não por fold/linha bruta), seguindo o mesmo
    pipeline de agregação usado nos scripts de teste estatístico:
        por dataset: média por fold -> média entre folds
        por grupo  : média simples entre os datasets do grupo (peso igual
                     a cada dataset, independente do número de folds/runs)

    Para os detectores listados em DETECTORES_COM_OTIMIZACAO_EXIBIDA
    (atualmente todos), calcula também colunas auxiliares
    "{rotulo}_otimizado", usadas apenas para anotar a célula correspondente
    no desenho do heatmap (não fazem parte de COLUNAS_HEATMAP nem da matriz
    principal).

    Retorna: {'_D_': DataFrame, '_I_': DataFrame}
    """
    resultado = {}

    colunas_extra = [
        f"{rotulo}_otimizado" for _, _, rotulo in PARES_CENARIO
    ]

    for grupo, pastas_grupo in grupos.items():
        matriz = pd.DataFrame(
            index=DETECTORES, columns=COLUNAS_HEATMAP + colunas_extra, dtype=float
        )

        for detector in DETECTORES:
            for cen_def, cen_ot, rotulo in PARES_CENARIO:
                medias_def = []
                for pasta in pastas_grupo:
                    m_def = load_media_por_fold(pasta, cen_def, detector, arquivo)
                    if not np.isnan(m_def):
                        medias_def.append(m_def)

                col_def = f"{rotulo}_default"
                matriz.loc[detector, col_def] = (
                    float(np.mean(medias_def)) if medias_def else np.nan
                )

                if not medias_def:
                    warnings.warn(
                        f"[{grupo}] Sem dados para {detector} / {col_def}"
                    )

                if detector in DETECTORES_COM_OTIMIZACAO_EXIBIDA:
                    medias_ot = []
                    for pasta in pastas_grupo:
                        m_ot = load_media_por_fold(pasta, cen_ot, detector, arquivo)
                        if not np.isnan(m_ot):
                            medias_ot.append(m_ot)

                    col_ot = f"{rotulo}_otimizado"
                    matriz.loc[detector, col_ot] = (
                        float(np.mean(medias_ot)) if medias_ot else np.nan
                    )

                    if not medias_ot:
                        warnings.warn(
                            f"[{grupo}] Sem dados para {detector} / {col_ot}"
                        )

        resultado[grupo] = matriz

    return resultado


def load_media_por_fold_sem_detector(pasta_dataset: str, cenario: str,
                                      arquivo: str) -> float:
    """
    Mesmo pipeline de load_media_por_fold, mas para cenários sem subpasta
    de detector (ex.: C2/catch24, onde o CSV fica direto dentro do cenário).
    """
    path = os.path.join(pasta_dataset, cenario, arquivo)
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    if not {"fold", COLUNA_METRICA}.issubset(df.columns):
        warnings.warn(f"Colunas esperadas ausentes em: {path}")
        return np.nan
    media_por_fold = df.groupby("fold")[COLUNA_METRICA].mean()
    if media_por_fold.empty:
        return np.nan
    return float(media_por_fold.mean())


def calcular_c2_por_grupo(grupos: dict, arquivo: str) -> dict:
    """
    Calcula, para cada grupo (_D_, _I_), a média de C2 (catch22/catch24
    puro) usando o mesmo pipeline de agregação por dataset (média por fold
    -> média entre folds -> média simples entre datasets do grupo). Não há
    separação por detector nem por default/otimizado, pois C2 não usa
    detector de drift.

    Retorna: {'_D_': float, '_I_': float}
    """
    resultado = {}
    for grupo, pastas_grupo in grupos.items():
        medias = [
            load_media_por_fold_sem_detector(pasta, CENARIO_C2, arquivo)
            for pasta in pastas_grupo
        ]
        medias = [m for m in medias if not np.isnan(m)]
        if not medias:
            warnings.warn(f"[{grupo}] Sem dados para C2 ({CENARIO_C2}/{arquivo})")
            resultado[grupo] = np.nan
        else:
            resultado[grupo] = float(np.mean(medias))
    return resultado


def calcular_c0_por_grupo(grupos: dict, arquivo: str) -> dict:
    """
    Calcula, para cada grupo (_D_, _I_), a média de C0 (baseline, sem
    detector de drift e sem catch22/catch24) usando o mesmo pipeline de
    agregação por dataset (média por fold -> média entre folds -> média
    simples entre datasets do grupo). Não há separação por detector nem
    por default/otimizado, pois C0 não usa detector de drift.

    Retorna: {'_D_': float, '_I_': float}
    """
    resultado = {}
    for grupo, pastas_grupo in grupos.items():
        medias = [
            load_media_por_fold_sem_detector(pasta, CENARIO_C0, arquivo)
            for pasta in pastas_grupo
        ]
        medias = [m for m in medias if not np.isnan(m)]
        if not medias:
            warnings.warn(f"[{grupo}] Sem dados para C0 ({CENARIO_C0}/{arquivo})")
            resultado[grupo] = np.nan
        else:
            resultado[grupo] = float(np.mean(medias))
    return resultado


# ============================================================
# GERAÇÃO DO HEATMAP
# ============================================================

def _cor_interpolada(valor: float, vmin: float, vmax: float) -> tuple:
    """Interpola entre branco e COR_BASE conforme o valor normalizado."""
    import matplotlib.colors as mcolors
    if np.isnan(valor) or vmax == vmin:
        t = 0.0
    else:
        t = (valor - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)
    branco = np.array(mcolors.to_rgb("#FFFFFF"))
    base   = np.array(mcolors.to_rgb(COR_BASE))
    return tuple(branco + t * (base - branco))


def gerar_figura_heatmap(matrizes: dict, valores_c0: dict, valores_c2: dict,
                          rotulo_metrica: str, caminho_saida: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    # ------------------------------------------------------------------
    # Conversao para escala percentual (0-100): os CSVs de origem trazem
    # a metrica em fracao (0-1); a figura exibe em percentual, tanto nas
    # celulas quanto na colorbar.
    # ------------------------------------------------------------------
    matrizes = {g: matrizes[g] * 100.0 for g in GRUPOS}
    valores_c0 = {g: (v * 100.0 if not np.isnan(v) else v) for g, v in valores_c0.items()}
    valores_c2 = {g: (v * 100.0 if not np.isnan(v) else v) for g, v in valores_c2.items()}

    # ------------------------------------------------------------------
    # Escala de cor global (branco -> COR_BASE), comum aos dois grupos,
    # para que a tonalidade seja diretamente comparavel entre _D_ e _I_
    # (mesmo principio da heatmap dos dados reais: escala absoluta).
    #
    # Usa-se PERCENTIL (5-95) em vez de min/max bruto, exatamente como em
    # tabela_heatmap.py (heatmap dos dados reais): outliers pontuais
    # puxariam o extremo da escala e comprimiriam a variacao util contra
    # uma faixa de cor estreita, tornando a tabela quase monocromatica.
    # ------------------------------------------------------------------
    PERCENTIL_BAIXO, PERCENTIL_ALTO = 5, 95
    todos_valores = np.concatenate([
        matrizes[g][COLUNAS_HEATMAP].to_numpy(dtype=float).flatten() for g in GRUPOS
    ] + [
        np.array([v for v in valores_c0.values() if not np.isnan(v)])
    ] + [
        np.array([v for v in valores_c2.values() if not np.isnan(v)])
    ])
    todos_valores = todos_valores[~np.isnan(todos_valores)]
    if todos_valores.size:
        vmin = float(np.percentile(todos_valores, PERCENTIL_BAIXO))
        vmax = float(np.percentile(todos_valores, PERCENTIL_ALTO))
    else:
        vmin, vmax = 0.0, 1.0
    print(f"[INFO] Escala de cor (percentis {PERCENTIL_BAIXO}-{PERCENTIL_ALTO}): "
          f"valor em [{vmin:.2f}, {vmax:.2f}] "
          f"(min/max bruto: [{todos_valores.min():.2f}, {todos_valores.max():.2f}])")

    n_det    = len(DETECTORES)
    n_cols_grupo = len(COLUNAS_HEATMAP)
    n_grupos = len(GRUPOS)
    n_cols   = n_cols_grupo * n_grupos      # 8 colunas: 4 (D) + 4 (I)
    n_linhas = n_det + 3                    # +1 para C0, +1 para C2, +1 para ORACLE

    cell_w, cell_h = 2.0, 0.7
    row_label_w    = 1.6
    group_hdr_h    = 0.8   # linha do cabecalho agrupador "Feature (D)/(I)"
    col_hdr_h      = 1.1   # linha dos rotulos de coluna (C1.A, C3.A)
    # Altura reduzida para deixar a barra mais fina na vertical.
    cbar_h         = 0.55
    gap_table_cbar = 0.10  # espaço fixo e pequeno entre a tabela e a colorbar

    # Conteudo vertical do eixo de dados (sem a colorbar): cabecalho de
    # grupo + matriz (C2 + detectores) + rodape de colunas. Esta formula
    # deve ser EXATAMENTE igual ao span real (y_top - y_footer) calculado
    # mais abaixo, ou a altura fisica do eixo (em polegadas) nao
    # corresponde ao conteudo desenhado, criando um espaco vazio.
    conteudo_tabela_h = group_hdr_h + col_hdr_h + cell_h * n_linhas

    fig_w = row_label_w + cell_w * n_cols
    fig_h = conteudo_tabela_h + gap_table_cbar + cbar_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Eixos posicionados com coordenadas ABSOLUTAS (fracao da figura),
    # calculadas a partir das mesmas medidas em polegadas usadas no
    # desenho — elimina qualquer ambiguidade de margem automatica do
    # matplotlib (GridSpec/tight_layout/inset_axes) que causava o espaco
    # vazio entre a tabela e a colorbar.
    ax_bottom_frac = (cbar_h + gap_table_cbar) / fig_h
    ax_height_frac = conteudo_tabela_h / fig_h
    ax  = fig.add_axes([0.0, ax_bottom_frac, 1.0, ax_height_frac])

    # cbar_ax criado DIRETO com add_axes (sem cax intermediario / sem
    # inset_axes): a faixa inteira reservada na base da figura já tem
    # exatamente a altura medida para a colorbar completa, então o
    # eixo da colorbar ocupa essa faixa quase por completo, centrado
    # horizontalmente, sem adivinhar uma posição relativa.
    cbar_width_frac = 0.45
    cbar_left_frac  = (1.0 - cbar_width_frac) / 2.0
    cbar_bottom_frac = 0.0
    cbar_height_frac = cbar_h / fig_h
    cbar_ax = fig.add_axes([cbar_left_frac, cbar_bottom_frac,
                             cbar_width_frac, cbar_height_frac])

    ax.set_xlim(0, row_label_w + cell_w * n_cols)
    ax.axis("off")

    y_top = group_hdr_h + col_hdr_h + cell_h * n_linhas

    # --- Cabeçalho agrupador: "Feature (D)" / "Feature (E)" ---
    GROUP_TITLE = {"_D_": "Feature (D)", "_I_": "Feature (E)"}
    for g_idx, grupo in enumerate(GRUPOS):
        x0 = row_label_w + g_idx * n_cols_grupo * cell_w
        x1 = x0 + n_cols_grupo * cell_w
        ax.add_patch(plt.Rectangle(
            (x0, y_top - group_hdr_h), x1 - x0, group_hdr_h,
            facecolor="#EFEFEF", edgecolor="#CCCCCC", linewidth=1.5
        ))
        ax.text((x0 + x1) / 2, y_top - group_hdr_h / 2, GROUP_TITLE[grupo],
                 ha="center", va="center", fontsize=24, fontweight="bold")

    y_top_matrix = y_top - group_hdr_h

    # --- Linha extra: C0 (baseline, sem detector e sem catch22/catch24),
    #     uma celula mesclada por grupo (4 colunas cada), pois C0 nao
    #     depende de detector ---
    y_c0 = y_top_matrix - cell_h
    for g_idx, grupo in enumerate(GRUPOS):
        x0 = row_label_w + g_idx * n_cols_grupo * cell_w
        valor_c0 = valores_c0.get(grupo, np.nan)
        cor_c0 = _cor_interpolada(valor_c0, vmin, vmax)
        ax.add_patch(plt.Rectangle(
            (x0, y_c0), n_cols_grupo * cell_w, cell_h,
            facecolor=cor_c0, edgecolor="#CCCCCC", linewidth=1.5
        ))
        texto_c0 = f"{valor_c0:.2f}" if not np.isnan(valor_c0) else "–"
        luminancia_c0 = 0.299 * cor_c0[0] + 0.587 * cor_c0[1] + 0.114 * cor_c0[2]
        cor_texto_c0 = "#1a1a1a" if luminancia_c0 > 0.55 else "#FFFFFF"
        ax.text(x0 + (n_cols_grupo * cell_w) / 2, y_c0 + cell_h / 2,
                 f"C0: {texto_c0}", ha="center", va="center",
                 fontsize=20, color=cor_texto_c0, fontweight="bold")

    # --- linha divisória entre C0 e a linha de C2 ---
    ax.plot([row_label_w, row_label_w + cell_w * n_cols], [y_c0, y_c0],
            color="#1a1a1a", linewidth=1.4, zorder=5)

    # --- Linha extra: C2 (catch22/catch24 puro), uma celula mesclada por
    #     grupo (4 colunas cada), pois C2 nao depende de detector ---
    y_c2 = y_c0 - cell_h
    for g_idx, grupo in enumerate(GRUPOS):
        x0 = row_label_w + g_idx * n_cols_grupo * cell_w
        valor_c2 = valores_c2.get(grupo, np.nan)
        cor_c2 = _cor_interpolada(valor_c2, vmin, vmax)
        ax.add_patch(plt.Rectangle(
            (x0, y_c2), n_cols_grupo * cell_w, cell_h,
            facecolor=cor_c2, edgecolor="#CCCCCC", linewidth=1.5
        ))
        texto_c2 = f"{valor_c2:.2f}" if not np.isnan(valor_c2) else "–"
        luminancia_c2 = 0.299 * cor_c2[0] + 0.587 * cor_c2[1] + 0.114 * cor_c2[2]
        cor_texto_c2 = "#1a1a1a" if luminancia_c2 > 0.55 else "#FFFFFF"
        ax.text(x0 + (n_cols_grupo * cell_w) / 2, y_c2 + cell_h / 2,
                 f"C2: {texto_c2}", ha="center", va="center",
                 fontsize=20, color=cor_texto_c2, fontweight="bold")

    # --- linha divisória entre C2 e a matriz de detectores ---
    ax.plot([row_label_w, row_label_w + cell_w * n_cols], [y_c2, y_c2],
            color="#1a1a1a", linewidth=1.4, zorder=5)

    # --- Linha extra: ORACLE (deteccao perfeita dos drifts), valores
    #     fixos fornecidos diretamente, uma celula por coluna (C1.A,
    #     C3.A diferem entre si, logo nao sao mescladas como C2) ---
    y_oracle = y_c2 - cell_h
    for g_idx, grupo in enumerate(GRUPOS):
        for c, col in enumerate(COLUNAS_HEATMAP):
            x = row_label_w + (g_idx * n_cols_grupo + c) * cell_w
            valor_oracle = VALORES_ORACLE[grupo][col]
            cor_oracle = _cor_interpolada(valor_oracle, vmin, vmax)
            ax.add_patch(plt.Rectangle(
                (x, y_oracle), cell_w, cell_h,
                facecolor=cor_oracle, edgecolor="#CCCCCC", linewidth=1.5
            ))
            luminancia_oracle = (0.299 * cor_oracle[0] + 0.587 * cor_oracle[1]
                                  + 0.114 * cor_oracle[2])
            cor_texto_oracle = "#1a1a1a" if luminancia_oracle > 0.55 else "#FFFFFF"
            ax.text(x + cell_w / 2, y_oracle + cell_h / 2,
                     f"{valor_oracle:.2f}", ha="center", va="center",
                     fontsize=20, color=cor_texto_oracle, fontweight="bold")

    ax.text(row_label_w - 0.15, y_oracle + cell_h / 2, "ORACLE",
             ha="right", va="center", fontsize=21, fontweight="bold")

    # --- linha divisória entre ORACLE e a matriz de detectores ---
    ax.plot([row_label_w, row_label_w + cell_w * n_cols], [y_oracle, y_oracle],
            color="#1a1a1a", linewidth=1.4, zorder=5)

    # --- matriz principal: detectores (linhas) x grupo*coluna (colunas) ---
    for i, detector in enumerate(DETECTORES):
        y = y_oracle - (i + 1) * cell_h

        ax.text(row_label_w - 0.15, y + cell_h / 2, NOME_PARA_LINHA[detector],
                 ha="right", va="center", fontsize=21, fontweight="bold")

        for g_idx, grupo in enumerate(GRUPOS):
            matriz = matrizes[grupo]
            for c, col in enumerate(COLUNAS_HEATMAP):
                x = row_label_w + (g_idx * n_cols_grupo + c) * cell_w
                valor = matriz.loc[detector, col]
                cor = _cor_interpolada(valor, vmin, vmax)
                ax.add_patch(plt.Rectangle(
                    (x, y), cell_w, cell_h,
                    facecolor=cor, edgecolor="#CCCCCC", linewidth=1.5
                ))
                luminancia = 0.299 * cor[0] + 0.587 * cor[1] + 0.114 * cor[2]
                cor_texto = "#1a1a1a" if luminancia > 0.55 else "#FFFFFF"

                rotulo_base = col.replace("_default", "")
                col_ot = f"{rotulo_base}_otimizado"
                valor_ot = matriz.loc[detector, col_ot]
                texto_def = f"{valor:.2f}" if not np.isnan(valor) else "–"
                texto_ot = f"{valor_ot:.2f}" if not np.isnan(valor_ot) else "–"
                ax.text(x + cell_w / 2, y + cell_h / 2,
                         f"{texto_def} → {texto_ot}",
                         ha="center", va="center",
                         fontsize=15, color=cor_texto, fontweight="bold")

    y_bottom_matrix = y_oracle - n_det * cell_h

    # --- linha divisória entre a matriz e o rodapé de colunas ---
    ax.plot([row_label_w, row_label_w + cell_w * n_cols],
            [y_bottom_matrix, y_bottom_matrix],
            color="#1a1a1a", linewidth=1.4, zorder=5)

    # --- Rodapé: nomes das colunas (C1.A / C3.A), repetido dentro de
    #     cada bloco de grupo, ABAIXO da ultima linha de detector ---
    y_footer = y_bottom_matrix - col_hdr_h
    for g_idx, grupo in enumerate(GRUPOS):
        for c, col in enumerate(COLUNAS_HEATMAP):
            x = row_label_w + (g_idx * n_cols_grupo + c) * cell_w
            ax.text(x + cell_w / 2, y_footer + col_hdr_h / 2,
                     ROTULO_COLUNA_HEATMAP[col],
                     ha="center", va="center",
                     fontsize=21, fontweight="bold")

    # --- linha vertical divisória entre o bloco D e o bloco I ---
    x_div = row_label_w + n_cols_grupo * cell_w
    ax.plot([x_div, x_div], [y_footer, y_top], color="#1a1a1a",
             linewidth=1.8, zorder=5)

    # ylim definido apenas agora, com os valores REAIS de y_top/y_footer
    # calculados acima — evita qualquer divergencia entre a formula do
    # ylim e a posicao efetiva dos elementos desenhados.
    ax.set_ylim(y_footer, y_top)

    # --- Colorbar continua na base, mesmo padrao da heatmap dos dados
    #     reais (tabela_heatmap.py): gradiente branco -> COR_BASE,
    #     extend="both" sinalizando saturacao nos extremos ---
    cmap = LinearSegmentedColormap.from_list("metric_cmap", ["#FFFFFF", COR_BASE])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(f"{rotulo_metrica} (%)" if "%" not in rotulo_metrica else rotulo_metrica,
                    fontsize=20)
    cbar.ax.tick_params(labelsize=19)

    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {caminho_saida}")


# ============================================================
# MAIN
# ============================================================

def main():
    pastas = descobrir_pastas_datasets(PASTA_RAIZ, SUBPASTA_EXPERIMENTO)

    print("\nClassificando datasets por grupo (_D_ / _I_)...")
    grupos = agrupar_pastas(pastas)

    for arquivo, rotulo_metrica, sufixo, nome_base in METRICAS:
        print(f"\n{'='*60}")
        print(f"  MÉTRICA: {rotulo_metrica} ({arquivo})")
        print(f"{'='*60}")

        matrizes = calcular_matriz(grupos, arquivo)
        valores_c0 = calcular_c0_por_grupo(grupos, arquivo)
        valores_c2 = calcular_c2_por_grupo(grupos, arquivo)

        for grupo in GRUPOS:
            print(f"\nMatriz [{grupo}] — {rotulo_metrica}:")
            print(matrizes[grupo].round(4).to_string())
            c0_str = f"{valores_c0[grupo]:.4f}" if not np.isnan(valores_c0[grupo]) else "–"
            print(f"  C0 (baseline, média geral): {c0_str}")
            c2_str = f"{valores_c2[grupo]:.4f}" if not np.isnan(valores_c2[grupo]) else "–"
            print(f"  C2 (catch22/catch24, média geral): {c2_str}")

        # Salva CSV combinado (uma coluna 'grupo' identifica a origem;
        # linhas extras 'C0' e 'C2' são adicionadas com o valor médio do
        # grupo)
        partes = []
        for grupo in GRUPOS:
            tmp = matrizes[grupo].copy()
            tmp.insert(0, "grupo", grupo)
            tmp.insert(0, "detector", tmp.index)
            partes.append(tmp.reset_index(drop=True))

            linha_c0 = {col: np.nan for col in COLUNAS_HEATMAP}
            linha_c0["detector"] = "C0"
            linha_c0["grupo"]    = grupo
            linha_c0["C0_media"] = valores_c0[grupo]
            partes.append(pd.DataFrame([linha_c0]))

            linha_c2 = {col: np.nan for col in COLUNAS_HEATMAP}
            linha_c2["detector"] = "C2"
            linha_c2["grupo"]    = grupo
            linha_c2["C2_media"] = valores_c2[grupo]
            partes.append(pd.DataFrame([linha_c2]))
        df_csv = pd.concat(partes, ignore_index=True)

        caminho_csv = os.path.join(PASTA_RAIZ, f"{nome_base}_dados.csv")
        df_csv.to_csv(caminho_csv, index=False)
        print(f"\nDados salvos em: {caminho_csv}")

        caminho_png = os.path.join(PASTA_RAIZ, f"{nome_base}.png")
        gerar_figura_heatmap(matrizes, valores_c0, valores_c2, rotulo_metrica, caminho_png)


if __name__ == "__main__":
    main()