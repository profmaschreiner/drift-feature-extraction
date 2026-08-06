"""
analise_comparativa_shap_mdi.py
================================
Analisa e visualiza os resultados SHAP e MDI (Mean Decrease in Impurity,
importância nativa da Random Forest) gerados pelos scripts de treino,
respondendo a três perguntas:

  1. Qual a importância das features de SCORE (metodologia drift) em C1?
  2. Quais features do catch22 são mais importantes em C2?
  3. O que muda quando combinamos catch22 + drift (C3)?

Esta versão estende a análise original (baseada apenas em SHAP) acrescentando
a importância de variáveis obtida diretamente da Random Forest (MDI), lida de
'feature_importance_mdi.csv', disponível em cada cenário/detector, no mesmo
padrão de pastas do 'shap_mean_abs.csv'.

Critério de agregação do MDI (definido pelo usuário):
  - catch22/24:      média das TOP-N features (mesmo N usado no SHAP, isto é,
                      N = número de features originais/score do dataset).
  - original / score: média de TODAS as features do tipo (sem corte top-N),
                      já que cada uma delas é, por natureza, um conjunto
                      pequeno e completo (uma feature por variável).

Estrutura de pastas esperada:
  <PASTA_RAIZ>/
    <dataset>/
      drift/
        <detector>/
          shap_mean_abs.csv
          feature_importance_mdi.csv
      catch24/
        shap_mean_abs.csv
        feature_importance_mdi.csv
      catch24_drift/
        <detector>/
          shap_mean_abs.csv
          feature_importance_mdi.csv

Configure a seção CONFIGURAÇÕES antes de executar.
"""

import os
from collections import Counter
from pathlib import Path

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

DETECTORES = [
    "ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
    "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED",
]

TOP_N = 15           # features exibidas nos barplots (ranking bruto)

PASTA_SAIDA = os.path.join(PASTA_RAIZ, "analise_comparativa")

SCENARIO_LABELS = {
    "drift":         "C1 — Drift features",
    "catch24":       "C2 — catch24",
    "catch24_drift": "C3 — catch24 + Drift",
}

COR_SCORE  = "#E05C2A"
COR_CATCH  = "#2A7AE0"
COR_ORIG   = "#2EAA5B"

# ---------------------------------------------------------------------------
# Estilo padronizado para as figuras destinadas ao material suplementar
# (Figuras globais de MDI/SHAP): fonte serifada compatível com o corpo do
# artigo (IEEEtran/Times), tamanhos legíveis em largura de uma coluna, e
# barras horizontais (mais adequadas ao layout de duas colunas do documento,
# pois evitam rótulos de dataset rotacionados).
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
ALTURA_MARGEM_IN      = 1.05  # margem fixa (eixo x, rótulo em 2 linhas, legenda)
ALTURA_MINIMA_IN      = 1.6

# Melhor detector por dataset e cenário, baseado no F1-score macro
# dos seus experimentos (tabela_media_f1.csv ou Figuras 13/14 do artigo).
# Se um dataset não estiver listado aqui, o script usa fallback automático
# (detector com maior SHAP top-1) — preencha conforme seus resultados.
# Formato: "nome_dataset": {"drift": "DETECTOR", "catch24_drift": "DETECTOR"}

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




# Cenários exibidos na figura global (C1 e C3 apenas).
CENARIOS_GLOBAIS = ["drift", "catch24_drift"]

# Espaçamento das barras na figura global, por cenário.
# largura : largura visual de cada barra
# gap     : espaço entre barras do mesmo grupo (0 = coladas)
BARCFG = {
    "drift":         {"largura": 0.26, "gap": 0.04},
    "catch24_drift": {"largura": 0.26, "gap": 0.04},
}

# =========================================================
# UTILITÁRIOS GERAIS
# =========================================================

def classificar_feature(nome):
    if nome.startswith("score_"):
        return "score"
    if "__" in nome:
        return "catch"
    return "original"


def cor_feature(tipo):
    return {"score": COR_SCORE, "catch": COR_CATCH, "original": COR_ORIG}[tipo]


def nome_curto(nome, max_len=35):
    return nome if len(nome) <= max_len else nome[:max_len - 1] + "…"


def _carregar_csv_generico(pasta, arquivo, col_valor):
    """Carrega um CSV de duas colunas (feature, valor) da pasta indicada.
    Retorna None se o arquivo não existir."""
    caminho = os.path.join(pasta, arquivo)
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    df.columns = ["feature", col_valor]
    df["tipo"] = df["feature"].apply(classificar_feature)
    return df


def carregar_shap(pasta):
    """Carrega shap_mean_abs.csv da pasta. Retorna None se não existir."""
    return _carregar_csv_generico(pasta, "shap_mean_abs.csv", "shap")


def carregar_mdi(pasta):
    """Carrega feature_importance_mdi.csv da pasta. Retorna None se não existir."""
    return _carregar_csv_generico(pasta, "feature_importance_mdi.csv", "mdi")


def melhor_detector(pasta_cenario, detectores):
    """
    Procura feature_importance_mdi.csv em pasta_cenario/<detector>/ para cada
    detector na lista. Retorna (nome, df) do detector com maior MDI top-1.

    Usado como fallback quando MELHOR_DETECTOR não tem entrada para o
    dataset/cenário. A seleção é baseada em MDI (não em SHAP) porque o
    cálculo de SHAP foi removido do pipeline de experimento por ser caro
    demais computacionalmente — shap_mean_abs.csv não é mais gerado, então
    basear a escolha do detector nele faria `det` ficar sempre None e
    silenciosamente impediria o carregamento do MDI também (que é o que
    realmente existe em disco). Se algum dia o SHAP voltar a ser calculado,
    ele ainda é carregado oportunistamente do detector escolhido, em
    `carregar_dados_dataset` — só não participa mais da escolha.
    """
    melhor_nome = None
    melhor_df   = None
    melhor_val  = -1.0
    for det in detectores:
        df = carregar_mdi(os.path.join(pasta_cenario, det))
        if df is None or df.empty:
            continue
        val = float(df["mdi"].iloc[0])
        if val > melhor_val:
            melhor_val  = val
            melhor_nome = det
            melhor_df   = df
    return melhor_nome, melhor_df


def carregar_dados_dataset(pasta_dataset, dataset_nome, detectores):
    """
    Retorna dict com chaves 'drift', 'catch24', 'catch24_drift',
    cada uma contendo:
      {'detector': str|None, 'df_shap': DataFrame|None, 'df_mdi': DataFrame|None}

    O detector escolhido (fixo via MELHOR_DETECTOR ou fallback via SHAP)
    é o mesmo usado para carregar tanto o SHAP quanto o MDI, garantindo
    que as duas análises sempre se refiram ao mesmo modelo treinado.
    """
    # --- Diagnóstico ---
    subpastas = sorted([
        d for d in os.listdir(pasta_dataset)
        if os.path.isdir(os.path.join(pasta_dataset, d))
    ]) if os.path.isdir(pasta_dataset) else []
    print(f"  Subpastas: {subpastas}")

    dados = {}

    # C2 — catch24 sem detector
    p_c2      = os.path.join(pasta_dataset, "catch24")
    df_c2_shap = carregar_shap(p_c2)
    df_c2_mdi  = carregar_mdi(p_c2)
    dados["catch24"] = {"detector": None, "df_shap": df_c2_shap, "df_mdi": df_c2_mdi}
    print(f"  catch24: shap={'OK' if df_c2_shap is not None else 'NÃO ENCONTRADO'}"
          f" | mdi={'OK' if df_c2_mdi is not None else 'NÃO ENCONTRADO'} [{p_c2}]")

    # C1 — drift
    # Prioridade: MELHOR_DETECTOR → fallback por maior MDI top-1.
    # (A escolha usa MDI, não SHAP — ver docstring de melhor_detector.)
    p_c1 = os.path.join(pasta_dataset, "drift")
    det_fixo_c1 = MELHOR_DETECTOR.get(dataset_nome, {}).get("drift")
    if det_fixo_c1:
        df_c1_mdi = carregar_mdi(os.path.join(p_c1, det_fixo_c1))
        det_c1 = det_fixo_c1 if df_c1_mdi is not None else None
        if df_c1_mdi is None:
            print(f"  [AVISO] drift: detector fixo '{det_fixo_c1}' não encontrado, usando fallback.")
            det_c1, df_c1_mdi = melhor_detector(p_c1, detectores)
    else:
        det_c1, df_c1_mdi = melhor_detector(p_c1, detectores)
    df_c1_shap = carregar_shap(os.path.join(p_c1, det_c1)) if det_c1 else None
    dados["drift"] = {"detector": det_c1, "df_shap": df_c1_shap, "df_mdi": df_c1_mdi}
    fonte_c1 = "fixo" if MELHOR_DETECTOR.get(dataset_nome, {}).get("drift") else "fallback"
    print(f"  drift [{fonte_c1}]: shap={'OK' if df_c1_shap is not None else 'NÃO ENCONTRADO'}"
          f" | mdi={'OK — ' + det_c1 if df_c1_mdi is not None else 'NÃO ENCONTRADO'} [{p_c1}]")

    # C3 — catch24_drift
    # Prioridade: MELHOR_DETECTOR → fallback por maior MDI top-1.
    p_c3 = os.path.join(pasta_dataset, "catch24_drift")
    det_fixo_c3 = MELHOR_DETECTOR.get(dataset_nome, {}).get("catch24_drift")
    if det_fixo_c3:
        df_c3_mdi = carregar_mdi(os.path.join(p_c3, det_fixo_c3))
        det_c3 = det_fixo_c3 if df_c3_mdi is not None else None
        if df_c3_mdi is None:
            print(f"  [AVISO] catch24_drift: detector fixo '{det_fixo_c3}' não encontrado, usando fallback.")
            det_c3, df_c3_mdi = melhor_detector(p_c3, detectores)
    else:
        det_c3, df_c3_mdi = melhor_detector(p_c3, detectores)
    df_c3_shap = carregar_shap(os.path.join(p_c3, det_c3)) if det_c3 else None
    dados["catch24_drift"] = {"detector": det_c3, "df_shap": df_c3_shap, "df_mdi": df_c3_mdi}
    fonte_c3 = "fixo" if MELHOR_DETECTOR.get(dataset_nome, {}).get("catch24_drift") else "fallback"
    print(f"  catch24_drift [{fonte_c3}]: shap={'OK' if df_c3_shap is not None else 'NÃO ENCONTRADO'}"
          f" | mdi={'OK — ' + det_c3 if df_c3_mdi is not None else 'NÃO ENCONTRADO'} [{p_c3}]")

    return dados


# =========================================================
# FIGURA 1 — Barplot top-N por cenário (SHAP e MDI)
# =========================================================

def _figura_barplot_generica(dados, dataset_nome, top_n, pasta_saida,
                              chave_df, col_valor, rotulo_valor, sufixo_arquivo,
                              titulo_base):
    cenarios = ["drift", "catch24", "catch24_drift"]
    validos  = [c for c in cenarios if dados[c][chave_df] is not None]
    if not validos:
        return

    fig, axes = plt.subplots(1, len(validos), figsize=(7 * len(validos), 7))
    if len(validos) == 1:
        axes = [axes]

    for ax, cenario in zip(axes, validos):
        df  = (dados[cenario][chave_df]
                 .sort_values(col_valor, ascending=False)
                 .head(top_n)
                 .copy())
        det = dados[cenario]["detector"]

        titulo = SCENARIO_LABELS.get(cenario, cenario)
        if det:
            titulo += f"\n({det})"

        cores = df["tipo"].map(cor_feature)
        ax.barh(range(len(df)), df[col_valor].iloc[::-1].values,
                color=cores.iloc[::-1].values, edgecolor="none", height=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([nome_curto(f) for f in df["feature"].iloc[::-1]],
                           fontsize=8)
        ax.set_xlabel(rotulo_valor, fontsize=9)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    legenda = [
        mpatches.Patch(color=COR_SCORE, label="Drift score"),
        mpatches.Patch(color=COR_CATCH, label="Catch22"),
        mpatches.Patch(color=COR_ORIG,  label="Original feature"),
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{titulo_base} — {dataset_nome}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, f"{dataset_nome}_{sufixo_arquivo}.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")


def figura_barplot(dados, dataset_nome, top_n, pasta_saida):
    _figura_barplot_generica(
        dados, dataset_nome, top_n, pasta_saida,
        chave_df="df_shap", col_valor="shap", rotulo_valor="Mean |SHAP|",
        sufixo_arquivo="barplot", titulo_base="SHAP Importance",
    )


def figura_barplot_mdi(dados, dataset_nome, top_n, pasta_saida):
    _figura_barplot_generica(
        dados, dataset_nome, top_n, pasta_saida,
        chave_df="df_mdi", col_valor="mdi", rotulo_valor="MDI (Random Forest)",
        sufixo_arquivo="barplot_mdi", titulo_base="MDI Importance (Random Forest)",
    )


# =========================================================
# FIGURA 2 — Eficiência informativa por tipo de feature (SHAP e MDI)
# =========================================================

def n_features_originais(df):
    """
    Retorna n = número de features de score no dataframe.
    Como a metodologia gera exatamente uma feature de score por variável
    original, esse valor é igual ao número de variáveis originais do dataset.
    Usado como K dinâmico para top-n dentro de cada tipo (SHAP) e para o
    corte top-n aplicado apenas ao catch22/24 no caso do MDI.
    """
    return max(1, int((df["tipo"] == "score").sum()))


def shap_medio_por_tipo(df, k=None):
    """
    Retorna dict com |SHAP| médio e desvio padrão das top-k features
    dentro de cada tipo (score / catch / original).

    k = n (número de variáveis originais do dataset), inferido automaticamente
    como o número de features de score presentes no dataframe.

    Justificativa: k=n garante comparação simétrica entre tipos —
    para score e original, top-n é o grupo completo (ambos têm exatamente n
    features); para catch22/24, top-n são as n mais informativas de 24n,
    normalizando a vantagem numérica desse conjunto sem introduzir um
    hiperparâmetro arbitrário.

    Retorna: {"score": (mean, std), "catch": (mean, std), "original": (mean, std)}
    """
    if k is None:
        k = n_features_originais(df)
    resultado = {}
    for tipo in ["score", "catch", "original"]:
        sub = (df.loc[df["tipo"] == tipo, "shap"]
                 .sort_values(ascending=False)
                 .head(k))
        if len(sub) > 0:
            resultado[tipo] = (float(sub.mean()), float(sub.std(ddof=0)))
        else:
            resultado[tipo] = (0.0, 0.0)
    return resultado


def mdi_medio_por_tipo(df, k=None):
    """
    Retorna dict com MDI médio e desvio padrão por tipo (score / catch / original),
    seguindo o critério de agregação definido para o MDI:

      - catch:              média das TOP-k features (k = n_features_originais,
                             mesmo critério do SHAP), pois o conjunto catch22/24
                             tem 24n features e precisa de um corte comparável.
      - score / original:   média de TODAS as features do tipo, sem corte,
                             já que cada uma delas tem exatamente n features
                             (uma por variável original) — não há vantagem
                             numérica a normalizar.

    Retorna: {"score": (mean, std), "catch": (mean, std), "original": (mean, std)}
    """
    if k is None:
        k = n_features_originais(df)
    resultado = {}
    for tipo in ["score", "catch", "original"]:
        sub_completo = df.loc[df["tipo"] == tipo, "mdi"]
        if tipo == "catch":
            sub = sub_completo.sort_values(ascending=False).head(k)
        else:
            sub = sub_completo
        if len(sub) > 0:
            resultado[tipo] = (float(sub.mean()), float(sub.std(ddof=0)))
        else:
            resultado[tipo] = (0.0, 0.0)
    return resultado


def topn_catch_por_dataset(df, col_valor, k=None):
    """
    Retorna lista com os nomes das top-n features catch22/24 de um dataset,
    sendo n = número de variáveis originais (inferido do número de features
    de score). Usado para agregar quais features catch22 se repetem entre
    datasets com comparação justa independente do tamanho do vetor.
    Funciona tanto para SHAP (col_valor='shap') quanto para MDI (col_valor='mdi').
    """
    if k is None:
        k = n_features_originais(df)
    return (df.loc[df["tipo"] == "catch", ["feature", col_valor]]
              .sort_values(col_valor, ascending=False)
              .head(k)["feature"]
              .tolist())


def _figura_importancia_generica(dados, dataset_nome, pasta_saida,
                                  chave_df, funcao_medio, rotulo_eixo,
                                  sufixo_arquivo, titulo_base, nota_topn=True):
    """
    Grouped bar: valor médio ± desvio padrão por feature de cada tipo,
    por cenário. Genérico para SHAP e MDI — a diferença de critério
    (top-k vs. todas as features) fica encapsulada em `funcao_medio`.
    """
    cenarios = ["drift", "catch24", "catch24_drift"]
    validos  = [c for c in cenarios if dados[c][chave_df] is not None]
    if not validos:
        return

    # Determina n a partir do primeiro cenário disponível
    n = n_features_originais(dados[validos[0]][chave_df])

    stats  = []
    labels = []
    for cenario in validos:
        df  = dados[cenario][chave_df]
        det = dados[cenario]["detector"]
        stats.append(funcao_medio(df, k=n))
        label = SCENARIO_LABELS.get(cenario, cenario).split("—")[-1].strip()
        labels.append(label + (f"\n({det})" if det else ""))

    tipos   = ["score", "catch", "original"]
    cores   = [COR_SCORE, COR_CATCH, COR_ORIG]
    n_tipos = len(tipos)
    largura = 0.22
    x       = np.arange(len(validos))

    fig, ax = plt.subplots(figsize=(max(6, len(validos) * 2.5), 4))

    for i, (tipo, cor) in enumerate(zip(tipos, cores)):
        offset = (i - n_tipos / 2 + 0.5) * largura
        vals   = np.array([s[tipo][0] for s in stats])
        errs   = np.array([s[tipo][1] for s in stats])
        bars     = ax.bar(x + offset, vals, width=largura, color=cor,
                        edgecolor="white", label=tipo)
        # Clipa o erro inferior em mean para não cruzar zero
        errs_low = np.minimum(errs, vals)
        ax.errorbar(x + offset, vals,
                    yerr=[errs_low, errs],
                    fmt="none", color="#333333",
                    capsize=3, linewidth=1.0, capthick=1.0)
        for bar, v, e in zip(bars, vals, errs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + e + 0.001,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(rotulo_eixo, fontsize=9)
    if nota_topn:
        titulo = f"{titulo_base} — {dataset_nome} (catch: top-{n}; score/original: all)"
    else:
        titulo = f"{titulo_base} — {dataset_nome} (top-{n} per type)"
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    legenda = [
        mpatches.Patch(color=COR_SCORE, label="Drift score"),
        mpatches.Patch(color=COR_CATCH, label="Catch22"),
        mpatches.Patch(color=COR_ORIG,  label="Original feature"),
    ]
    ax.legend(handles=legenda, loc="upper right", fontsize=8, frameon=False)

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, f"{dataset_nome}_{sufixo_arquivo}.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")


def figura_importancia_shap(dados, dataset_nome, pasta_saida):
    _figura_importancia_generica(
        dados, dataset_nome, pasta_saida,
        chave_df="df_shap", funcao_medio=shap_medio_por_tipo,
        rotulo_eixo="Mean |SHAP| ± SD per feature",
        sufixo_arquivo="shap_medio",
        titulo_base="Informative efficiency by feature type (SHAP)",
        nota_topn=False,
    )


def figura_importancia_mdi(dados, dataset_nome, pasta_saida):
    _figura_importancia_generica(
        dados, dataset_nome, pasta_saida,
        chave_df="df_mdi", funcao_medio=mdi_medio_por_tipo,
        rotulo_eixo="Mean MDI ± SD per feature",
        sufixo_arquivo="mdi_medio",
        titulo_base="Informative efficiency by feature type (MDI — RF)",
        nota_topn=True,
    )


# =========================================================
# FIGURA 3 — Δ rank das features de score: C1 → C3 (SHAP e MDI)
# =========================================================

def _figura_delta_rank_generica(dados, dataset_nome, pasta_saida,
                                 chave_df, col_valor, sufixo_arquivo, titulo_base):
    df_c1 = dados["drift"][chave_df]
    df_c3 = dados["catch24_drift"][chave_df]
    if df_c1 is None or df_c3 is None:
        return

    scores_c1 = (df_c1[df_c1["tipo"] == "score"][["feature", col_valor]]
                   .sort_values(col_valor, ascending=False)
                   .copy())
    if scores_c1.empty:
        return
    scores_c1 = scores_c1.reset_index(drop=True)
    scores_c1["rank_c1"] = scores_c1.index + 1  # rank dentro das score features

    # Rank de cada feature de score dentro das score features de C3
    scores_c3 = (df_c3[df_c3["tipo"] == "score"]
                   .sort_values(col_valor, ascending=False)
                   .reset_index(drop=True))
    rank_c3_map = {row["feature"]: i + 1 for i, row in scores_c3.iterrows()}

    scores_c1["rank_c3"]   = scores_c1["feature"].map(rank_c3_map)
    scores_c1              = scores_c1.dropna(subset=["rank_c3"])
    scores_c1["rank_c3"]   = scores_c1["rank_c3"].astype(int)
    scores_c1["delta"]     = scores_c1["rank_c1"] - scores_c1["rank_c3"]

    if scores_c1.empty:
        return

    scores_c1 = scores_c1.sort_values("rank_c1")
    cores = [COR_SCORE if d >= 0 else "#999999" for d in scores_c1["delta"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(scores_c1) * 0.45)))
    ax.barh(range(len(scores_c1)), scores_c1["delta"].values,
            color=cores, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(scores_c1)))
    ax.set_yticklabels([nome_curto(f, 30) for f in scores_c1["feature"]], fontsize=8)
    ax.set_xlabel(
        "Δ rank (rank in C1 − rank in C3)\n"
        "Positive = rose in C3 | Negative = fell in C3", fontsize=8)
    ax.set_title(
        f"{titulo_base}\n{dataset_nome}: C1 → C3",
        fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, f"{dataset_nome}_{sufixo_arquivo}.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")


def figura_delta_rank(dados, dataset_nome, pasta_saida):
    _figura_delta_rank_generica(
        dados, dataset_nome, pasta_saida,
        chave_df="df_shap", col_valor="shap",
        sufixo_arquivo="delta_rank",
        titulo_base="Rank redistribution (SHAP) — score features",
    )


def figura_delta_rank_mdi(dados, dataset_nome, pasta_saida):
    _figura_delta_rank_generica(
        dados, dataset_nome, pasta_saida,
        chave_df="df_mdi", col_valor="mdi",
        sufixo_arquivo="delta_rank_mdi",
        titulo_base="Rank redistribution (MDI — RF) — score features",
    )


# =========================================================
# FIGURA 4a — Features catch22 mais frequentes no top-5 entre datasets
# =========================================================

def figura_catch_frequencia(resumo_catch, pasta_saida, cenario,
                             top_n=15, sufixo_arquivo="", titulo_extra=""):
    """
    Barplot horizontal com as features catch22/24 que mais aparecem
    no top-n (dentro do tipo catch) entre datasets, para um dado cenário.

    Permite identificar quais propriedades estatísticas do catch22 são
    mais discriminativas nos domínios avaliados e verificar se há
    sobreposição com o que os detectores de drift capturam.
    Genérico para SHAP e MDI.
    """
    if not resumo_catch:
        return

    contagem = Counter(resumo_catch)
    mais_freq = contagem.most_common(top_n)
    if not mais_freq:
        return

    features, freqs = zip(*mais_freq)
    # Extrai só o nome da estatística (parte após __)
    labels_curtos = [f.split("__")[-1] if "__" in f else f for f in features]
    y = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
    ax.barh(y, list(reversed(freqs)),
            color=COR_CATCH, edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(list(reversed(labels_curtos)), fontsize=8)
    ax.set_xlabel(f"Number of datasets in which it appears in the top-{top_n}", fontsize=9)
    titulo_cen = SCENARIO_LABELS.get(cenario, cenario)
    ax.set_title(f"Most frequent catch22 features in the top{titulo_extra} — {titulo_cen}",
                 fontsize=10, fontweight="bold")
    ax.axvline(len(set(resumo_catch)) / top_n, color="#999", linestyle="--",
               linewidth=0.8, label="mean")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(freqs) + 0.5)

    for xi, (feat, freq) in enumerate(zip(reversed(features), reversed(freqs))):
        ax.text(freq + 0.05, xi, feat, va="center", fontsize=6.5,
                color="#555555")

    plt.tight_layout()
    label_arquivo = cenario.replace("/", "_").replace(" ", "_")
    nome_arq = f"catch_top5_frequencia_{label_arquivo}"
    if sufixo_arquivo:
        nome_arq += f"_{sufixo_arquivo}"
    caminho = os.path.join(pasta_saida, f"{nome_arq}.pdf")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {caminho}")


# =========================================================
# FIGURA 4 — Visão global: valor médio por tipo, todos os datasets
# =========================================================

def _label_dataset(dataset_nome, n):
    """Retorna nome abreviado com n entre parênteses: 'Gait (n=6)'."""
    abrev = NOMES_ABREVIADOS.get(dataset_nome, dataset_nome)
    return f"{abrev}({n})"


def _figura_global_generica(resumo, pasta_saida, prefixo_colunas, rotulo_eixo,
                             sufixo_arquivo, titulo_base=None):
    """
    Gera uma figura horizontal por cenário (CENARIOS_GLOBAIS), padronizada
    com o restante do artigo (fonte serifada, largura de uma coluna IEEE),
    mostrando valor médio ± std do top-n (ou de todas, conforme o tipo)
    por tipo de feature — um dataset por linha, barras na horizontal.

    A orientação horizontal evita rotacionar os rótulos de dataset, o que
    é importante no layout de duas colunas do material suplementar. O
    título é omitido propositalmente: no artigo, essa informação é dada
    pela legenda (caption) em LaTeX logo abaixo da figura, como nas demais
    figuras do manuscrito (Fig. 5, 9, 10 etc.); `titulo_base` é mantido
    apenas por compatibilidade com as chamadas existentes.

    Textos em inglês para publicação (artigo em inglês).
    """
    if not resumo:
        return

    df    = pd.DataFrame(resumo)
    tipos = ["score", "catch", "original"]
    cores = [COR_SCORE, COR_CATCH, COR_ORIG]

    nomes_tipo = {
        "score":    "Drift score",
        "catch":    "Catch22",
        "original": "Original feature",
    }

    for cenario in CENARIOS_GLOBAIS:
        sub = df[df["cenario"] == cenario].copy()
        if sub.empty:
            continue

        datasets = list(df["dataset"].unique())
        sub = sub.set_index("dataset")

        labels_y = []
        for d in datasets:
            n_val = int(sub.loc[d, "n"]) if d in sub.index else 0
            labels_y.append(_label_dataset(d, n_val))

        cfg     = BARCFG.get(cenario, {"largura": 0.26, "gap": 0.04})
        largura = cfg["largura"]   # agora usada como altura de cada barra
        gap     = cfg["gap"]

        tipos_ativos = [
            (tipo, cor) for tipo, cor in zip(tipos, cores)
            if sub[f"{prefixo_colunas}_med_{tipo}"].max() > 0
        ]
        n_ativos = len(tipos_ativos)
        passo    = largura + gap
        y        = np.arange(len(datasets))

        altura_fig = max(
            ALTURA_MINIMA_IN,
            len(datasets) * ALTURA_POR_DATASET_IN + ALTURA_MARGEM_IN,
        )

        with plt.rc_context(ESTILO_SUPLEMENTAR):
            fig, ax = plt.subplots(figsize=(LARGURA_COLUNA_IN, altura_fig))

            for i, (tipo, cor) in enumerate(tipos_ativos):
                offset = (i - n_ativos / 2 + 0.5) * passo
                col_m  = f"{prefixo_colunas}_med_{tipo}"
                col_s  = f"{prefixo_colunas}_std_{tipo}"
                vals   = np.array([sub.loc[d, col_m] if d in sub.index else 0.0
                                   for d in datasets])
                errs   = np.array([sub.loc[d, col_s] if d in sub.index else 0.0
                                   for d in datasets])
                ax.barh(y + offset, vals, height=largura, color=cor,
                        edgecolor="white", linewidth=0.4, label=tipo)
                errs_low = np.minimum(errs, vals)
                ax.errorbar(vals, y + offset,
                            xerr=[errs_low, errs],
                            fmt="none", color="#333333",
                            capsize=1.5, linewidth=0.6, capthick=0.6)

            ax.set_yticks(y)
            ax.set_yticklabels(labels_y)
            ax.invert_yaxis()  # primeiro dataset no topo
            ax.set_xlabel(rotulo_eixo, labelpad=6)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

            legenda = [
                mpatches.Patch(color=cor, label=nomes_tipo[tipo])
                for tipo, cor in tipos_ativos
            ]
            # Reserva uma faixa fixa na parte inferior da figura para a
            # legenda, para que ela nunca se sobreponha ao rótulo do eixo x
            # (que tem duas linhas), independentemente do tight_layout.
            plt.tight_layout(rect=[0, 0.14, 1, 1])
            fig.legend(handles=legenda, loc="lower center", ncol=n_ativos,
                       frameon=False, handlelength=1.2,
                       bbox_to_anchor=(0.5, 0.01))

            caminho = os.path.join(pasta_saida, f"global_{sufixo_arquivo}_{cenario}.pdf")
            fig.savefig(caminho, bbox_inches="tight", dpi=300)
            plt.close(fig)
        print(f"  [OK] {caminho}")


def figura_global_shap(resumo_shap, pasta_saida):
    _figura_global_generica(
        resumo_shap, pasta_saida, prefixo_colunas="shap",
        rotulo_eixo=f"Mean |SHAP| per feature (top-{TOP_N})",
        sufixo_arquivo="shap_medio",
    )


def figura_global_mdi(resumo_mdi, pasta_saida):
    _figura_global_generica(
        resumo_mdi, pasta_saida, prefixo_colunas="mdi",
        rotulo_eixo=f"Mean MDI per feature\n(catch: top-{TOP_N} | score/original: all)",
        sufixo_arquivo="mdi_medio",
    )


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

    resumo_proporcao      = []   # SHAP
    resumo_proporcao_mdi   = []   # MDI
    resumo_detalhado       = []   # SHAP (top-n por tipo)
    resumo_detalhado_mdi   = []   # MDI (catch: top-n | score/original: todas)

    # Acumula nomes de features catch22 no top-n por cenário (entre datasets)
    resumo_catch_freq     = {c: [] for c in ["drift", "catch24", "catch24_drift"]}
    resumo_catch_freq_mdi = {c: [] for c in ["drift", "catch24", "catch24_drift"]}

    for dataset_nome in datasets_lista:
        pasta_dataset = os.path.join(PASTA_RAIZ, dataset_nome)
        print(f"{'=' * 55}")
        print(f"Dataset: {dataset_nome}")

        dados = carregar_dados_dataset(pasta_dataset, dataset_nome, DETECTORES)

        tem_algum_dado = any(
            v["df_shap"] is not None or v["df_mdi"] is not None
            for v in dados.values()
        )
        if not tem_algum_dado:
            print(f"  [SKIP] Nenhum dado encontrado.\n")
            continue

        pasta_ds = os.path.join(PASTA_SAIDA, dataset_nome)
        os.makedirs(pasta_ds, exist_ok=True)

        # --- Figuras SHAP (originais) ---
        figura_barplot(dados, dataset_nome, TOP_N, pasta_ds)
        figura_importancia_shap(dados, dataset_nome, pasta_ds)
        figura_delta_rank(dados, dataset_nome, pasta_ds)

        # --- Figuras MDI (novas) ---
        figura_barplot_mdi(dados, dataset_nome, TOP_N, pasta_ds)
        figura_importancia_mdi(dados, dataset_nome, pasta_ds)
        figura_delta_rank_mdi(dados, dataset_nome, pasta_ds)

        for cenario, info in dados.items():
            det = info["detector"]

            # ---------- SHAP ----------
            df_shap = info["df_shap"]
            if df_shap is not None:
                n = n_features_originais(df_shap)
                med = shap_medio_por_tipo(df_shap, k=n)
                resumo_proporcao.append({
                    "dataset":       dataset_nome,
                    "cenario":       cenario,
                    "detector":      det or "",
                    "n":             n,
                    "shap_med_score":    med["score"][0],
                    "shap_std_score":    med["score"][1],
                    "shap_med_catch":    med["catch"][0],
                    "shap_std_catch":    med["catch"][1],
                    "shap_med_original": med["original"][0],
                    "shap_std_original": med["original"][1],
                })

                catch_topn = topn_catch_por_dataset(df_shap, "shap", k=n)
                resumo_catch_freq[cenario].extend(catch_topn)

                for tipo in ["score", "catch", "original"]:
                    sub = (df_shap[df_shap["tipo"] == tipo]
                             .sort_values("shap", ascending=False)
                             .head(n))
                    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
                        resumo_detalhado.append({
                            "dataset":      dataset_nome,
                            "cenario":      cenario,
                            "detector":     det or "",
                            "tipo":         tipo,
                            "n":            n,
                            "rank_no_tipo": rank,
                            "feature":      row["feature"],
                            "shap":         round(row["shap"], 6),
                        })

            # ---------- MDI ----------
            df_mdi = info["df_mdi"]
            if df_mdi is not None:
                n_mdi = n_features_originais(df_mdi)
                med_mdi = mdi_medio_por_tipo(df_mdi, k=n_mdi)
                resumo_proporcao_mdi.append({
                    "dataset":       dataset_nome,
                    "cenario":       cenario,
                    "detector":      det or "",
                    "n":             n_mdi,
                    "mdi_med_score":    med_mdi["score"][0],
                    "mdi_std_score":    med_mdi["score"][1],
                    "mdi_med_catch":    med_mdi["catch"][0],
                    "mdi_std_catch":    med_mdi["catch"][1],
                    "mdi_med_original": med_mdi["original"][0],
                    "mdi_std_original": med_mdi["original"][1],
                })

                catch_topn_mdi = topn_catch_por_dataset(df_mdi, "mdi", k=n_mdi)
                resumo_catch_freq_mdi[cenario].extend(catch_topn_mdi)

                for tipo in ["score", "catch", "original"]:
                    sub_completo = df_mdi[df_mdi["tipo"] == tipo].sort_values("mdi", ascending=False)
                    # catch: corta em top-n | score/original: todas as features
                    sub = sub_completo.head(n_mdi) if tipo == "catch" else sub_completo
                    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
                        resumo_detalhado_mdi.append({
                            "dataset":      dataset_nome,
                            "cenario":      cenario,
                            "detector":     det or "",
                            "tipo":         tipo,
                            "n":            n_mdi,
                            "rank_no_tipo": rank,
                            "feature":      row["feature"],
                            "mdi":          round(row["mdi"], 6),
                        })

            # ---------- Log resumido ----------
            top3_shap = df_shap.sort_values("shap", ascending=False).head(3)["feature"].tolist() if df_shap is not None else []
            top3_mdi  = df_mdi.sort_values("mdi", ascending=False).head(3)["feature"].tolist() if df_mdi is not None else []
            label = SCENARIO_LABELS.get(cenario, cenario)
            print(f"  {label}" + (f" [{det}]" if det else "")
                  + f": shap_top3={top3_shap} | mdi_top3={top3_mdi}")

        print()

    # Figuras de frequência das features catch22 por cenário — SHAP
    for cenario, lista_catch in resumo_catch_freq.items():
        figura_catch_frequencia(lista_catch, PASTA_SAIDA, cenario)

    # Figuras de frequência das features catch22 por cenário — MDI
    for cenario, lista_catch in resumo_catch_freq_mdi.items():
        figura_catch_frequencia(lista_catch, PASTA_SAIDA, cenario,
                                 sufixo_arquivo="mdi", titulo_extra="-n (MDI)")

    figura_global_shap(resumo_proporcao, PASTA_SAIDA)
    figura_global_mdi(resumo_proporcao_mdi, PASTA_SAIDA)

    if resumo_detalhado:
        caminho_csv = os.path.join(PASTA_SAIDA, "resumo_topn_por_tipo_shap.csv")
        pd.DataFrame(resumo_detalhado).to_csv(caminho_csv, index=False)
        print(f"  [OK] {caminho_csv}")

    if resumo_detalhado_mdi:
        caminho_csv = os.path.join(PASTA_SAIDA, "resumo_por_tipo_mdi.csv")
        pd.DataFrame(resumo_detalhado_mdi).to_csv(caminho_csv, index=False)
        print(f"  [OK] {caminho_csv}")

    print("\nAnálise concluída.")
    print(f"Resultados em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()