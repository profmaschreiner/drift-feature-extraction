"""
gerar_resultados_artigo.py
==========================
Tabelas (LaTeX + CSV) e figuras (PDF + PNG) dos datasets sinteticos
controlados, a partir das saidas consolidadas da rodada aninhada. Nao roda
nenhum modelo. Sem selecao de detector: os 9 detectores sao reportados.

Entradas (pasta --entrada):
  resumo_aninhado_por_arquivo_T0_499.csv   F1 pareado por arquivo de teste
  resumo_aninhado_T0_499.csv               cobertura por externo
  alarmes_por_classe.csv                   taxa de alarmes por classe
  alarmes_entrada_transicao.csv            C: alarmes na entrada de T_AB e T_BA (opcional)
  <dataset>/resultados_aninhado/T0_499/... previsoes por amostra (fronteira); sem elas,
                                           usa fronteira_todas_configuracoes.csv

Saidas (pasta --saida):
  fig_f1_absoluto.{pdf,png,csv}     TEXTO PRINCIPAL: macro F1 de C0, C2 (3 janelas), C1 e C3
                                    dos nove detectores e EWMA_VAR, por dataset
  fig_mecanismo.{pdf,png}           TEXTO PRINCIPAL (ver abaixo)
  Demais saidas: material suplementar
  tab_referencia.{tex,csv}          baseline, catch24 por janela, EWMA_VAR
  fig_ganho_detectores.{pdf,png}    ganho de cada detector (sem EWMA_VAR)
                                    sobre a referencia: (a) sozinho vs
                                    baseline, (b) catch24 w130 + score vs
                                    catch24 w130
  fig_mecanismo.{pdf,png}           diferenca de alarmes entre classes por segmento vs
                                    ganho no F1: (a) C1 - C0, (b) C3 - C2 w130; em C,
                                    origem (posturas A, B) -> destino (F1 das transicoes)
  tab_fronteira_C1.{tex,csv}        erros de fronteira (U_alpha, fr, nao reconhecidos;
  tab_fronteira_C3.{tex,csv}        em C, so as transicoes) nas situacoes da Figura
                                    de ganhos com ganho >= 5 p.p., com referencia e
                                    controles do cenario (requer as previsoes por
                                    amostra: rodar no cluster)
  fronteira_todas_configuracoes.csv metricas de fronteira de todas as configuracoes
  fig_fronteira.{pdf,png}           U_alpha x fr em F 210, C 700, H 150 e V 1.5 (C2 nas
                                    tres janelas, C1 e C3 por detector, EWMA_VAR)
  tab_fronteira_resumo_{C1,C3}.{tex,csv}  uma linha por dataset: medianas entre os
                                    detectores selecionados, referencia e controles
  fig_controles.{pdf,png}           Algorithm 1 - EWMA dos alarmes
  tab_supl_f1.{tex,csv}             F1 medio e dp de todas as configuracoes

Uso:
    python3 gerar_resultados_artigo.py --entrada consolidado --saida saida_artigo
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ ordem e rotulos
DATASETS = ["V_r1.25", "V_r1.5", "V_r2.0",
            "F_L1_280", "F_L1_210", "F_L1_105",
            "M_a1_0.75", "M_a1_1.0", "M_a1_1.25",
            "H_g230", "H_g190", "H_g150",
            "C_T100", "C_T300", "C_T700"]          # do menor ao maior contraste (C: transicao)
FATOR = {"V": r"$\sigma_1/\sigma_0$", "F": r"$L_1$", "M": r"$a_1$", "H": r"$g$", "C": r"$T$"}
DETECTORES = ["ADWIN", "CUSUM", "EWMAChart", "GeometricMovingAverage", "HDDMAverage",
              "HDDMWeighted", "KSWIN", "PageHinkley", "SEED"]
CURTO = {"ADWIN": "ADWIN", "CUSUM": "CUSUM", "EWMAChart": "EWMAC",
         "GeometricMovingAverage": "GMA", "HDDMAverage": "HDDMA", "HDDMWeighted": "HDDMW",
         "KSWIN": "KSWIN", "PageHinkley": "PH", "SEED": "SEED", "EWMA_VAR": "EWMA$_{var}$"}

# paleta (instancia de referencia do skill dataviz): divergente azul <-> vermelho,
# ponto medio cinza; categoricas nas tres primeiras posicoes
AZUL, VERMELHO, NEUTRO = "#2a78d6", "#e34948", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d9d8d4"
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#a32b2b", VERMELHO, NEUTRO, AZUL, "#184f95"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "text.color": TINTA, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def rotulo(ds):
    cond = ds.split("_")[0]
    valor = ds.split("_", 1)[1].replace("r", "").replace("L1_", "").replace("a1_", "").replace("g", "").replace("T", "")
    return f"{cond}  {valor}"


# ------------------------------------------------------------------ carga
def carregar(entrada: Path):
    arq = pd.read_csv(entrada / "resumo_aninhado_por_arquivo_T0_499.csv")
    arq = arq[arq["modelo"] == "RF"].copy()
    arq["detector"] = arq["detector"].fillna("")
    f1 = (arq.groupby(["dataset", "bloco", "cenario", "detector"])["f1_pareada"]
          .agg(media="mean", dp="std", n="count").reset_index())
    assert (f1["n"] == 12).all(), "ha configuracoes sem os 12 arquivos de teste"
    res = pd.read_csv(entrada / "resumo_aninhado_T0_499.csv")
    res["detector"] = res["detector"].fillna("")
    cob = res.groupby(["dataset", "bloco", "cenario", "detector"])["cobertura"].mean().reset_index()
    f1 = f1.merge(cob, on=["dataset", "bloco", "cenario", "detector"], how="left")
    alarmes = pd.read_csv(entrada / "alarmes_por_classe.csv")
    return f1, alarmes


def valor(f1, ds, bloco, cen, det=""):
    m = f1[(f1.dataset == ds) & (f1.bloco == bloco) & (f1.cenario == cen) & (f1.detector == det)]
    return float(m["media"].iloc[0]) if len(m) else np.nan


# ------------------------------------------------------------------ tabela de referencia
def tabela_referencia(f1, saida):
    linhas = []
    for ds in DATASETS:
        linhas.append({
            "dataset": ds,
            "baseline": valor(f1, ds, "sem_janela", "baseline"),
            "catch24_w130": valor(f1, ds, "w130", "catch24"),
            "catch24_w300": valor(f1, ds, "w300", "catch24"),
            "catch24_w500": valor(f1, ds, "w500", "catch24"),
        })
    t = pd.DataFrame(linhas)
    t.to_csv(saida / "tab_referencia.csv", index=False)

    tex = [r"\begin{tabular}{llcccc}", r"\hline",
           r" & & C0 & \multicolumn{3}{c}{C2 (catch22)} \\",
           r"Cond. & Factor & Baseline & w130 & w300 & w500 \\ \hline"]
    for cond in "VFMHC":
        sub = t[t.dataset.str.startswith(cond + "_")]
        for i, (_, r) in enumerate(sub.iterrows()):
            cel = [f"{100 * r[c]:.2f}" for c in ["baseline", "catch24_w130", "catch24_w300", "catch24_w500"]]
            pref = cond if i == 0 else ""
            tex.append(f"{pref} & {rotulo(r.dataset).split()[-1]} & " + " & ".join(cel) + r" \\")
        tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    (saida / "tab_referencia.tex").write_text("\n".join(tex) + "\n")
    return t


# ------------------------------------------------------------------ mapa de calor de ganhos
def matriz_ganho(f1, bloco, cen, ref_cen):
    cols = list(DETECTORES)
    m = np.full((len(DATASETS), len(cols)), np.nan)
    for i, ds in enumerate(DATASETS):
        ref = valor(f1, ds, bloco, ref_cen)
        for j, d in enumerate(cols):
            m[i, j] = valor(f1, ds, bloco, cen, d) - ref
    return m, cols


def fmt_pp(v):
    r = int(round(100 * v))
    return "0" if r == 0 else f"{r:+d}"


def desenhar_heatmap(ax, m, cols, titulo, lim, mostrar_y=True, fmt=fmt_pp):
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    # so os nove detectores (o EWMA_VAR fica fora desta figura)
    manter = [j for j, c in enumerate(cols) if c != "EWMA_VAR"]
    m, cols = m[:, manter], [cols[j] for j in manter]
    x = np.arange(len(cols), dtype=float)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.add_patch(plt.Rectangle((x[j] - 0.46, i - 0.46), 0.92, 0.92,
                                       facecolor=CMAP_DIV(norm(np.clip(v, -lim, lim))), edgecolor="none"))
            cor = "white" if abs(v) > 0.6 * lim else TINTA
            ax.text(x[j], i, fmt(v), ha="center", va="center", fontsize=6.2, color=cor)
    for k in [3, 6, 9, 12]:
        ax.axhline(k - 0.5, color="white", lw=2.5)
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_ylim(len(DATASETS) - 0.5, -0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CURTO[c] for c in cols], rotation=60, ha="right")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([rotulo(d) for d in DATASETS] if mostrar_y else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo, loc="left")
    return norm


def figura_ganhos(f1, saida):
    ma, cols = matriz_ganho(f1, "sem_janela", "drift", "baseline")
    mb, _ = matriz_ganho(f1, "w130", "catch24_drift", "catch24")
    lim = 0.40
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw={"wspace": 0.08})
    norm = desenhar_heatmap(axs[0], ma, cols, "(a) C1 $-$ C0 (no window)", lim)
    desenhar_heatmap(axs[1], mb, cols, "(b) C3 $-$ C2 (window 130)", lim, mostrar_y=False)
    cax = fig.add_axes([0.92, 0.30, 0.012, 0.45])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Difference in macro F1-score (×100)")
    cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])
    cb.set_ticklabels(["−40", "−20", "0", "+20", "+40"])
    cb.outline.set_visible(False)
    salvar(fig, saida / "fig_ganho_detectores")
    pd.DataFrame(ma, index=DATASETS, columns=cols).to_csv(saida / "fig_ganho_detectores_a.csv")
    pd.DataFrame(mb, index=DATASETS, columns=cols).to_csv(saida / "fig_ganho_detectores_b.csv")


# ------------------------------------------------------------------ F1 absoluto
CONDICOES = [("V", "V (variance)"), ("F", "F (frequency)"), ("M", "M (magnitude)"),
             ("H", "H (arrangement)"), ("C", "C (context)")]
CINZAS_C2 = {130: "#c9c8c4", 300: "#8a8986", 500: "#3d3c3a"}


def carregar_controle_isolado(entrada: Path, controle="ewma_h", variante="hsel"):
    """F1 medio (12 arquivos) de um controle com a otimizacao isolada
    (controle_ewma_h_sintetico.py) por (dataset, cenario C1/C3, detector):
    controle = "ewma_h" (EWMA dos alarmes) ou "ewmavar_h" (EWMA_VAR com a
    configuracao do Algorithm 1 de cada detector). None se ainda nao existir."""
    arq = entrada / f"{controle}_por_arquivo_T0_499.csv"
    if not arq.exists():
        print(f"[f1_absoluto] {arq} nao encontrado")
        return None
    t = pd.read_csv(arq)
    return t.groupby(["dataset", "cenario", "detector"])[f"f1_{variante}"].mean().to_dict()


def carregar_resumos_isolados(entrada: Path):
    """Resumos dos controles com a otimizacao isolada (controle_ewma_h_sintetico.py),
    indexados por (dataset, cenario C1/C3, detector). F1 e troca em %, como no
    resumo; U_alpha em amostras; fr e nao_rec em fracao."""
    out = {}
    for controle in ("ewma_h", "ewmavar_h"):
        arq = entrada / f"{controle}_resumo_T0_499.csv"
        if arq.exists():
            out[controle] = pd.read_csv(arq).set_index(["dataset", "cenario", "detector"])
        else:
            print(f"[controles] {arq} nao encontrado")
            out[controle] = None
    return out


def linha_isolada(resumos, controle, ds, cenario, det):
    """Metricas de um controle isolado no mesmo formato das tabelas de fronteira."""
    t = resumos.get(controle) if resumos else None
    if t is None or (ds, cenario, det) not in t.index:
        return None
    s = t.loc[(ds, cenario, det)]
    return dict(f1_macro=s["f1_hsel"] / 100,
                f1_transicoes=s["f1_transicoes_hsel"] / 100 if "f1_transicoes_hsel" in s and pd.notna(s["f1_transicoes_hsel"]) else np.nan,
                U_alpha=s["Ua_hsel"], fr=s["fr_hsel"], nao_rec=s["nao_rec_hsel"],
                troca=s["troca_hsel"] / 100 if "troca_hsel" in s and pd.notna(s["troca_hsel"]) else np.nan)


def controles_otimizados(f1):
    """EWMA dos alarmes e EWMA_VAR com otimizacao propria (rodada principal):
    alarmes por detector; EWMA_VAR um valor por dataset e cenario."""
    alarm, var = {}, {}
    for c, (b, cen) in {"C1": ("sem_janela", "drift"), "C3": ("w130", "catch24_drift")}.items():
        for ds in DATASETS:
            var[(ds, c)] = valor(f1, ds, b, cen, "EWMA_VAR")
            for d in DETECTORES:
                alarm[(ds, c, d)] = valor(f1, ds, b, cen, f"EWMA_ALARM_{d}")
    return alarm, var


def figura_f1_absoluto(f1, saida, alarm=None, var=None, nome="fig_f1_absoluto", titulo_var=None):
    """Macro F1 de teste por dataset. Em cada nivel, de cima para baixo: C1 com o
    Algorithm 1, C1 com a EWMA dos alarmes (`alarm`, por detector), C1 com o
    EWMA_VAR, as referencias (C0 e C2 em tres janelas) e as mesmas tres linhas
    para C3 (janela de 130). `var` pode ser um valor por (dataset, cenario)
    (EWMA_VAR otimizado, um losango) ou um por (dataset, cenario, detector)
    (EWMA_VAR com a configuracao de cada detector, nove losangos). Linhas sem
    dados sao omitidas. A linha pontilhada marca a melhor janela do C2; pontos
    a direita dela superam o catch22 em qualquer janela."""
    if var is None:
        var = controles_otimizados(f1)[1]
    var_por_det = len(next(iter(var))) == 3
    linhas = []
    for ds in DATASETS:
        base = dict(dataset=ds, C0=valor(f1, ds, "sem_janela", "baseline"),
                    **{f"C2_w{w}": valor(f1, ds, f"w{w}", "catch24") for w in (130, 300, 500)})
        for c, (b, cen) in {"C1": ("sem_janela", "drift"), "C3": ("w130", "catch24_drift")}.items():
            for det in DETECTORES:
                base[f"{c}_{det}"] = valor(f1, ds, b, cen, det)
                if alarm is not None:
                    base[f"{c}_EWMA_ALARM_{det}"] = alarm.get((ds, c, det), np.nan)
                if var_por_det:
                    base[f"{c}_EWMA_VAR_{det}"] = var.get((ds, c, det), np.nan)
            if not var_por_det:
                base[f"{c}_EWMA_VAR"] = var.get((ds, c), np.nan)
        linhas.append(base)
    t = pd.DataFrame(linhas)
    t.to_csv(saida / f"{nome}.csv", index=False)

    # linhas de cada cenario: Algorithm 1, [EWMA dos alarmes], [EWMA_VAR por detector]
    sub_linhas = ["alg"] + (["alarm"] if alarm is not None else []) + (["var"] if var_por_det else [])
    k = len(sub_linhas)
    passo = 0.42 / (k + 0.5)
    pos = {}
    for j, nome_l in enumerate(sub_linhas):
        pos[("C1", nome_l)] = passo * (k - j)
        pos[("C3", nome_l)] = -passo * (j + 1)
    altura = {1: 2.7, 2: 3.6, 3: 4.4}[k]
    fig, axs = plt.subplots(1, 5, figsize=(7.2, altura), gridspec_kw={"wspace": 0.42})
    for ax, (cond, titulo) in zip(axs, CONDICOES):
        sub = t[t.dataset.str.startswith(cond + "_")].reset_index(drop=True)
        for i, r in sub.iterrows():
            y = len(sub) - 1 - i                       # menor contraste em cima
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f5f4f1", lw=0, zorder=0)
            c2 = [100 * r[f"C2_w{w}"] for w in (130, 300, 500)]
            ax.plot([max(c2)] * 2, [y - 0.47, y + 0.47], color=TINTA2, lw=0.9, ls=(0, (1.5, 1.5)), zorder=1)
            ax.scatter(100 * r.C0, y, marker="|", s=70, color=TINTA, lw=1.4, zorder=3)
            for w, v in zip((130, 300, 500), c2):
                ax.scatter(v, y, marker="s", s=20, color=CINZAS_C2[w], edgecolors=TINTA, linewidths=0.4, zorder=3)
            for c, cor in (("C1", CAT[0]), ("C3", CAT[1])):
                ya = y + pos[(c, "alg")]
                ax.scatter([100 * r[f"{c}_{d}"] for d in DETECTORES], np.full(len(DETECTORES), ya),
                           s=11, color=cor, alpha=0.85, lw=0, zorder=4)
                if alarm is not None:
                    ax.scatter([100 * r[f"{c}_EWMA_ALARM_{d}"] for d in DETECTORES],
                               np.full(len(DETECTORES), y + pos[(c, "alarm")]),
                               s=11, facecolors="none", edgecolors=cor, linewidths=0.7, zorder=4)
                if var_por_det:
                    ax.scatter([100 * r[f"{c}_EWMA_VAR_{d}"] for d in DETECTORES],
                               np.full(len(DETECTORES), y + pos[(c, "var")]),
                               marker="D", s=9, facecolors="none", edgecolors=cor, linewidths=0.6, zorder=4)
                else:
                    ax.scatter(100 * r[f"{c}_EWMA_VAR"], ya, marker="D", s=18, facecolors="white",
                               edgecolors=cor, linewidths=1.0, zorder=5)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([rotulo(d).split()[-1] for d in sub.dataset][::-1])
        ax.set_ylim(-0.5, len(sub) - 0.5)
        ax.set_xlim(15 if cond == "C" else 40, 101)
        ax.set_xticks([20, 40, 60, 80, 100] if cond == "C" else [40, 60, 80, 100])
        ax.set_title(titulo, loc="left")
        ax.grid(axis="x", color=GRADE, lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
    fig.supxlabel("Macro F1-score (%)", y=-0.04 if k == 1 else -0.02, fontsize=7.5)
    c2 = {w: Line2D([], [], marker="s", color=CINZAS_C2[w], mec=TINTA, mew=0.4, ls="", ms=4.5,
                    label=f"C2, w{w}") for w in (130, 300, 500)}
    ref = Line2D([], [], marker="|", color=TINTA, ls="", ms=7, mew=1.4, label="C0")
    melhor = Line2D([], [], color=TINTA2, lw=0.9, ls=(0, (1.5, 1.5)), label="best C2 window")
    cima = [Line2D([], [], marker="o", color=TINTA2, ls="", ms=3.5, label="Algorithm 1")]
    if alarm is not None:
        cima.append(Line2D([], [], marker="o", mfc="none", mec=TINTA2, ls="", ms=3.5, label="EWMA$_{alarm}$"))
    cima.append(Line2D([], [], marker="D", mfc="white" if not var_por_det else "none", mec=TINTA2, ls="",
                       ms=3.8 if not var_por_det else 3.2,
                       label="EWMA$_{var}$" + (" (per detector config.)" if var_por_det else "")))
    cima += [Line2D([], [], color=CAT[0], lw=3, label="C1 (no window)"),
             Line2D([], [], color=CAT[1], lw=3, label="C3 (w130)")]
    baixo = [ref, c2[130], c2[300], c2[500], melhor]
    leg = [h for par in zip(cima, baixo) for h in par]      # matplotlib preenche por coluna
    fig.legend(handles=leg, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.97 if k == 1 else 0.975),
               ncol=5, handletextpad=0.3, columnspacing=1.2, title=titulo_var, title_fontsize=8)
    salvar(fig, saida / nome)
    return t


# ------------------------------------------------------------------ mecanismo
SEGMENTO_2C = 1680          # amostras por segmento de classe em V, F, M e H
SEGMENTO_POSTURA_C = 800     # duracao media das posturas A e B em C
LIMIAR_ASSIMETRIA = 0.5      # C: |dif.| de alarmes por segmento de transicao entre T_BA e T_AB


def figura_mecanismo(f1, alarmes, fr_todas, entrada_tr, saida):
    """Ganho dos drift scores em funcao da diferenca no numero de alarmes entre
    as duas classes que se quer distinguir, por segmento de classe.
    V, F, M e H: entre as duas classes; eixo y no macro F1.
    C: a informacao que distingue T_AB de T_BA vem da classe anterior, entao o
    eixo x usa a diferenca de alarmes entre as posturas A e B (origem) e o eixo
    y, o F1 das transicoes T_AB e T_BA (destino; requer as previsoes por
    amostra). Pontos de C em que o numero de alarmes por segmento de transicao
    difere entre T_AB e T_BA em pelo menos LIMIAR_ASSIMETRIA (inicio e restante
    da transicao, alarmes_entrada_transicao.csv) recebem marcador proprio.
    (a) C1 - C0; (b) C3 - C2, janela de 130 amostras."""
    linhas = []
    for (ds, det), g in alarmes.groupby(["dataset", "detector"]):
        g = g.set_index("classe")
        cond = ds[0]
        if cond in "VFMH":
            if set(g.index) != {0, 1}:
                continue
            r0, r1 = (g.loc[c, "alarmes_total"] / g.loc[c, "amostras"] for c in (0, 1))
            dif = abs(r1 - r0) * SEGMENTO_2C
            g1 = valor(f1, ds, "sem_janela", "drift", det) - valor(f1, ds, "sem_janela", "baseline")
            g3 = valor(f1, ds, "w130", "catch24_drift", det) - valor(f1, ds, "w130", "catch24")
            tipo = cond
        else:
            if fr_todas is None or not {0, 1}.issubset(g.index):
                continue
            rA, rB = (g.loc[c, "alarmes_total"] / g.loc[c, "amostras"] for c in (0, 1))
            dif = abs(rA - rB) * SEGMENTO_POSTURA_C

            def ft(cen, bloco, d=""):
                m = fr_todas[(fr_todas.dataset == ds) & (fr_todas.cenario == cen) &
                             (fr_todas.bloco == bloco) & (fr_todas.detector == d)]
                return float(m["f1_fronteira"].iloc[0]) if len(m) else np.nan
            g1 = ft("drift", "sem_janela", det) - ft("baseline", "sem_janela")
            g3 = ft("catch24_drift", "w130", det) - ft("catch24", "w130")
            tipo = "C_mem"
            if entrada_tr is not None:
                e = entrada_tr[(entrada_tr.dataset == ds) & (entrada_tr.detector == det)].set_index("classe")
                if {2, 3}.issubset(e.index):
                    T = int(ds.split("_T")[1])          # duracao da transicao

                    def por_segmento(c):
                        # alarmes no inicio da transicao + no restante dela, por segmento
                        resto = e.loc[c, "taxa_resto_por_1000"]
                        resto = 0.0 if pd.isna(resto) else resto
                        return (e.loc[c, "alarmes_por_segmento"]
                                + resto * max(T - e.loc[c, "janela_entrada"], 0) / 1000)
                    if abs(por_segmento(3) - por_segmento(2)) >= LIMIAR_ASSIMETRIA:
                        tipo = "C_fro"
        linhas.append({"dataset": ds, "condicao": cond, "detector": det, "tipo": tipo,
                       "dif_alarmes_segmento": dif, "ganho_C1": g1, "ganho_C3": g3})
    t = pd.DataFrame(linhas)
    t.to_csv(saida / "fig_mecanismo.csv", index=False)
    if fr_todas is None:
        print("[mecanismo] sem as metricas por transicao: C fica fora da figura")

    estilos = {"V": (CAT[0], "o", "V (variance)"), "F": (CAT[1], "o", "F (frequency)"),
               "M": (CAT[2], "o", "M (magnitude)"), "H": (TINTA2, "X", "H (arrangement)"),
               "C_mem": (TINTA, "s", "C, transitions with similar alarms"),
               "C_fro": (TINTA, "^", "C, transitions differ in alarms")}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, gridspec_kw={"wspace": 0.25})
    for ax, col, tit in [(axs[0], "ganho_C1", "(a) C1 $-$ C0 (no window)"),
                         (axs[1], "ganho_C3", "(b) C3 $-$ C2 (window 130)")]:
        ax.axhline(0, color=GRADE, lw=0.8, zorder=0)
        for tipo, (cor, mk, nome) in estilos.items():
            s = t[t.tipo == tipo]
            if s.empty:
                continue
            ax.scatter(s["dif_alarmes_segmento"], s[col], s=22, c=cor, marker=mk, alpha=0.9,
                       edgecolors="white", linewidths=0.6, label=nome, zorder=3)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.1, 150)
        ax.set_xticks([0, 1, 10, 100])
        ax.set_xticklabels(["0", "1", "10", "100"])
        ax.set_title(tit, loc="left")
        ax.grid(axis="both", color=GRADE, lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("Macro F1-score gain (C: transitions only)")
    fig.supxlabel("|Difference in alarms per class segment| (C: between classes A and B)", y=-0.04, fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               handletextpad=0.3, columnspacing=1.0)
    salvar(fig, saida / "fig_mecanismo")
    return t


# ------------------------------------------------------------------ fronteira
# ------------------------------------------------------------------ fronteira (filtrada)
TRANSICOES = [2, 3]          # classes T_AB e T_BA no dataset C
LIMIAR_GANHO = 0.05          # mesmo criterio descritivo da Figura de ganhos


def _metricas_arquivo(y, p, alvo, outra=None):
    """Contagens de fronteira (Ward et al., 2011) dos segmentos verdadeiros das
    classes em `alvo`, num arquivo (trecho pareado, ordenado no tempo). O
    primeiro segmento comeca no corte do trecho pareado, nao numa troca real,
    e e ignorado."""
    a = dict(seg=0, rec=0, Ua=0, frag=0, amostras=0, troca=0)
    ini = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
    fim = np.r_[ini[1:], len(y)]
    for k, (i, j) in enumerate(zip(ini, fim)):
        c = y[i]
        if k == 0 or c not in alvo:
            continue
        certo = p[i:j] == c
        a["seg"] += 1
        a["amostras"] += j - i
        if outra is not None:
            a["troca"] += int((p[i:j] == outra[c]).sum())
        if certo.any():
            primeiro = int(np.argmax(certo))
            ultimo = len(certo) - 1 - int(np.argmax(certo[::-1]))
            a["rec"] += 1
            a["Ua"] += primeiro
            a["frag"] += int((~certo[primeiro:ultimo + 1]).sum())
    return a


def _fronteira_dataset(entrada: Path, ds: str):
    """Metricas de fronteira de todas as configuracoes de um dataset, a partir
    das previsoes por amostra (predicoes.csv.gz). Em C, so as transicoes."""
    import json
    from sklearn.metrics import f1_score
    base = entrada / ds / "resultados_aninhado" / "T0_499"
    eh_c = ds.startswith("C_")
    outra = {2: 3, 3: 2} if eh_c else None
    acum, f1s = {}, {}
    for arq in sorted(base.glob("ext*/*/teste/**/predicoes.csv.gz")):
        res = json.loads((arq.parent / "resultado_teste.json").read_text())
        chave = (res["scenario"], arq.relative_to(base).parts[1], res.get("detector_name") or "")
        p = pd.read_csv(arq)
        col = [c for c in p.columns if c.startswith("pred_")][0]
        p = p[p["pareada"]].sort_values(["arquivo", "t"])
        a = acum.setdefault(chave, dict(seg=0, rec=0, Ua=0, frag=0, amostras=0, troca=0))
        for _, g in p.groupby("arquivo"):
            y, pr = g["y"].to_numpy(), g[col].to_numpy()
            classes = TRANSICOES if eh_c else list(np.unique(y))
            for k, v in _metricas_arquivo(y, pr, classes, outra).items():
                a[k] += v
            f1s.setdefault(chave, []).append(f1_score(y, pr, labels=classes, average="macro", zero_division=0))
    linhas = []
    for (cen, bloco, det), a in acum.items():
        linhas.append({"dataset": ds, "cenario": cen, "bloco": bloco, "detector": det,
                       "f1_fronteira": float(np.mean(f1s[(cen, bloco, det)])),
                       "U_alpha": a["Ua"] / a["rec"] if a["rec"] else np.nan,
                       "fr": a["frag"] / a["amostras"] if a["amostras"] else np.nan,
                       "nao_rec": 1 - a["rec"] / a["seg"] if a["seg"] else np.nan,
                       "troca": a["troca"] / a["amostras"] if (eh_c and a["amostras"]) else np.nan,
                       "segmentos": a["seg"]})
    return linhas


def carregar_fronteira(entrada: Path, saida: Path):
    """Calcula as metricas a partir das previsoes (no cluster) e guarda em
    fronteira_todas_configuracoes.csv. Sem previsoes (copia local), usa esse
    arquivo, se existir na entrada ou na saida."""
    if any((entrada / ds / "resultados_aninhado" / "T0_499").exists() for ds in DATASETS):
        linhas = []
        for ds in DATASETS:
            if (entrada / ds / "resultados_aninhado" / "T0_499").exists():
                linhas += _fronteira_dataset(entrada, ds)
                print(f"[fronteira] {ds}", flush=True)
        t = pd.DataFrame(linhas)
        t.to_csv(saida / "fronteira_todas_configuracoes.csv", index=False)
        return t
    for pasta in (entrada, saida):
        arq = pasta / "fronteira_todas_configuracoes.csv"
        if arq.exists():
            t = pd.read_csv(arq)
            t["detector"] = t["detector"].fillna("")
            return t
    print("[fronteira] previsoes por amostra nao encontradas; tabelas de fronteira nao geradas")
    return None


def fronteira(f1, fr_todas, saida, resumos=None):
    """Erros de fronteira nas situacoes da Figura de ganhos com ganho >= 5 p.p.:
    C1 - C0 >= 5 p.p. (painel a) e C3 - C2 w130 >= 5 p.p. (painel b), por
    detector. Cada situacao traz a referencia e os controles do cenario."""
    if fr_todas is None:
        return None
    tabelas = {}

    def linha(ds, cen, bloco, det=""):
        m = fr_todas[(fr_todas.dataset == ds) & (fr_todas.cenario == cen) &
                     (fr_todas.bloco == bloco) & (fr_todas.detector == det)]
        return m.iloc[0] if len(m) else None

    cenarios = {
        "C1": dict(bloco="sem_janela", cen="drift", ref=("baseline", "sem_janela"),
                   metodos=lambda det: [("C0", "baseline", "sem_janela", ""),
                                        ("C1 Alg. 1", "drift", "sem_janela", det),
                                        ("C1 EWMA$_{alarm}$", "iso", "ewma_h", det),
                                        ("C1 EWMA$_{var}$", "iso", "ewmavar_h", det)]),
        "C3": dict(bloco="w130", cen="catch24_drift", ref=("catch24", "w130"),
                   metodos=lambda det: [("C2 w130", "catch24", "w130", ""),
                                        ("C3 Alg. 1", "catch24_drift", "w130", det),
                                        ("C3 EWMA$_{alarm}$", "iso", "ewma_h", det),
                                        ("C3 EWMA$_{var}$", "iso", "ewmavar_h", det),
                                        ("C2 w300", "catch24", "w300", ""),
                                        ("C2 w500", "catch24", "w500", "")]),
    }
    for nome, cfg in cenarios.items():
        linhas = []
        for ds in DATASETS:
            ref = valor(f1, ds, cfg["ref"][1], cfg["ref"][0])
            for det in DETECTORES:
                ganho = valor(f1, ds, cfg["bloco"], cfg["cen"], det) - ref
                if not ganho >= LIMIAR_GANHO:
                    continue
                for metodo, cen, bloco, d in cfg["metodos"](det):
                    if cen == "iso":
                        # controle com a configuracao do Algorithm 1 deste detector
                        iso = linha_isolada(resumos, bloco, ds, nome, d)
                        if iso is not None:
                            linhas.append({"dataset": ds, "detector": det, "ganho_figura": ganho,
                                           "metodo": metodo, **iso})
                        continue
                    r = linha(ds, cen, bloco, d)
                    if r is None:
                        continue
                    linhas.append({"dataset": ds, "detector": det, "ganho_figura": ganho, "metodo": metodo,
                                   "f1_macro": valor(f1, ds, bloco, cen, d),
                                   "f1_transicoes": r.f1_fronteira if ds.startswith("C_") else np.nan,
                                   "U_alpha": r.U_alpha, "fr": r.fr, "nao_rec": r.nao_rec, "troca": r.troca})
        t = pd.DataFrame(linhas)
        t.to_csv(saida / f"tab_fronteira_{nome}.csv", index=False)
        # LaTeX (longtable): uma situacao por bloco de linhas
        tex = [r"\begin{longtable}{lllrrrrr}", r"\hline",
               r"Dataset & Detector & Method & F1 (\%) & $U_\alpha$ & fr & Not rec. (\%) & Swap (\%) \\ \hline",
               r"\endhead"]
        for (ds, det), g in t.groupby(["dataset", "detector"], sort=False):
            for k, (_, r) in enumerate(g.iterrows()):
                f1v = r.f1_transicoes if ds.startswith("C_") else r.f1_macro
                cel = [f"{100 * f1v:.2f}",
                       "--" if pd.isna(r.U_alpha) else f"{r.U_alpha:.0f}",
                       f"{r.fr:.3f}", f"{100 * r.nao_rec:.1f}",
                       "--" if pd.isna(r.troca) else f"{100 * r.troca:.1f}"]
                pref = f"{rotulo(ds)} & {CURTO[det]}" if k == 0 else " & "
                tex.append(f"{pref} & {r.metodo} & " + " & ".join(cel) + r" \\")
            tex.append(r"\hline")
        tex.append(r"\end{longtable}")
        (saida / f"tab_fronteira_{nome}.tex").write_text("\n".join(tex) + "\n")
        print(f"[fronteira] {nome}: {t.groupby(['dataset', 'detector']).ngroups} situacoes", flush=True)
        tabelas[nome] = t
    return tabelas


# ------------------------------------------------------------------ curva de acerto em torno da troca de classe
CURVA_XMAX, CURVA_PASSO = 700, 10                       # ate o fim da transicao mais longa (C 700)
PAINEIS_CURVA = ["F_L1_210", "H_g150", "C_T700"]        # texto: frequencia, arranjo, contexto
DATASETS_CURVA = ["F_L1_280", "F_L1_210", "F_L1_105", "H_g230", "H_g190", "H_g150",
                  "C_T100", "C_T300", "C_T700"]         # suplementar: todos os niveis de F, H e C
# Em C, as curvas terminam no fim da transicao (100, 300 e 700 amostras).


def _configs_curva(f1, ds):
    """Configuracoes da curva: catch22 nas tres janelas e os detectores com ganho
    >= 5 p.p. (mesmo criterio da analise de fronteira) em C1 e em C3."""
    cfg = {("catch24", f"w{w}", ""): f"C2 w{w}" for w in (130, 300, 500)}
    for nome, bloco, cen, ref_cen in [("C1", "sem_janela", "drift", "baseline"),
                                      ("C3", "w130", "catch24_drift", "catch24")]:
        ref = valor(f1, ds, bloco, ref_cen)
        for det in DETECTORES:
            if valor(f1, ds, bloco, cen, det) - ref >= LIMIAR_GANHO:
                cfg[(cen, bloco, det)] = nome
    return cfg


def _curvas_dataset(entrada: Path, ds: str, cfg: dict):
    """Fracao de acertos por posicao desde o inicio de cada segmento verdadeiro
    (trecho pareado; o primeiro segmento de cada arquivo e ignorado, pois comeca
    no corte e nao numa troca real). Em C, so os segmentos de transicao."""
    import json
    base = entrada / ds / "resultados_aninhado" / "T0_499"
    eh_c = ds.startswith("C_")
    nb = CURVA_XMAX // CURVA_PASSO
    acum = {}
    for arq in sorted(base.glob("ext*/*/teste/**/predicoes.csv.gz")):
        res = json.loads((arq.parent / "resultado_teste.json").read_text())
        chave = (res["scenario"], arq.relative_to(base).parts[1], res.get("detector_name") or "")
        if chave not in cfg:
            continue
        p = pd.read_csv(arq)
        col = [c for c in p.columns if c.startswith("pred_")][0]
        p = p[p["pareada"]].sort_values(["arquivo", "t"])
        soma, n = acum.setdefault(chave, (np.zeros(nb), np.zeros(nb)))
        for _, g in p.groupby("arquivo"):
            y, pr = g["y"].to_numpy(), g[col].to_numpy()
            ini = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
            fim = np.r_[ini[1:], len(y)]
            for k, (i, j) in enumerate(zip(ini, fim)):
                if k == 0 or (eh_c and y[i] not in TRANSICOES):
                    continue
                certo = (pr[i:j] == y[i])[:CURVA_XMAX].astype(float)
                b = np.arange(len(certo)) // CURVA_PASSO
                soma += np.bincount(b, weights=certo, minlength=nb)[:nb]
                n += np.bincount(b, minlength=nb)[:nb]
    linhas = []
    for chave, (soma, n) in acum.items():
        for k in range(nb):
            if n[k]:
                linhas.append({"dataset": ds, "metodo": cfg[chave], "detector": chave[2],
                               "x": (k + 0.5) * CURVA_PASSO, "acerto": soma[k] / n[k], "n": int(n[k])})
    return linhas


def carregar_curvas(f1, entrada: Path, saida: Path):
    """Calcula as curvas a partir das previsoes (no cluster) e guarda em
    curvas_fronteira.csv; sem previsoes, usa esse arquivo, se existir."""
    if any((entrada / ds / "resultados_aninhado" / "T0_499").exists() for ds in DATASETS_CURVA):
        linhas = []
        for ds in DATASETS_CURVA:
            if (entrada / ds / "resultados_aninhado" / "T0_499").exists():
                linhas += _curvas_dataset(entrada, ds, _configs_curva(f1, ds))
                print(f"[curvas] {ds}", flush=True)
        t = pd.DataFrame(linhas)
        t.to_csv(saida / "curvas_fronteira.csv", index=False)
        return t
    for pasta in (entrada, saida):
        arq = pasta / "curvas_fronteira.csv"
        if arq.exists():
            t = pd.read_csv(arq)
            t["detector"] = t["detector"].fillna("")
            return t
    print("[curvas] previsoes por amostra nao encontradas; curvas nao geradas")
    return None


def figura_curvas(curvas, saida, datasets, nome, ncol=3):
    """Fracao de acertos desde o inicio do segmento: catch22 nas tres janelas e a
    mediana, entre os detectores selecionados, de C1 e de C3."""
    if curvas is None or curvas.empty:
        return
    nlin = int(np.ceil(len(datasets) / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=(7.2, 2.3 * nlin), sharey=True, squeeze=False,
                            gridspec_kw={"wspace": 0.10, "hspace": 0.45})
    estilos = {"C2 w130": dict(color=CINZAS_C2[130], lw=1.4), "C2 w300": dict(color=CINZAS_C2[300], lw=1.4),
               "C2 w500": dict(color=CINZAS_C2[500], lw=1.4),
               "C1": dict(color=CAT[0], lw=1.8), "C3": dict(color=CAT[1], lw=1.8)}
    rotulos = {"C2 w130": "C2, w130", "C2 w300": "C2, w300", "C2 w500": "C2, w500",
               "C1": "C1, median of detectors", "C3": "C3 (w130), median of detectors"}
    linhas = []
    for ax, ds in zip(axs.flat, datasets):
        t = curvas[curvas.dataset == ds]
        for met, est in estilos.items():
            m = t[t.metodo == met]
            if m.empty:
                continue
            c = m.groupby("x").acerto.median()
            ax.plot(c.index, 100 * c.values, **est, label=rotulos[met])
            nd = m.detector.nunique() if met in ("C1", "C3") else None
            linhas.append({"dataset": ds, "metodo": met, "n_detectores": nd,
                           **{f"x{int(x)}": round(100 * v, 2) for x, v in c.items()}})
        ax.set_title(rotulo(ds) + (" (transitions)" if ds.startswith("C_") else ""), loc="left")
        ax.set_xlim(0, CURVA_XMAX)
        ax.set_ylim(0, 100)
        ax.grid(color=GRADE, lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    for ax in axs.flat[len(datasets):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Correct predictions (%)")
    fig.supxlabel("Samples since the start of the segment", y=0.0 if nlin > 1 else -0.06, fontsize=7.5)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02 if nlin > 1 else 1.10))
    salvar(fig, saida / nome)
    pd.DataFrame(linhas).to_csv(saida / f"{nome}.csv", index=False)


PAINEIS_FRONTEIRA = ["F_L1_210", "C_T700", "H_g150", "V_r1.5"]   # frequencia, contexto, arranjo, local


def figura_fronteira(fr_todas, tabelas, saida):
    """Atraso (U_alpha) x fragmentacao (fr) em 4 datasets: catch22 (C2) nas tres
    janelas, ligadas por uma linha; um ponto por detector em C1 e em C3 nas
    situacoes com ganho >= 5 p.p.; EWMA_VAR em C1 e em C3. Em C, so as
    transicoes."""
    if fr_todas is None or tabelas is None:
        return

    def ft(ds, cen, bloco, det=""):
        m = fr_todas[(fr_todas.dataset == ds) & (fr_todas.cenario == cen) &
                     (fr_todas.bloco == bloco) & (fr_todas.detector == det)]
        return m.iloc[0] if len(m) else None

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, ds in zip(axs, PAINEIS_FRONTEIRA):
        c2 = [ft(ds, "catch24", w) for w in ["w130", "w300", "w500"]]
        c2 = [(w, r) for w, r in zip(["w130", "w300", "w500"], c2) if r is not None]
        ax.plot([r.U_alpha for _, r in c2], [100 * r.fr for _, r in c2], color=TINTA, lw=0.8, zorder=2)
        for w, r in c2:
            ax.scatter(r.U_alpha, 100 * r.fr, s=26, c=TINTA, marker="s", zorder=4, edgecolors="white", linewidths=0.6)
            dx, ha = {"w130": ((-4, -10), "right"), "w300": ((0, 6), "center"), "w500": ((4, -10), "left")}[w]
            ax.annotate(w, (r.U_alpha, 100 * r.fr), textcoords="offset points", xytext=dx, ha=ha, fontsize=6,
                        color=TINTA, zorder=5)
        for nome, cor, cen, bloco in [("C1", CAT[0], "drift", "sem_janela"), ("C3", CAT[1], "catch24_drift", "w130")]:
            t = tabelas.get(nome)
            if t is not None:
                s = t[(t.dataset == ds) & (t.metodo == f"{nome} Alg. 1")]
                ax.scatter(s["U_alpha"], 100 * s["fr"], s=18, c=cor, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.6)
            if t is not None:
                ev = t[(t.dataset == ds) & (t.metodo == f"{nome} EWMA$_{{var}}$")]
                ax.scatter(ev["U_alpha"], 100 * ev["fr"], s=22, facecolors="white", edgecolors=cor, linewidths=1.1,
                           marker="D", zorder=4)
        titulo = rotulo(ds) + (" (transitions)" if ds.startswith("C_") else "")
        ax.set_title(titulo, loc="left")
        ax.grid(color=GRADE, lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.08)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("Fragmentation rate, fr (%)")
    fig.supxlabel(r"$U_\alpha$, delay to recognize the new class (samples)", y=-0.06, fontsize=7.5)
    leg = [Line2D([], [], color=TINTA, marker="s", lw=0.8, ms=4, label="C2, catch22 (w130, w300, w500)"),
           Line2D([], [], color=CAT[0], marker="o", lw=0, ms=4, label="C1, per detector"),
           Line2D([], [], color=CAT[1], marker="o", lw=0, ms=4, label="C3 (w130), per detector"),
           Line2D([], [], color=TINTA2, marker="D", lw=0, ms=4, markerfacecolor="white", label="EWMA$_{var}$, same configuration (C1, C3)")]
    fig.legend(handles=leg, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    salvar(fig, saida / "fig_fronteira")


def tabela_fronteira_resumo(tabelas, saida):
    """Tabela de fronteira em formato longo (metodos nas linhas), so F, H e C,
    em que a informacao esta alem da janela. Para Algorithm 1 e EWMA_var, mediana entre os n detectores selecionados (ganho >= 5 p.p.);
    para as referencias, o valor da configuracao. F1, fr, nao reconhecidos e
    troca em %; U_alpha em amostras. Em C, F1 e metricas so das transicoes."""
    if tabelas is None:
        return
    metodos = {"C1": [("C0", "C0", False), ("Algorithm~1", "C1 Alg. 1", True),
                      ("EWMA$_{var}$", "C1 EWMA$_{var}$", True)],
               "C3": [("C2 w130", "C2 w130", False), ("Algorithm~1", "C3 Alg. 1", True), ("EWMA$_{var}$", "C3 EWMA$_{var}$", True),
                      ("C2 w500", "C2 w500", False)]}
    for nome, mets in metodos.items():
        t = tabelas.get(nome)
        if t is None or t.empty:
            continue
        t = t[t.dataset.str[0].isin(list("FHC"))].copy()
        t["f1"] = np.where(t.dataset.str.startswith("C_"), t["f1_transicoes"], t["f1_macro"])
        linhas, tex = [], [r"\begin{tabular}{lclrrrrr}", r"\hline",
                           r"Dataset & $n$ & Method & F1 (\%) & $U_\alpha$ & fr (\%) & Not rec. (\%) & Swap (\%) \\ \hline"]
        for ds in DATASETS:
            g = t[t.dataset == ds]
            if g.empty:
                continue
            n = g.detector.nunique()
            k = 0
            for rot, metodo, mediana in mets:
                m = g[g.metodo == metodo]
                if m.empty:
                    continue
                a = m.median(numeric_only=True) if mediana else m.iloc[0]
                eh_c = ds.startswith("C_")
                linhas.append({"dataset": ds, "n": n, "metodo": rot, "f1": 100 * a["f1"], "U_alpha": a["U_alpha"],
                               "fr": 100 * a["fr"], "nao_rec": 100 * a["nao_rec"],
                               "troca": 100 * a["troca"] if eh_c else np.nan})
                cel = [f"{100 * a['f1']:.2f}", "--" if pd.isna(a["U_alpha"]) else f"{a['U_alpha']:.0f}",
                       f"{100 * a['fr']:.1f}", f"{100 * a['nao_rec']:.1f}",
                       f"{100 * a['troca']:.1f}" if eh_c and pd.notna(a["troca"]) else "--"]
                pref = f"{rotulo(ds)} & {n}" if k == 0 else " & "
                tex.append(f"{pref} & {rot} & " + " & ".join(cel) + r" \\")
                k += 1
            tex.append(r"\hline")
        tex.append(r"\end{tabular}")
        pd.DataFrame(linhas).to_csv(saida / f"tab_fronteira_resumo_{nome}.csv", index=False)
        (saida / f"tab_fronteira_resumo_{nome}.tex").write_text("\n".join(tex) + "\n")


def figura_controles(resumos, saida):
    """Algorithm 1 menos cada controle com a otimizacao isolada (mesma
    configuracao e mesmo horizonte de memoria de cada detector), por detector e
    dataset. Linha de cima: EWMA dos alarmes. Linha de baixo: EWMA_VAR.
    Colunas: C1 (sem janela) e C3 (janela de 130)."""
    if not resumos or resumos.get("ewmavar_h") is None:
        print("[controles] resumos isolados ausentes; fig_controles nao gerada")
        return None
    # So o EWMA_var: a equivalencia com a EWMA dos alarmes ja aparece na figura do F1 absoluto
    linhas_fig = [("ewmavar_h", "EWMA$_{var}$", 0.40, [-0.4, -0.2, 0, 0.2, 0.4], ["−40", "−20", "0", "+20", "+40"])]
    fig, axs = plt.subplots(len(linhas_fig), 2, figsize=(6.6, 4.3 * len(linhas_fig)), squeeze=False,
                            gridspec_kw={"wspace": 0.08, "hspace": 0.32})
    letras = iter("abcd")
    mats = {}
    for lin, (controle, rot, lim, ticks, tlab) in enumerate(linhas_fig):
        t = resumos[controle]
        for col, (cenario, tit) in enumerate([("C1", "C1 (no window)"), ("C3", "C3 (window 130)")]):
            m = np.full((len(DATASETS), len(DETECTORES)), np.nan)
            for i, ds in enumerate(DATASETS):
                for j, d in enumerate(DETECTORES):
                    if (ds, cenario, d) in t.index:
                        r = t.loc[(ds, cenario, d)]
                        m[i, j] = (r["f1_alg1"] - r["f1_hsel"]) / 100
            mats[(controle, cenario)] = m
            norm = desenhar_heatmap(axs[lin, col], m, list(DETECTORES),
                                    f"({next(letras)}) Algorithm 1 $-$ {rot}, {tit}", lim, mostrar_y=(col == 0))
            pd.DataFrame(m, index=DATASETS, columns=DETECTORES).to_csv(saida / f"fig_controles_{controle}_{cenario}.csv")
        pos = axs[lin, 1].get_position()
        cax = fig.add_axes([0.92, pos.y0 + 0.2 * pos.height, 0.012, 0.6 * pos.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV), cax=cax)
        cb.set_label(f"Algorithm 1 $-$ {rot} (macro F1 ×100)")
        cb.set_ticks(ticks)
        cb.set_ticklabels(tlab)
        cb.outline.set_visible(False)
    salvar(fig, saida / "fig_controles")
    return mats


# ------------------------------------------------------------------ suplementar
def tabela_suplementar(f1, saida):
    t = f1.copy()
    t["ordem"] = t.dataset.map({d: i for i, d in enumerate(DATASETS)})
    t = t.sort_values(["ordem", "bloco", "cenario", "detector"]).drop(columns=["ordem", "n"])
    t.to_csv(saida / "tab_supl_f1.csv", index=False)
    # LaTeX: por dataset, linhas = detector/controle; colunas = blocos (score sozinho e catch24 + score)
    nomes = DETECTORES + ["EWMA_VAR"] + [f"EWMA_ALARM_{d}" for d in DETECTORES]
    # sem \caption: a legenda e a numeracao (Table S-n) ficam no suplementar, via \stabcap
    tex = [r"\begin{longtable}{ll" + "c" * 4 + "}",
           r"\hline", r"Dataset & Features & No window & w130 & w300 & w500 \\ \hline", r"\endfirsthead",
           r"\hline", r"Dataset & Features & No window & w130 & w300 & w500 \\ \hline", r"\endhead"]
    for ds in DATASETS:
        def cel(bloco, cen, det=""):
            m = t[(t.dataset == ds) & (t.bloco == bloco) & (t.cenario == cen) & (t.detector == det)]
            return f"{100 * m.media.iloc[0]:.2f} $\\pm$ {100 * m.dp.iloc[0]:.2f}" if len(m) else "--"
        tex.append(f"{rotulo(ds)} & C0 / C2 (catch22) & {cel('sem_janela', 'baseline')} & "
                   f"{cel('w130', 'catch24')} & {cel('w300', 'catch24')} & {cel('w500', 'catch24')} \\\\")
        for d in nomes:
            nome = CURTO.get(d, d.replace("EWMA_ALARM_", "EWMA$_{alarm}$ ")).replace(
                "EWMA$_{alarm}$ " + d.replace("EWMA_ALARM_", ""), "EWMA$_{alarm}$ " + CURTO.get(d.replace("EWMA_ALARM_", ""), ""))
            tex.append(f" & {nome} & {cel('sem_janela', 'drift', d)} & {cel('w130', 'catch24_drift', d)} & "
                       f"{cel('w300', 'catch24_drift', d)} & {cel('w500', 'catch24_drift', d)} \\\\")
        tex.append(r"\hline")
    tex.append(r"\end{longtable}")
    (saida / "tab_supl_f1.tex").write_text("\n".join(tex) + "\n")


def tabela_borda(f1, entrada, saida):
    """Taxa de acerto (%) no inicio de cada arquivo, por faixa de amostras, e
    cobertura do C2 (% das amostras dos arquivos completos com previsao).
    C1 = mediana entre os detectores que nao sao o GMA (sem selecao)."""
    b = pd.read_csv(entrada / "tabela_borda_T0_499.csv")
    b["detector"] = b["detector"].fillna("")
    dets = [d for d in DETECTORES if d != "GeometricMovingAverage"]
    faixas = ["0_130", "130_300", "300_500"]
    linhas = []
    for ds in DATASETS:
        g = b[b.dataset == ds]
        r = {"dataset": ds}
        for f in faixas:
            r[f"C0_{f}"] = g[g.cenario == "baseline"][f"acerto_{f}"].iloc[0]
            r[f"C1_mediana_{f}"] = g[(g.cenario == "drift") & (g.detector.isin(dets))][f"acerto_{f}"].median()
            for w in ["w130", "w300", "w500"]:
                m = g[(g.cenario == "catch24") & (g.bloco == w)].iloc[0]
                # faixa so parcialmente coberta (janela ainda incompleta): nao reporta
                r[f"C2_{w}_{f}"] = m[f"acerto_{f}"] if m[f"cobertura_{f}"] >= 0.99 else np.nan
        for w in ["w130", "w300", "w500"]:
            m = f1[(f1.dataset == ds) & (f1.bloco == w) & (f1.cenario == "catch24")]
            r[f"cobertura_C2_{w}"] = float(m["cobertura"].iloc[0])
        linhas.append(r)
    t = pd.DataFrame(linhas)
    t.to_csv(saida / "tab_supl_borda.csv", index=False)
    cols = [("0_130", ["C0", "C1_mediana"]), ("130_300", ["C0", "C2_w130", "C1_mediana"]),
            ("300_500", ["C0", "C2_w130", "C2_w300", "C1_mediana"])]
    nomes = {"C0": "C0", "C2_w130": "C2 w130", "C2_w300": "C2 w300", "C1_mediana": "C1"}
    tex = [r"\begin{tabular}{ll" + "c" * 12 + "}", r"\hline",
           r" & & \multicolumn{2}{c}{0--129} & \multicolumn{3}{c}{130--299} & \multicolumn{4}{c}{300--499}"
           r" & \multicolumn{3}{c}{Coverage of C2} \\",
           "Cond. & Factor & " + " & ".join(nomes[c] for _, cs in cols for c in cs)
           + r" & w130 & w300 & w500 \\ \hline"]
    for cond in "VFMHC":
        sub = t[t.dataset.str.startswith(cond + "_")]
        for i, (_, r) in enumerate(sub.iterrows()):
            cel = []
            for f, cs in cols:
                for c in cs:
                    v = r[f"{c}_{f}"]
                    cel.append("--" if pd.isna(v) else f"{100 * v:.2f}")
            cel += [f"{100 * r[f'cobertura_C2_{w}']:.2f}" for w in ["w130", "w300", "w500"]]
            tex.append(f"{cond if i == 0 else ''} & {rotulo(r.dataset).split()[-1]} & " + " & ".join(cel) + r" \\")
        tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    (saida / "tab_supl_borda.tex").write_text("\n".join(tex) + "\n")


def salvar(fig, base: Path):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ falsos alarmes em V (suplementar S1)
def tabela_falsos_alarmes_V(alarmes, saida):
    """Taxa de falsos alarmes por 1.000 amostras em cada classe da condicao V
    (alarmes_por_classe.csv, de contar_alarmes.py). Em V nao ha saltos de
    media, e a taxa de falsos alarmes estima 1/ARL0 do detector com a variancia
    de cada classe. Classe 0 = menor variancia, classe 1 = maior (sigma1/sigma0)."""
    v = alarmes[alarmes.dataset.str.startswith("V_")]
    niveis = [d for d in DATASETS if d.startswith("V_")]
    linhas, tex = [], [r"\begin{tabular}{l" + "rr" * len(niveis) + "}", r"\hline",
                        " & " + " & ".join(r"\multicolumn{2}{c}{V " + rotulo(d).split()[-1] + "}" for d in niveis) + r" \\",
                        "Detector" + r" & $\sigma_0$ & $\sigma_1$" * len(niveis) + r" \\ \hline"]
    for det in DETECTORES:
        cel = []
        for d in niveis:
            for c in (0, 1):
                m = v[(v.dataset == d) & (v.detector == det) & (v.classe == c)]
                x = float(m["taxa_falsos_por_1000"].iloc[0]) if len(m) else np.nan
                arl = float(m["ARL0_estimado"].iloc[0]) if len(m) else np.nan
                linhas.append({"dataset": d, "detector": det, "classe": c,
                               "falsos_por_1000": x, "ARL0_estimado": arl})
                cel.append("--" if np.isnan(x) else f"{x:.2f}")
        tex.append(CURTO[det] + " & " + " & ".join(cel) + r" \\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    pd.DataFrame(linhas).to_csv(saida / "tab_falsos_alarmes_V.csv", index=False)
    (saida / "tab_falsos_alarmes_V.tex").write_text("\n".join(tex) + "\n")
    print("[falsos V] tab_falsos_alarmes_V.{tex,csv}", flush=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--entrada", type=Path, default=Path("datasetsNew/sintetico/fatorial"))
    pa.add_argument("--saida", type=Path, default=Path("saida_artigo"))
    a = pa.parse_args()
    a.saida.mkdir(parents=True, exist_ok=True)
    f1, alarmes = carregar(a.entrada)
    tabela_referencia(f1, a.saida)
    tabela_falsos_alarmes_V(alarmes, a.saida)
    # FIGURA DO TEXTO: controles com a otimizacao isolada (configuracao do
    # Algorithm 1), como nos dados reais. Se o ewmavar_h ainda nao existir, o
    # EWMA_VAR cai para a versao otimizada e o script avisa.
    al_is = carregar_controle_isolado(a.entrada, "ewma_h")
    var_is = carregar_controle_isolado(a.entrada, "ewmavar_h")
    if var_is is None:
        print("[AVISO] fig_f1_absoluto com EWMA_VAR OTIMIZADO: rode antes "
              "controle_ewma_h_sintetico.py --controle var --so-resumo")
    figura_f1_absoluto(f1, a.saida, al_is, var_is)
    resumos = carregar_resumos_isolados(a.entrada)
    # verificacao de robustez (suplementar): controles com otimizacao propria
    al_ot, var_ot = controles_otimizados(f1)
    figura_f1_absoluto(f1, a.saida, al_ot, var_ot, "fig_f1_absoluto_otimizado",
                       "Controls with their own optimization")
    figura_ganhos(f1, a.saida)
    fr_todas = carregar_fronteira(a.entrada, a.saida)
    arq_ent = a.entrada / "alarmes_entrada_transicao.csv"
    entrada_tr = pd.read_csv(arq_ent) if arq_ent.exists() else None
    figura_mecanismo(f1, alarmes, fr_todas, entrada_tr, a.saida)
    tabelas_fr = fronteira(f1, fr_todas, a.saida, resumos)
    figura_fronteira(fr_todas, tabelas_fr, a.saida)
    tabela_fronteira_resumo(tabelas_fr, a.saida)
    curvas = carregar_curvas(f1, a.entrada, a.saida)
    figura_curvas(curvas, a.saida, PAINEIS_CURVA, "fig_curva_fronteira")              # texto principal
    figura_curvas(curvas, a.saida, DATASETS_CURVA, "fig_curva_fronteira_todos")       # suplementar
    figura_controles(resumos, a.saida)
    print(f"Saidas em {a.saida.resolve()}")


if __name__ == "__main__":
    main()