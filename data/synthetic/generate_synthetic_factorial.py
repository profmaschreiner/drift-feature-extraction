"""
gerar_sintetico_fatorial.py
===========================
Gera os quatro datasets sinteticos do desenho fatorial (ASOC, Secao 6.1.2).
Em cada condicao, as duas classes diferem em UM unico fator:

  V  so variancia local: media 0 nas duas classes, desvio padrao diferente,
     sem nenhum drift dentro dos segmentos.
  F  so frequencia de drift: saltos de media entre os niveis +A e -A, com a
     MESMA distribuicao marginal nas duas classes (mesmos valores, apenas
     reposicionados); muda so a taxa de troca de nivel.
  M  so magnitude do drift: mesma taxa de troca e o mesmo ruido dentro de
     cada nivel; muda so o tamanho do salto (a0 < a1). Trechos sem salto sao
     estatisticamente identicos entre classes; a variancia marginal maior da
     classe 1 e consequencia inevitavel do salto maior (ver M_COMPENSAR_RUIDO).
  H  so horizonte temporal: mesmos eventos, mesma taxa media e mesma
     distribuicao marginal; a classe 0 tem eventos regulares e a classe 1
     eventos agrupados em trios. Todo intervalo entre eventos e maior que a
     janela maxima do catch22 (130), de modo que qualquer janela local
     contem no maximo um evento nas duas classes.

  C  contexto (analogo as transicoes posturais do SmPh): 4 classes em ciclo
     A -> T_AB -> B -> T_BA -> A ... ("standing" -> "stand-to-sit" ->
     "sitting" -> "sit-to-stand"). A tem saltos de media frequentes, B tem
     media constante; as duas transicoes sao localmente identicas (media 0,
     mesma variancia, sem eventos) e a fronteira que leva a elas e um salto de
     mesma magnitude. T_AB e T_BA so se distinguem pela historia anterior.
     O nivel de contraste e a duracao da transicao (C_TRANSICAO): quanto mais
     longa, mais a transicao se afasta do alcance da janela do catch24.

Cada condicao e gerada em tres niveis de contraste (NIVEIS). Cada nivel vira
uma subpasta com N_ARQUIVOS arquivos base_<i>.csv (colunas x1, y), no formato
lido por load_dados() do experimento catch24/drift, mais base_drift_<i>.csv
com as posicoes verdadeiras dos drifts (mesmo formato do CaDrift, para a
avaliacao de deteccao e o ORACLE).

Estrutura de cada arquivo: dois segmentos de classe com N_SEGMENTO amostras
(3360 amostras por arquivo, proximo das ~3400 dos arquivos do CaDrift). A
ordem das classes alterna entre arquivos ([0,1] nos pares, [1,0] nos
impares), para que posicao no arquivo nao identifique a classe.

Pareamento contrafactual (ideia do gerador do PAKDD): dentro de cada arquivo,
os dois segmentos de classe compartilham a mesma realizacao de ruido.
  V, M: mesmo vetor de ruido padronizado, apenas reescalado.
  F, H: exatamente o mesmo multiconjunto de valores (verificado por assert).

Uso
---
    python3 gerar_sintetico_fatorial.py                 # gera em OUTPUT_ROOT
    python3 gerar_sintetico_fatorial.py --saida <pasta> # outra pasta raiz
    python3 gerar_sintetico_fatorial.py --preview       # tambem salva figura

Configuracao no experimento (substitui o bloco do sintetico):

    PASTA_ENTRADA = "datasetsNew/sintetico/fatorial"
    PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")
    ROTULO = "y"
    FEATURES_ORIGINAIS = ["x1"]
    NOME_BASE = "base_"
    END_BASE = ".csv"
    CONJ_TESTE = [[i] for i in range(12)]
    DATASET_SINTETICO = True
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACAO
# ============================================================
SEED = 42
OUTPUT_ROOT = Path("datasetsNew/sintetico/fatorial")

N_ARQUIVOS = 12          # = numero de folds (CONJ_TESTE = [[0], ..., [11]])
N_SEGMENTO = 1680        # amostras por segmento de classe (2 por arquivo)
FEATURE = "x1"
ROTULO = "y"
NOME_BASE = "base_"
END_BASE = ".csv"

JANELA_ACEITACAO = 50    # mesma tolerancia usada na avaliacao do CaDrift
JANELA_MAX_CATCH = 130   # WINDOW_MAX do experimento (restricao da condicao H)

SIGMA = 1.0              # desvio padrao do ruido
A = 1.0                  # nivel +-A dos saltos em F e H (salto = 2A)

# F: a classe 0 troca de nivel a cada F_L0 amostras; a classe 1 a cada L1.
F_L0 = 420
# M: mesma duracao de nivel nas duas classes; classe 0 com salto pequeno.
M_L = 280
M_A0 = 0.5
# False (padrao): mesmo ruido SIGMA nas duas classes. Trechos sem salto sao
#   identicos; a variancia marginal difere (SIGMA^2 + a^2) por causa do salto.
# True: ruido compensado para igualar a variancia marginal (a^2 + sigma_c^2 =
#   M_VAR_TOTAL). Iguala a marginal, mas faz a variancia DENTRO de cada nivel
#   diferir entre classes, o que torna M separavel por janelas curtas sem
#   envolver o salto (confunde M com V).
M_COMPENSAR_RUIDO = False
M_VAR_TOTAL = A ** 2 + SIGMA ** 2
# H: classe 0 com eventos a cada H_G0; classe 1 em trios (g, g, L), L = 3*G0 - 2g.
H_G0 = 280

# C (contexto): A = saltos de media a cada C_RUN_A amostras; B = media
# constante (+-A, sinal sorteado); T_AB e T_BA = media 0. Duracao das posturas
# com variacao de +-C_JITTER; duracao da transicao fixa (nivel do contraste).
C_POSTURA = 800
C_RUN_A = 100
C_JITTER = 0.2
C_CICLOS = 4             # ciclos A -> T_AB -> B -> T_BA por arquivo
C_TRANSICAO = [100, 300, 700]   # < 130 | entre 130 e 500 | > 500
C_CLASSES = {"A": 0, "B": 1, "T_AB": 2, "T_BA": 3}

# Indice fixo de cada condicao na semente (gerar so uma condicao nao muda as
# sementes das demais).
IDX_SEMENTE = {"V": 0, "F": 1, "M": 2, "H": 3, "N": 4, "C": 5}

# Niveis de contraste (do mais fraco ao mais forte).
NIVEIS = {
    "V": [1.25, 1.5, 2.0],   # razao sigma1/sigma0
    "F": [280, 210, 105],    # L1 -> razao de frequencia 1.5x, 2x, 4x
    "M": [0.75, 1.0, 1.25],  # a1 (a0 = 0.5) -> razao 1.5x, 2x, 2.5x
    "H": [230, 190, 150],    # intervalo curto g -> intervalo longo 380, 460, 540
}

# Controle nulo opcional: classes identicas (deve ficar no acaso, macro F1 ~0.5).
INCLUIR_CONTROLE_NULO = False


# ============================================================
# BLOCOS BASICOS
# ============================================================
def sinais_por_runs(runs, sinal_inicial):
    """Vetor de niveis (+1/-1) a partir das duracoes dos runs, alternando o
    sinal a cada run."""
    partes = [np.full(r, sinal_inicial * (1 if k % 2 == 0 else -1), dtype=int)
              for k, r in enumerate(runs)]
    return np.concatenate(partes)


def preencher_com_pools(sinais, pool_pos, pool_neg):
    """Preenche as amostras de nivel +1 com pool_pos e as de nivel -1 com
    pool_neg, na ordem. Dois vetores de sinais com a mesma contagem por nivel
    recebem exatamente o mesmo multiconjunto de valores."""
    x = np.empty(sinais.size)
    pos, neg = sinais == 1, sinais == -1
    assert pos.sum() == pool_pos.size and neg.sum() == pool_neg.size
    x[pos] = pool_pos
    x[neg] = pool_neg
    return x


def trocas(sinais):
    """Indices em que o nivel muda (inicio de cada novo run)."""
    return (np.flatnonzero(np.diff(sinais) != 0) + 1).tolist()


def checar_multiconjunto(x0, x1):
    assert np.array_equal(np.sort(x0), np.sort(x1)), "multiconjuntos diferentes"


# ============================================================
# CONDICOES (cada funcao devolve os dois segmentos de um arquivo)
# ============================================================
def segmentos_V(rng, razao):
    e = rng.normal(size=N_SEGMENTO)
    s0, s1 = SIGMA / np.sqrt(razao), SIGMA * np.sqrt(razao)  # media geometrica = SIGMA
    return (s0 * e, []), (s1 * e, [])


def segmentos_F(rng, L1):
    assert N_SEGMENTO % (2 * F_L0) == 0 and N_SEGMENTO % (2 * L1) == 0
    metade = N_SEGMENTO // 2
    pool_pos = A + SIGMA * rng.normal(size=metade)
    pool_neg = -A + SIGMA * rng.normal(size=metade)
    saida = []
    for L in (F_L0, L1):
        sinais = sinais_por_runs([L] * (N_SEGMENTO // L), rng.choice([1, -1]))
        sinais = np.roll(sinais, rng.integers(N_SEGMENTO))  # preserva a contagem por nivel
        saida.append((preencher_com_pools(sinais, pool_pos, pool_neg), trocas(sinais)))
    checar_multiconjunto(saida[0][0], saida[1][0])
    return saida[0], saida[1]


def segmentos_M(rng, a1):
    assert N_SEGMENTO % (2 * M_L) == 0 and a1 ** 2 < M_VAR_TOTAL
    sinais = sinais_por_runs([M_L] * (N_SEGMENTO // M_L), rng.choice([1, -1]))
    sinais = np.roll(sinais, rng.integers(N_SEGMENTO))
    e = rng.normal(size=N_SEGMENTO)
    saida = []
    for a in (M_A0, a1):
        sigma_c = np.sqrt(M_VAR_TOTAL - a ** 2) if M_COMPENSAR_RUIDO else SIGMA
        saida.append((a * sinais + sigma_c * e, trocas(sinais)))
    return saida[0], saida[1]


def segmentos_H(rng, g):
    L = 3 * H_G0 - 2 * g
    assert N_SEGMENTO == 6 * H_G0 and min(g, L) > JANELA_MAX_CATCH
    metade = N_SEGMENTO // 2
    pool_pos = A + SIGMA * rng.normal(size=metade)
    pool_neg = -A + SIGMA * rng.normal(size=metade)
    runs0 = [H_G0] * 6
    runs1 = list(np.roll([g, g, L, g, g, L], rng.integers(6)))  # rotacao por runs inteiros
    saida = []
    for runs in (runs0, runs1):
        # sem roll por amostra: os runs das bordas ficam inteiros, e o
        # intervalo minimo entre eventos vale tambem junto a troca de classe
        sinais = sinais_por_runs(runs, rng.choice([1, -1]))
        saida.append((preencher_com_pools(sinais, pool_pos, pool_neg), trocas(sinais)))
    checar_multiconjunto(saida[0][0], saida[1][0])
    return saida[0], saida[1]


def segmentos_nulo(rng, _):
    e0, e1 = rng.normal(size=N_SEGMENTO), rng.normal(size=N_SEGMENTO)
    return (SIGMA * e0, []), (SIGMA * e1, [])


CONDICOES = {
    "V": (segmentos_V, "razao_sigma"),
    "F": (segmentos_F, "L1"),
    "M": (segmentos_M, "a1"),
    "H": (segmentos_H, "g"),
}


# ============================================================
# CONDICAO C (contexto): ciclo A -> T_AB -> B -> T_BA
# ============================================================
def segmento_A(rng, n):
    """Media alternando +-A a cada C_RUN_A amostras (fase inicial sorteada)."""
    deslocamento = int(rng.integers(C_RUN_A))
    runs = [C_RUN_A - deslocamento] if deslocamento else []
    while sum(runs) < n:
        runs.append(C_RUN_A)
    sinais = sinais_por_runs(runs, rng.choice([1, -1]))[:n]
    return A * sinais + SIGMA * rng.normal(size=n), trocas(sinais)


def segmento_B(rng, n):
    return rng.choice([A, -A]) + SIGMA * rng.normal(size=n), []


def segmento_T(rng, n):
    return SIGMA * rng.normal(size=n), []


def gerar_arquivo_C(rng, duracao_transicao, indice):
    ciclo = ["A", "T_AB", "B", "T_BA"]
    inicio = 0 if indice % 2 == 0 else 2      # metade dos arquivos comeca em B
    sequencia = [ciclo[(inicio + k) % 4] for k in range(4 * C_CICLOS)]
    xs, ys, pontos, t = [], [], [], 0
    for k, nome in enumerate(sequencia):
        if nome.startswith("T"):
            n = duracao_transicao
            x, ev = segmento_T(rng, n)
        else:
            n = int(round(C_POSTURA * rng.uniform(1 - C_JITTER, 1 + C_JITTER)))
            x, ev = (segmento_A if nome == "A" else segmento_B)(rng, n)
        if k > 0:
            pontos.append((t, f"phase_change_{nome}"))
        pontos.extend((t + e, "abrupto") for e in ev)
        xs.append(x)
        ys.append(np.full(n, C_CLASSES[nome], dtype=int))
        t += n
    df = pd.DataFrame({FEATURE: np.concatenate(xs), ROTULO: np.concatenate(ys)})
    return df, metadados_drift(len(df), sorted(set(pontos)))


def gerar_condicao_C(raiz, duracao_transicao, semente):
    rng = np.random.default_rng(semente)
    pasta = raiz / f"C_T{duracao_transicao}"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for i in range(N_ARQUIVOS):
        df, meta = gerar_arquivo_C(rng, duracao_transicao, i)
        df.to_csv(pasta / f"{NOME_BASE}{i}{END_BASE}", index=False)
        meta.to_csv(pasta / f"base_drift_{i}.csv", index=False)
        arquivos.append((df, meta))
    parametros = {
        "condicao": "C", "duracao_transicao": duracao_transicao, "semente": semente,
        "classes": C_CLASSES, "C_POSTURA": C_POSTURA, "C_RUN_A": C_RUN_A,
        "C_JITTER": C_JITTER, "C_CICLOS": C_CICLOS, "N_ARQUIVOS": N_ARQUIVOS,
        "SIGMA": SIGMA, "A": A, "JANELA_ACEITACAO": JANELA_ACEITACAO,
    }
    (pasta / "parametros.json").write_text(json.dumps(parametros, indent=2))
    return pasta, arquivos


def validar_C(duracao_transicao, arquivos):
    """Por classe: media, variancia, duracao media dos segmentos, eventos
    dentro do segmento e eventos nas ultimas JANELA_MAX_CATCH amostras antes
    do inicio do segmento (o que uma janela curta ainda alcanca)."""
    nomes = {v: k for k, v in C_CLASSES.items()}
    linhas = []
    for c in sorted(nomes):
        valores, dur, ev_dentro, ev_antes = [], [], [], []
        for df, meta in arquivos:
            y = df[ROTULO].to_numpy()
            ev = np.flatnonzero(meta["is_drift_point"].to_numpy())
            inicios = np.flatnonzero(np.diff(np.r_[-1, y]) != 0)
            fins = np.r_[inicios[1:], len(y)]
            for a, b in zip(inicios, fins):
                if y[a] != c:
                    continue
                valores.append(df[FEATURE].to_numpy()[a:b])
                dur.append(b - a)
                ev_dentro.append(int(np.sum((ev > a) & (ev < b))))
                ev_antes.append(int(np.sum((ev >= a - JANELA_MAX_CATCH) & (ev < a))))
        v = np.concatenate(valores)
        linhas.append({
            "dataset": f"C_T{duracao_transicao}", "classe": f"{c} ({nomes[c]})",
            "media": round(float(v.mean()), 3), "variancia": round(float(v.var()), 3),
            "n_segmentos": len(dur), "duracao_media": round(float(np.mean(dur)), 1),
            "eventos_dentro": round(float(np.mean(ev_dentro)), 2),
            "eventos_130_antes": round(float(np.mean(ev_antes)), 2),
        })
    return linhas


def salvar_preview_C(raiz, pastas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cores = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:red"}
    nomes = {v: k for k, v in C_CLASSES.items()}
    fig, eixos = plt.subplots(len(pastas), 1, figsize=(11, 2.0 * len(pastas)))
    for ax, pasta in zip(np.atleast_1d(eixos), pastas):
        df = pd.read_csv(pasta / f"{NOME_BASE}0{END_BASE}")
        y = df[ROTULO].to_numpy()
        ax.plot(df[FEATURE].to_numpy(), lw=0.3, color="0.25")
        inicios = np.flatnonzero(np.diff(np.r_[-1, y]) != 0)
        for a, b in zip(inicios, np.r_[inicios[1:], len(y)]):
            ax.axvspan(a, b, color=cores[y[a]], alpha=0.12, lw=0)
        ax.set_ylabel(pasta.name, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cores[c], alpha=0.3) for c in cores]
    np.atleast_1d(eixos)[0].legend(handles, [nomes[c] for c in cores], ncol=4,
                                   fontsize=8, loc="upper right")
    np.atleast_1d(eixos)[-1].set_xlabel("data sequence (arquivo 0)")
    fig.tight_layout()
    fig.savefig(raiz / "preview_contexto.png", dpi=150)
    plt.close(fig)


# ============================================================
# MONTAGEM DOS ARQUIVOS
# ============================================================
def metadados_drift(n, pontos):
    """Mesmo formato do base_drift_<i>.csv do CaDrift."""
    is_drift = np.zeros(n, dtype=int)
    janela = np.zeros(n, dtype=int)
    tipo = np.array(["none"] * n, dtype=object)
    for idx, t in pontos:
        is_drift[idx] = 1
        janela[idx:idx + JANELA_ACEITACAO] = 1
        tipo[idx] = t
    return pd.DataFrame({"is_drift_point": is_drift,
                         "acceptance_window": janela,
                         "drift_type": tipo})


def gerar_arquivo(rng, fn_segmentos, nivel, indice):
    seg = fn_segmentos(rng, nivel)               # (classe 0, classe 1)
    ordem = [0, 1] if indice % 2 == 0 else [1, 0]
    xs, ys, pontos = [], [], []
    for k, classe in enumerate(ordem):
        x, trocas_seg = seg[classe]
        inicio = k * N_SEGMENTO
        if k > 0:
            pontos.append((inicio, f"phase_change_{classe}"))
        pontos.extend((inicio + t, "abrupto") for t in trocas_seg)
        xs.append(x)
        ys.append(np.full(N_SEGMENTO, classe, dtype=int))
    df = pd.DataFrame({FEATURE: np.concatenate(xs), ROTULO: np.concatenate(ys)})
    return df, metadados_drift(len(df), sorted(set(pontos)))


def nome_subpasta(cond, nivel):
    rotulo = {"V": "r", "F": "L1_", "M": "a1_", "H": "g"}.get(cond, "")
    return f"{cond}_{rotulo}{nivel}"


def gerar_condicao(raiz, cond, nivel, fn, nome_nivel, semente):
    rng = np.random.default_rng(semente)
    pasta = raiz / nome_subpasta(cond, nivel)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for i in range(N_ARQUIVOS):
        df, meta = gerar_arquivo(rng, fn, nivel, i)
        df.to_csv(pasta / f"{NOME_BASE}{i}{END_BASE}", index=False)
        meta.to_csv(pasta / f"base_drift_{i}.csv", index=False)
        arquivos.append((df, meta))
    parametros = {
        "condicao": cond, nome_nivel: nivel, "semente": semente,
        "N_SEGMENTO": N_SEGMENTO, "N_ARQUIVOS": N_ARQUIVOS, "SIGMA": SIGMA, "A": A,
        "F_L0": F_L0, "M_L": M_L, "M_A0": M_A0,
        "M_COMPENSAR_RUIDO": M_COMPENSAR_RUIDO, "M_VAR_TOTAL": M_VAR_TOTAL,
        "H_G0": H_G0, "JANELA_ACEITACAO": JANELA_ACEITACAO,
    }
    (pasta / "parametros.json").write_text(json.dumps(parametros, indent=2))
    return pasta, arquivos


# ============================================================
# VALIDACAO
# ============================================================
def validar(cond, nivel, arquivos):
    """Resumo por classe, agregado nos arquivos: media, variancia marginal,
    eventos por segmento, menor intervalo entre eventos e maximo de eventos
    numa janela de JANELA_MAX_CATCH amostras (inclui a troca de classe
    quando ela gera salto)."""
    linhas = []
    for classe in (0, 1):
        valores, n_ev, gaps, max_jan = [], [], [], 0
        for df, meta in arquivos:
            m = df[ROTULO].to_numpy() == classe
            valores.append(df.loc[m, FEATURE].to_numpy())
            ev = np.flatnonzero(meta["is_drift_point"].to_numpy()
                                & (meta["drift_type"].to_numpy() == "abrupto")
                                & m)
            n_ev.append(ev.size)
            if ev.size > 1:
                gaps.extend(np.diff(ev).tolist())
            if ev.size:
                cont = [np.sum((ev >= t) & (ev < t + JANELA_MAX_CATCH)) for t in ev]
                max_jan = max(max_jan, int(max(cont)))
        v = np.concatenate(valores)
        linhas.append({
            "dataset": nome_subpasta(cond, nivel), "classe": classe,
            "media": round(float(v.mean()), 3), "variancia": round(float(v.var()), 3),
            "eventos_por_segmento": round(float(np.mean(n_ev)), 2),
            "intervalo_minimo": int(min(gaps)) if gaps else None,
            "intervalo_medio": round(float(np.mean(gaps)), 1) if gaps else None,
            "max_eventos_janela_130": max_jan,
        })
    if cond == "H":
        assert all(l["intervalo_minimo"] > JANELA_MAX_CATCH for l in linhas)
        assert all(l["max_eventos_janela_130"] <= 1 for l in linhas)
    return linhas


def salvar_preview(raiz, pastas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(len(pastas), 1, figsize=(10, 1.8 * len(pastas)), sharex=True)
    for ax, pasta in zip(np.atleast_1d(eixos), pastas):
        df = pd.read_csv(pasta / f"{NOME_BASE}0{END_BASE}")
        meta = pd.read_csv(pasta / "base_drift_0.csv")
        ax.plot(df[FEATURE].to_numpy(), lw=0.4, color="0.2")
        for t in np.flatnonzero(meta["is_drift_point"].to_numpy()):
            ax.axvline(t, color="tab:red", ls="--", lw=0.7)
        ax.axvspan(0, N_SEGMENTO, color="tab:blue", alpha=0.06)
        ax.set_ylabel(pasta.name, fontsize=8)
    np.atleast_1d(eixos)[-1].set_xlabel("data sequence (arquivo 0: classe 0, depois classe 1)")
    fig.tight_layout()
    fig.savefig(raiz / "preview_fatorial.png", dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--condicoes", nargs="+", default=["V", "F", "M", "H", "C"],
                        help="condicoes a gerar (padrao: todas)")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    condicoes = {k: v for k, v in CONDICOES.items() if k in args.condicoes}
    niveis = dict(NIVEIS)
    if INCLUIR_CONTROLE_NULO and "N" in args.condicoes:
        condicoes["N"] = (segmentos_nulo, "nenhum")
        niveis["N"] = ["nulo"]

    if "C" in args.condicoes:
        resumo_c, pastas_c = [], []
        for il, dur in enumerate(C_TRANSICAO):
            semente = SEED * 1000 + IDX_SEMENTE["C"] * 10 + il
            pasta, arquivos = gerar_condicao_C(args.saida, dur, semente)
            resumo_c.extend(validar_C(dur, arquivos))
            pastas_c.append(pasta)
            n_amostras = int(np.mean([len(df) for df, _ in arquivos]))
            print(f"[OK] {pasta}  ({N_ARQUIVOS} arquivos x ~{n_amostras} amostras, 4 classes)")
        df_c = pd.DataFrame(resumo_c)
        df_c.to_csv(args.saida / "validacao_contexto.csv", index=False)
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print("\n", df_c.to_string(index=False), "\n")
        if args.preview:
            salvar_preview_C(args.saida, pastas_c)
            print(f"Preview: {args.saida / 'preview_contexto.png'}\n")
        if not condicoes:
            return

    resumo, ultimas = [], []
    for cond, (fn, nome_nivel) in condicoes.items():
        for il, nivel in enumerate(niveis[cond]):
            semente = SEED * 1000 + IDX_SEMENTE[cond] * 10 + il   # independente e reprodutivel
            pasta, arquivos = gerar_condicao(args.saida, cond, nivel, fn, nome_nivel, semente)
            resumo.extend(validar(cond, nivel, arquivos))
            print(f"[OK] {pasta}  ({N_ARQUIVOS} arquivos x {2 * N_SEGMENTO} amostras)")
        ultimas.append(pasta)   # nivel de maior contraste de cada condicao

    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(args.saida / "validacao_fatorial.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n", df_resumo.to_string(index=False))
    if args.preview:
        salvar_preview(args.saida, ultimas)
        print(f"\nPreview: {args.saida / 'preview_fatorial.png'}")


if __name__ == "__main__":
    main()