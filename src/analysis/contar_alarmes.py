"""
contar_alarmes.py
=================
Taxa de alarmes por classe e por detector nos sinteticos fatoriais, para
testar a cadeia da Proposicao: propriedade do sinal na classe -> ARL0 do
detector naquele sinal -> taxa de alarmes -> drift score -> classificacao.

Usa os mesmos detectores e a mesma funcao de atualizacao do experimento
(build_default_detector_list e update_detector_and_get_flag, importados do
arquivo do experimento), um detector novo por arquivo, desde t = 0, como no
pipeline. A normalizacao reproduz exatamente o teste externo da rodada
aninhada: para cada externo k, o scaler_detectors escolhido pelo Optuna
naquele externo (cenario drift, mesmo detector) e ajustado nos 8 arquivos de
otimizacao e aplicado aos 4 arquivos de teste. Assim, os alarmes contados
sao os mesmos que geraram as previsoes de teste. Externo sem resultado e
pulado (e indicado na coluna externos_lidos).

Cada alarme e classificado como:
  verdadeiro: ate TOLERANCIA amostras depois de um evento real (salto de
              media em F, ou a troca de classe);
  falso:      qualquer outro.
Por classe, a taxa de falsos alarmes estima 1/ARL0 do detector naquele
sinal; a taxa total e o que o drift score efetivamente acumula.

Saida (na pasta raiz do fatorial): alarmes_por_classe.csv, uma linha por
dataset, detector e classe, com o F1 de teste do cenario drift (C1) ao lado.

No dataset C, tambem alarmes_entrada_transicao.csv: para cada detector e cada
tipo de entrada (A -> T_AB e B -> T_BA), a fracao dos segmentos de transicao
com pelo menos um alarme nas primeiras --janela-entrada amostras (padrao 100),
os alarmes por segmento nesse trecho e a taxa no restante da transicao. Mede o
alarme gerado pela propria fronteira: a entrada B -> T_BA muda a media (de +-1
para 0) e a entrada A -> T_AB nao (a media de A, que alterna +-1, ja e 0).

Uso (em ~/am_reais):
    python3 contar_alarmes.py --condicoes V F M H C
    python3 contar_alarmes.py --condicoes V --detectores HDDMAverage PageHinkley
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

CAMINHO_EXPERIMENTO = Path("experimento_catch24_precompute_janelas_corrigido.py")
RAIZ = Path("datasetsNew/sintetico/fatorial")
PASTA_RESULTADOS = "resultados_aninhado"
T0 = 499
TOLERANCIA = 50          # mesma janela de aceitacao do gerador
EXTERNOS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]   # arquivos de teste de cada externo
TRANSICOES = (2, 3)                                   # T_AB e T_BA no dataset C
NOME_ENTRADA = {2: "A->T_AB", 3: "B->T_BA"}
DETECTORES = ["ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
              "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED"]


def carregar_experimento():
    # optuna e pycatch22 so sao importados no topo do experimento; a contagem de
    # alarmes nao os usa. Se nao estiverem instalados, entram modulos vazios.
    import types
    for nome in ("optuna", "pycatch22"):
        try:
            __import__(nome)
        except ImportError:
            sys.modules[nome] = types.ModuleType(nome)
    spec = importlib.util.spec_from_file_location("experimento_real", CAMINHO_EXPERIMENTO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experimento_real"] = mod
    spec.loader.exec_module(mod)
    mod.FEATURES_ORIGINAIS = ["x1"]
    return mod


def plano_externos(pasta_ds: Path, detector: str):
    """[(externo, scaler escolhido nesse externo)], F1 pareado medio de teste."""
    base = pasta_ds / PASTA_RESULTADOS / f"T0_{T0}"
    plano, f1 = [], []
    for k in range(len(EXTERNOS)):
        arq = base / f"ext{k}" / "sem_janela" / "teste" / "drift" / detector / "resultado_teste.json"
        if arq.exists():
            r = json.loads(arq.read_text())
            plano.append((k, r["best_params"]["scaler_detectors"]))
            f1.append(r.get("RF_f1_pareada"))
    return plano, (float(np.mean(f1)) if f1 else np.nan)


def eventos_reais(meta: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(meta["is_drift_point"].to_numpy() == 1)


def contar(mod, pasta_ds: Path, detector: str, plano, tol: int, janela_entrada: int = 100):
    dfs = [pd.read_csv(pasta_ds / f"base_{i}.csv") for i in range(12)]
    acum = {}
    entrada = {}          # so em C: alarmes no inicio de cada segmento de transicao
    ev_total = ev_det = 0
    atrasos = []
    tarefas = []
    for k, scaler_nome in plano:
        teste = EXTERNOS[k]
        otim = [i for i in range(12) if i not in teste]
        sc = mod.build_scaler_from_name(scaler_nome)
        sc.fit(pd.concat([dfs[i] for i in otim], ignore_index=True)[["x1"]])  # como no teste externo
        tarefas += [(i, sc) for i in teste]
    for i, sc in tarefas:
        df = dfs[i]
        meta = pd.read_csv(pasta_ds / f"base_drift_{i}.csv")
        x = sc.transform(df[["x1"]])[:, 0]
        y = df["y"].to_numpy()
        det = mod.build_default_detector_list(detector, 1)[0]
        alarmes = np.array([t for t, v in enumerate(x) if mod.update_detector_and_get_flag(det, v)], int)
        ev = eventos_reais(meta)
        verdadeiro = np.zeros(len(alarmes), bool)
        for e in ev:
            dentro = (alarmes >= e) & (alarmes < e + tol)
            verdadeiro |= dentro
            ev_total += 1
            if dentro.any():
                ev_det += 1
                atrasos.append(int(alarmes[dentro][0] - e))
        for c in np.unique(y):
            m = y == c
            a = acum.setdefault(int(c), dict(n=0, total=0, falsos=0))
            a["n"] += int(m.sum())
            a["total"] += int(m[alarmes].sum()) if len(alarmes) else 0
            a["falsos"] += int(m[alarmes[~verdadeiro]].sum()) if len(alarmes) else 0
        if pasta_ds.name.startswith("C_"):
            ini = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
            fim = np.r_[ini[1:], len(y)]
            for k, (i0, i1) in enumerate(zip(ini, fim)):
                c = int(y[i0])
                if k == 0 or c not in TRANSICOES:      # so entradas reais em T_AB e T_BA
                    continue
                j = min(i0 + janela_entrada, i1)
                n_ent = int(((alarmes >= i0) & (alarmes < j)).sum())
                n_resto = int(((alarmes >= j) & (alarmes < i1)).sum())
                e = entrada.setdefault(c, dict(seg=0, com_alarme=0, alarmes=0, amostras=0,
                                               alarmes_resto=0, amostras_resto=0))
                e["seg"] += 1
                e["com_alarme"] += int(n_ent > 0)
                e["alarmes"] += n_ent
                e["amostras"] += j - i0
                e["alarmes_resto"] += n_resto
                e["amostras_resto"] += i1 - j
    deteccao = dict(eventos=ev_total, detectados=ev_det,
                    recall=ev_det / ev_total if ev_total else np.nan,
                    atraso_medio=float(np.mean(atrasos)) if atrasos else np.nan)
    return acum, deteccao, entrada


def main():
    global T0
    pa = argparse.ArgumentParser()
    pa.add_argument("--raiz", type=Path, default=RAIZ)
    pa.add_argument("--condicoes", nargs="+", default=["V", "F"])
    pa.add_argument("--detectores", nargs="+", default=DETECTORES)
    pa.add_argument("--tolerancia", type=int, default=TOLERANCIA)
    pa.add_argument("--t0", type=int, default=T0, help="T0 da rodada aninhada (pasta T0_<t0>)")
    pa.add_argument("--janela-entrada", type=int, default=100,
                    help="C: amostras do inicio de cada transicao em que os alarmes sao contados")
    a = pa.parse_args()
    T0 = a.t0

    mod = carregar_experimento()
    datasets = sorted(d.name for d in a.raiz.iterdir()
                      if d.is_dir() and d.name.split("_")[0] in a.condicoes and (d / "base_0.csv").exists())
    linhas, linhas_ent = [], []
    for ds in datasets:
        for det in a.detectores:
            plano, f1 = plano_externos(a.raiz / ds, det)
            if not plano:
                print(f"[SKIP] {ds} {det}: sem resultados de teste do cenario drift", flush=True)
                continue
            acum, dt, ent = contar(mod, a.raiz / ds, det, plano, a.tolerancia, a.janela_entrada)
            scalers = ",".join(sc for _, sc in plano)
            n_ext = len(plano)
            for c, v in sorted(acum.items()):
                linhas.append({
                    "dataset": ds, "detector": det, "scaler": scalers, "classe": c,
                    "amostras": v["n"], "alarmes_total": v["total"], "alarmes_falsos": v["falsos"],
                    "taxa_total_por_1000": 1000 * v["total"] / v["n"],
                    "taxa_falsos_por_1000": 1000 * v["falsos"] / v["n"],
                    "ARL0_estimado": v["n"] / v["falsos"] if v["falsos"] else np.inf,
                    "eventos_reais": dt["eventos"], "recall_deteccao": dt["recall"],
                    "atraso_medio_deteccao": dt["atraso_medio"],
                    "f1_C1_teste": f1, "externos_lidos": n_ext,
                })
            for c, e in sorted(ent.items()):
                linhas_ent.append({
                    "dataset": ds, "detector": det,
                    "entrada": NOME_ENTRADA[c], "classe": c, "segmentos": e["seg"],
                    "prop_segmentos_com_alarme": e["com_alarme"] / e["seg"] if e["seg"] else np.nan,
                    "alarmes_por_segmento": e["alarmes"] / e["seg"] if e["seg"] else np.nan,
                    "taxa_entrada_por_1000": 1000 * e["alarmes"] / e["amostras"] if e["amostras"] else np.nan,
                    "taxa_resto_por_1000": (1000 * e["alarmes_resto"] / e["amostras_resto"]
                                            if e["amostras_resto"] else np.nan),
                    "janela_entrada": a.janela_entrada,
                })
            print(f"[OK] {ds} {det} (scalers por externo: {scalers})", flush=True)
    df = pd.DataFrame(linhas)
    saida = a.raiz / "alarmes_por_classe.csv"
    df.to_csv(saida, index=False)

    # resumo: taxas total e de falsos alarmes por classe, ao lado do F1 do C1
    piv = df.pivot_table(index=["dataset", "detector"], columns="classe",
                         values=["taxa_total_por_1000", "taxa_falsos_por_1000"], aggfunc="first")
    piv.columns = [f"{'total' if v.startswith('taxa_total') else 'falsos'}/1000 c{c}" for v, c in piv.columns]
    extra = df.groupby(["dataset", "detector"]).agg(
        recall=("recall_deteccao", "first"), atraso=("atraso_medio_deteccao", "first"),
        f1_C1=("f1_C1_teste", "first"))
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(piv.join(extra))
    print(f"\nTabela: {saida}")

    if linhas_ent:
        de = pd.DataFrame(linhas_ent)
        saida_e = a.raiz / "alarmes_entrada_transicao.csv"
        de.to_csv(saida_e, index=False)
        pe = de.pivot_table(index=["dataset", "detector"], columns="entrada",
                            values=["prop_segmentos_com_alarme", "taxa_resto_por_1000"], aggfunc="first")
        with pd.option_context("display.width", 220, "display.max_rows", 300, "display.float_format", "{:.2f}".format):
            print(f"\n=== Entrada nas transicoes (primeiras {a.janela_entrada} amostras de cada segmento) ===")
            print(pe)
        print(f"\nTabela: {saida_e}")


if __name__ == "__main__":
    main()