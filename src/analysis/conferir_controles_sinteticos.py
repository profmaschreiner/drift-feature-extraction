"""Confere os controles com otimizacao isolada dos sinteticos antes de gerar as figuras.

Uso (na mesma pasta do gerar_resultados_artigo.py):
    python3 conferir_controles_sinteticos.py --entrada datasetsNew/sintetico/fatorial

Para cada controle (EWMA_alarm = ewma_h, EWMA_var = ewmavar_h) verifica:
  1. Completude: 15 datasets x 2 cenarios x 9 detectores, cada um com os 12
     arquivos de teste. O consolidador pula em silencio o que falta, por isso
     esta conferencia e necessaria.
  2. Valores ausentes (NaN) no F1 que vai para a figura (f1_hsel).
  3. Mesma origem: o F1 do Algorithm 1 recalculado pelo consolidador a partir
     das previsoes deve ser igual ao F1 da rodada principal usado na figura.
     Se bater, controle e Algorithm 1 vem dos mesmos externos e arquivos.
  4. Sanidade do EWMA_alarm: o GMA quase nao emite alarmes, entao o EWMA dos
     alarmes deve dar o mesmo F1 que o Algorithm 1 com ele.
Termina com OK ou com a lista do que corrigir.
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import gerar_resultados_artigo as g

pa = argparse.ArgumentParser()
pa.add_argument("--entrada", type=Path, default=Path("datasetsNew/sintetico/fatorial"))
a = pa.parse_args()

f1_principal, _ = g.carregar(a.entrada)
CEN = {"C1": ("sem_janela", "drift"), "C3": ("w130", "catch24_drift")}
problemas = []

for controle, nome in (("ewma_h", "EWMA_alarm"), ("ewmavar_h", "EWMA_var")):
    arq = a.entrada / f"{controle}_por_arquivo_T0_499.csv"
    print(f"\n=== {nome}: {arq}")
    if not arq.exists():
        problemas.append(f"{nome}: arquivo nao encontrado ({arq})")
        print("    NAO ENCONTRADO")
        continue
    print(f"    modificado em {dt.datetime.fromtimestamp(arq.stat().st_mtime):%d/%m/%Y %H:%M}")
    t = pd.read_csv(arq)

    # 1. completude
    n = t.groupby(["dataset", "cenario", "detector"]).size()
    faltam = [(ds, c, d) for ds in g.DATASETS for c in CEN for d in g.DETECTORES if (ds, c, d) not in n.index]
    incompletos = n[n != 12]
    print(f"    combinacoes: {len(n)} de {len(g.DATASETS) * 2 * len(g.DETECTORES)} | "
          f"faltando: {len(faltam)} | com numero de arquivos diferente de 12: {len(incompletos)}")
    for x in faltam[:15]:
        print("      falta", x)
    for k, v in incompletos.items():
        print(f"      {k}: {v} arquivos")
    if faltam or len(incompletos):
        problemas.append(f"{nome}: {len(faltam)} combinacoes faltando, {len(incompletos)} incompletas")

    # 2. NaN no valor plotado
    nan = int(t["f1_hsel"].isna().sum())
    print(f"    linhas com f1_hsel ausente: {nan}")
    if nan:
        problemas.append(f"{nome}: {nan} linhas sem f1_hsel")

    # 3. mesma origem do Algorithm 1
    m = t.groupby(["dataset", "cenario", "detector"])[["f1_alg1", "f1_hsel"]].mean().reset_index()
    m["f1_principal"] = [g.valor(f1_principal, r.dataset, *CEN[r.cenario], r.detector) for r in m.itertuples()]
    dif = (100 * (m["f1_alg1"] - m["f1_principal"])).abs()
    print(f"    Algorithm 1 do controle vs rodada principal: maior diferenca {dif.max():.4f} p.p.")
    ruins = m[dif > 0.01]
    for r in ruins.head(15).itertuples():
        print(f"      {r.dataset} {r.cenario} {r.detector}: controle {100 * r.f1_alg1:.2f} | principal {100 * r.f1_principal:.2f}")
    if len(ruins):
        problemas.append(f"{nome}: Algorithm 1 diferente da rodada principal em {len(ruins)} combinacoes")

    # 4. sanidade (so EWMA_alarm): GMA
    if controle == "ewma_h":
        gma = m[m.detector == "GeometricMovingAverage"]
        d = (100 * (gma["f1_alg1"] - gma["f1_hsel"])).abs()
        print(f"    GMA, Algorithm 1 vs EWMA_alarm: maior diferenca {d.max():.4f} p.p. (esperado ~0)")
        if d.max() > 0.5:
            problemas.append("EWMA_alarm: GMA difere do Algorithm 1, conferir a montagem do controle")

    # valores que vao para a figura (media dos 12 arquivos, em %)
    m["hsel_%"] = (100 * m["f1_hsel"]).round(2)
    saida = a.entrada / f"conferencia_{controle}.csv"
    m.to_csv(saida, index=False)
    print(f"    valores da figura salvos em {saida}")

print("\n" + ("OK: os controles estao completos e vem da mesma rodada do Algorithm 1."
              if not problemas else "CORRIGIR ANTES DE GERAR AS FIGURAS:\n  - " + "\n  - ".join(problemas)))