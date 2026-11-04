"""
comparar_controles.py
=====================
Compara o drift score (Algorithm 1) com os controles da rodada aninhada,
a partir de resumo_aninhado_por_arquivo_T0_<T0>.csv (gerado pelo --so-resumo
do rodar_sintetico_aninhado.py). Nao roda nenhum modelo.

Tabela 1 (por detector): para cada dataset e bloco, F1 pareado medio dos 12
arquivos de teste com os alarmes do detector acumulados pelo Algorithm 1 e
pela EWMA de lambda fixo (EWMA_ALARM_<det>), e a diferenca Alg1 - EWMA.
Tabela 2 (EWMA_VAR): referencia (baseline ou catch24), EWMA_VAR e o melhor
detector com Algorithm 1 no mesmo bloco.
Bloco sem_janela = cenario drift (so x1 + score); w130/w300/w500 =
catch24_drift (catch24 da janela + score).

Uso (em ~/am_reais):
    python3 rodar_sintetico_aninhado.py --so-resumo --condicoes V --janelas 130 300 500
    python3 comparar_controles.py --condicoes V
"""
from pathlib import Path
import argparse

import pandas as pd

RAIZ = Path("datasetsNew/sintetico/fatorial")
DETECTORES = ["ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
              "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED"]
BLOCOS = ["sem_janela", "w130", "w300", "w500"]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--raiz", type=Path, default=RAIZ)
    pa.add_argument("--condicoes", nargs="+", default=["V", "F", "M", "H", "C"])
    pa.add_argument("--t0", type=int, default=499)
    a = pa.parse_args()

    df = pd.read_csv(a.raiz / f"resumo_aninhado_por_arquivo_T0_{a.t0}.csv")
    if "modelo" in df.columns and (df["modelo"] == "RF").any():
        df = df[df["modelo"] == "RF"]
    df = df[df["dataset"].str.split("_").str[0].isin(a.condicoes)]
    df["detector"] = df["detector"].fillna("")
    g = (df.groupby(["dataset", "bloco", "cenario", "detector"])["f1_pareada"]
         .agg(["mean", "std", "count"]))

    def f1(ds, bloco, cen, det=""):
        k = (ds, bloco, cen, det)
        return (g.loc[k, "mean"], g.loc[k, "std"], int(g.loc[k, "count"])) if k in g.index else (None, None, 0)

    linhas, linhas_var = [], []
    for ds in sorted(df["dataset"].unique()):
        for bloco in BLOCOS:
            cen, ref_cen = ("drift", "baseline") if bloco == "sem_janela" else ("catch24_drift", "catch24")
            ref = f1(ds, bloco, ref_cen)[0]
            if ref is None:
                continue
            melhor = (None, "")
            for det in DETECTORES:
                alg1, dp1, n1 = f1(ds, bloco, cen, det)
                ewma, dp2, n2 = f1(ds, bloco, cen, f"EWMA_ALARM_{det}")
                if alg1 is not None and (melhor[0] is None or alg1 > melhor[0]):
                    melhor = (alg1, det)
                if ewma is None:
                    continue
                linhas.append({"dataset": ds, "bloco": bloco, "detector": det, "ref": ref,
                               "alg1": alg1, "dp_alg1": dp1, "ewma_alarm": ewma, "dp_ewma": dp2,
                               "alg1_menos_ewma": alg1 - ewma if alg1 is not None else None,
                               "n_arquivos": min(n1, n2)})
            var, dpv, nv = f1(ds, bloco, cen, "EWMA_VAR")
            if var is not None:
                linhas_var.append({"dataset": ds, "bloco": bloco, "ref": ref, "ewma_var": var,
                                   "dp_ewma_var": dpv, "melhor_alg1": melhor[0], "detector": melhor[1],
                                   "n_arquivos": nv})

    if not linhas and not linhas_var:
        raise SystemExit("Nenhum resultado de controle encontrado; rode antes o --so-resumo dessas condicoes.")
    t1 = pd.DataFrame(linhas)
    t2 = pd.DataFrame(linhas_var)
    nome = "_".join(a.condicoes)
    t1.to_csv(a.raiz / f"controles_ewma_alarm_{nome}_T0_{a.t0}.csv", index=False)
    t2.to_csv(a.raiz / f"controles_ewma_var_{nome}_T0_{a.t0}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        if not t1.empty:
            print("=== Algorithm 1 vs EWMA dos alarmes (mesmo detector) ===")
            print(t1.set_index(["dataset", "bloco", "detector"]))
        if not t2.empty:
            print("\n=== EWMA_VAR vs referencia e melhor detector (Algorithm 1) ===")
            print(t2.set_index(["dataset", "bloco"]))
    print(f"\nTabelas em {a.raiz}: controles_ewma_alarm_{nome}_T0_{a.t0}.csv, "
          f"controles_ewma_var_{nome}_T0_{a.t0}.csv")


if __name__ == "__main__":
    main()