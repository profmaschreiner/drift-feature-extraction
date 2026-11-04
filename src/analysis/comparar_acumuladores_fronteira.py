"""
comparar_acumuladores_fronteira.py
==================================
Algorithm 1 x EWMA dos alarmes (EWMA_ALARM) nas metricas de fronteira e, em C,
so nas transicoes T_AB e T_BA. Complementa a comparacao pelo macro F1 geral
(comparar_controles.py), que em C mistura as quatro classes.

Entrada: fronteira_todas_configuracoes.csv, gerado por gerar_resultados_artigo.py
(colunas dataset, cenario, bloco, detector, f1_fronteira, U_alpha, fr, nao_rec,
troca). Em C, f1_fronteira e o macro F1 das transicoes; nas demais condicoes,
o macro F1 das duas classes.

Para cada dataset, detector e cenario (C1 sem janela; C3 com janela de 130):
  dif_f1     = F1 Alg. 1 - F1 EWMA_ALARM (p.p.; positivo favorece o Alg. 1)
  dif_Ua     = U_alpha Alg. 1 - U_alpha EWMA (amostras; negativo favorece o Alg. 1)
  dif_fr     = fr Alg. 1 - fr EWMA (negativo favorece o Alg. 1)
  dif_troca  = troca Alg. 1 - troca EWMA (so C; p.p.; negativo favorece o Alg. 1)
  responde   = ganho do Alg. 1 sobre a referencia (C0 ou C2 w130) >= 5 p.p. no
               macro F1 geral (mesmo criterio da figura de ganhos), isto e,
               situacoes em que os alarmes carregam informacao

Saidas (pasta --saida):
  acumuladores_fronteira.csv           uma linha por dataset x detector x cenario
  acumuladores_fronteira_resumo.csv    por dataset e cenario: mediana das
                                       diferencas e quantos detectores ficam em
                                       +-3 p.p., entre todos e entre os que respondem

Uso (no cluster, na pasta do fatorial ou onde estiver o CSV):
    python3 comparar_acumuladores_fronteira.py --fronteira saida_artigo/fronteira_todas_configuracoes.csv \
        --f1 datasetsNew/sintetico/fatorial/resumo_aninhado_por_arquivo_T0_499.csv --saida saida_artigo
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

DATASETS = ["V_r1.25", "V_r1.5", "V_r2.0", "F_L1_280", "F_L1_210", "F_L1_105",
            "M_a1_0.75", "M_a1_1.0", "M_a1_1.25", "H_g230", "H_g190", "H_g150",
            "C_T100", "C_T300", "C_T700"]
DETECTORES = ["ADWIN", "CUSUM", "EWMAChart", "GeometricMovingAverage", "HDDMAverage",
              "HDDMWeighted", "KSWIN", "PageHinkley", "SEED"]
CENARIOS = {"C1": ("drift", "sem_janela", "baseline", "sem_janela"),
            "C3": ("catch24_drift", "w130", "catch24", "w130")}
LIMIAR_GANHO = 0.05
FAIXA_EMPATE = 0.03


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--fronteira", type=Path, default=Path("saida_artigo/fronteira_todas_configuracoes.csv"))
    pa.add_argument("--f1", type=Path,
                    default=Path("datasetsNew/sintetico/fatorial/resumo_aninhado_por_arquivo_T0_499.csv"))
    pa.add_argument("--saida", type=Path, default=Path("saida_artigo"))
    a = pa.parse_args()

    fr = pd.read_csv(a.fronteira)
    fr["detector"] = fr["detector"].fillna("")
    arq = pd.read_csv(a.f1)
    arq = arq[arq["modelo"] == "RF"].copy()
    arq["detector"] = arq["detector"].fillna("")
    f1g = arq.groupby(["dataset", "bloco", "cenario", "detector"])["f1_pareada"].mean()

    def linha(ds, cen, bloco, det):
        m = fr[(fr.dataset == ds) & (fr.cenario == cen) & (fr.bloco == bloco) & (fr.detector == det)]
        return m.iloc[0] if len(m) else None

    linhas = []
    for ds in DATASETS:
        for rot, (cen, bloco, cen_ref, bloco_ref) in CENARIOS.items():
            ref_geral = f1g.get((ds, bloco_ref, cen_ref, ""), np.nan)
            for det in DETECTORES:
                alg = linha(ds, cen, bloco, det)
                ewm = linha(ds, cen, bloco, f"EWMA_ALARM_{det}")
                if alg is None or ewm is None:
                    continue
                ganho = f1g.get((ds, bloco, cen, det), np.nan) - ref_geral
                linhas.append(dict(
                    dataset=ds, cenario=rot, detector=det, responde=bool(ganho >= LIMIAR_GANHO),
                    f1_alg1=alg.f1_fronteira, f1_ewma=ewm.f1_fronteira,
                    dif_f1=alg.f1_fronteira - ewm.f1_fronteira,
                    Ua_alg1=alg.U_alpha, Ua_ewma=ewm.U_alpha, dif_Ua=alg.U_alpha - ewm.U_alpha,
                    fr_alg1=alg.fr, fr_ewma=ewm.fr, dif_fr=alg.fr - ewm.fr,
                    troca_alg1=alg.troca, troca_ewma=ewm.troca, dif_troca=alg.troca - ewm.troca,
                    nao_rec_alg1=alg.nao_rec, nao_rec_ewma=ewm.nao_rec))
    t = pd.DataFrame(linhas)
    if t.empty:
        raise SystemExit("[ERRO] nenhum par Alg. 1 / EWMA_ALARM encontrado no CSV de fronteira "
                         "(confira se as previsoes dos controles entraram em fronteira_todas_configuracoes.csv)")
    t.to_csv(a.saida / "acumuladores_fronteira.csv", index=False)

    resumo = []
    for (ds, rot), g in t.groupby(["dataset", "cenario"], sort=False):
        for grupo, s in [("todos", g), ("respondem", g[g.responde])]:
            if s.empty:
                continue
            resumo.append(dict(
                dataset=ds, cenario=rot, grupo=grupo, n=len(s),
                med_dif_f1_pp=100 * s.dif_f1.median(),
                em_3pp=int((s.dif_f1.abs() <= FAIXA_EMPATE).sum()),
                alg1_melhor_5pp=int((s.dif_f1 > 0.05).sum()), ewma_melhor_5pp=int((s.dif_f1 < -0.05).sum()),
                med_dif_Ua=s.dif_Ua.median(), med_dif_fr=s.dif_fr.median(),
                med_dif_troca_pp=100 * s.dif_troca.median() if ds.startswith("C_") else np.nan))
    r = pd.DataFrame(resumo)
    r.to_csv(a.saida / "acumuladores_fronteira_resumo.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.2f}".format):
        print("F1 em C: so transicoes (T_AB, T_BA). Diferencas = Alg. 1 - EWMA_ALARM.")
        print("dif_f1 e dif_troca em p.p.; positivo em F1 e negativo em U_alpha, fr e troca favorecem o Alg. 1.\n")
        print(r.to_string(index=False))
        c = t[t.dataset.str.startswith("C_")].copy()
        if not c.empty:
            for col in ["f1_alg1", "f1_ewma", "dif_f1", "troca_alg1", "troca_ewma", "dif_troca"]:
                c[col] = 100 * c[col]
            print("\n=== C, por detector (F1 das transicoes e troca em %) ===")
            print(c[["dataset", "cenario", "detector", "responde", "f1_alg1", "f1_ewma", "dif_f1",
                     "Ua_alg1", "Ua_ewma", "fr_alg1", "fr_ewma", "troca_alg1", "troca_ewma"]].to_string(index=False))
        tot = t[t.responde]
        print(f"\nSituacoes que respondem: {len(tot)}; |dif F1| <= 3 p.p. em {int((tot.dif_f1.abs() <= 0.03).sum())}; "
              f"Alg. 1 melhor em > 5 p.p. em {int((tot.dif_f1 > 0.05).sum())}; EWMA melhor em > 5 p.p. em "
              f"{int((tot.dif_f1 < -0.05).sum())}")
    print(f"\nSalvo: {a.saida / 'acumuladores_fronteira.csv'} e {a.saida / 'acumuladores_fronteira_resumo.csv'}")


if __name__ == "__main__":
    main()
