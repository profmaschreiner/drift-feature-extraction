"""
analisar_transicoes.py
======================
Erros de temporizacao na fronteira entre classes, segundo o esquema de
segmentos (2SET) de Ward, Lukowicz e Gellersen (2011, ACM TIST 2(1), art. 6),
a partir das previsoes de teste salvas pelo rodar_sintetico_aninhado.py.
Nao roda nenhum modelo.

Procedimento (para cada arquivo de teste e cada classe c, uma contra todas):
  1. cada amostra e TP, TN, FP ou FN para a classe c;
  2. segmentos = trechos em que verdade e previsao ficam constantes;
  3. um segmento FN entre dois TP e fragmentacao (F); um FN seguido de TP e
     underfill de inicio (U_alpha); um FP precedido de TP e overfill de fim
     (O_omega);
  4. taxas: u_alpha = U_alpha / P, fr = F / P, o_omega = O_omega / N, com P e
     N o total de amostras positivas e negativas da classe (Secao 4.1.2).
     As contagens sao somadas sobre as classes e os 12 arquivos de teste.

Amostras avaliadas: o trecho pareado (T0 <= t < n - T0), igual para todos os
cenarios e janelas. Como o primeiro e o ultimo evento de cada arquivo ficam
cortados por esse trecho, U_alpha e O_omega so sao contados em segmentos que
comecam numa troca real de classe; o inicio do trecho nao e uma troca.
Alem das taxas, as contagens sao reportadas em amostras por troca de classe
(Secao 6.2.1 recomenda reportar tambem o numero de amostras).

Inicio do arquivo (borda): sobre todas as previsoes (nao so o trecho
pareado), taxa de acerto e cobertura nas faixas 0-130, 130-300 e 300-500
amostras desde o inicio de cada arquivo. Cobertura = fracao das amostras da
faixa que tem previsao: o catch24 com janela W so preve a partir de t = W-1,
enquanto baseline e drift preveem desde t = 0.

Saidas na pasta raiz do fatorial, uma linha por dataset, cenario, bloco
(janela) e detector:
  tabela_2set_T0_<T0>.csv   U_alpha, O_omega e fr
  tabela_borda_T0_<T0>.csv  acerto e cobertura no inicio do arquivo

Uso (em ~/am_reais):
    python3 analisar_transicoes.py --condicoes V
    python3 analisar_transicoes.py --condicoes V F M H C
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

RAIZ = Path("datasetsNew/sintetico/fatorial")
PASTA_RESULTADOS = "resultados_aninhado"
FAIXAS_BORDA = [(0, 130), (130, 300), (300, 500)]


def erros_2set(verdade: np.ndarray, previsao: np.ndarray, classes) -> dict:
    """Contagens de U_alpha, O_omega e F (somadas sobre as classes), P, N e
    numero de trocas reais de classe, para um arquivo."""
    tot = dict(Ua=0, Ow=0, F=0, P=0, N=0)
    troca = np.r_[False, verdade[1:] != verdade[:-1]]      # True onde a classe real muda
    for c in classes:
        cod = 2 * (verdade == c) + (previsao == c)          # 0 TN, 1 FP, 2 FN, 3 TP
        tot["P"] += int((verdade == c).sum())
        tot["N"] += int((verdade != c).sum())
        ini = np.r_[0, np.flatnonzero(np.diff(cod)) + 1]     # inicio de cada segmento
        fim = np.r_[ini[1:], len(cod)]
        seg = cod[ini]
        ant = np.r_[-1, seg[:-1]]
        prox = np.r_[seg[1:], -1]
        tam = fim - ini
        fn, fp = seg == 2, seg == 1
        tot["F"] += int(tam[fn & (ant == 3) & (prox == 3)].sum())
        tot["Ua"] += int(tam[fn & (ant != 3) & (prox == 3) & troca[ini]].sum())
        tot["Ow"] += int(tam[fp & (ant == 3) & (prox != 3) & troca[ini]].sum())
    tot["trocas"] = int(troca.sum())
    return tot


def analisar_dataset(raiz: Path, ds: str, t0: int) -> list:
    base = raiz / ds / PASTA_RESULTADOS / f"T0_{t0}"
    acum, borda = {}, {}
    for arq in sorted(base.glob("ext*/*/teste/**/predicoes.csv.gz")):
        res = json.loads((arq.parent / "resultado_teste.json").read_text())
        bloco = arq.relative_to(base).parts[1]
        chave = (res["scenario"], bloco, res.get("detector_name") or "")
        p = pd.read_csv(arq)
        col = [c for c in p.columns if c.startswith("pred_")][0]
        b = borda.setdefault(chave, {f: [0, 0, 0] for f in FAIXAS_BORDA})  # acertos, previstas, esperadas
        n_arq = p["arquivo"].nunique()
        for lo, hi in FAIXAS_BORDA:
            m = p[(p["t"] >= lo) & (p["t"] < hi)]
            b[(lo, hi)][0] += int((m["y"] == m[col]).sum())
            b[(lo, hi)][1] += len(m)
            b[(lo, hi)][2] += (hi - lo) * n_arq
        p = p[p["pareada"]]
        classes = np.unique(p["y"])
        a = acum.setdefault(chave, dict(Ua=0, Ow=0, F=0, P=0, N=0, trocas=0))
        for _, g in p.sort_values(["arquivo", "t"]).groupby("arquivo"):
            for k, v in erros_2set(g["y"].to_numpy(), g[col].to_numpy(), classes).items():
                a[k] += v
    linhas = []
    for (cen, bloco, det), a in acum.items():
        tr = max(a["trocas"], 1)
        linhas.append({
            "dataset": ds, "condicao": ds.split("_")[0], "cenario": cen, "bloco": bloco,
            "detector": det,
            "u_alpha": a["Ua"] / a["P"], "o_omega": a["Ow"] / a["N"], "fr": a["F"] / a["P"],
            "U_alpha_por_troca": a["Ua"] / tr, "O_omega_por_troca": a["Ow"] / tr,
            "F_por_troca": a["F"] / tr,
            "U_alpha_amostras": a["Ua"], "O_omega_amostras": a["Ow"], "F_amostras": a["F"],
            "P": a["P"], "N": a["N"], "trocas": a["trocas"],
        })
    linhas_borda = []
    for (cen, bloco, det), b in borda.items():
        r = {"dataset": ds, "condicao": ds.split("_")[0], "cenario": cen, "bloco": bloco, "detector": det}
        for (lo, hi), (ac, prev, esp) in b.items():
            r[f"acerto_{lo}_{hi}"] = ac / prev if prev else np.nan
            r[f"cobertura_{lo}_{hi}"] = prev / esp
        linhas_borda.append(r)
    return linhas, linhas_borda


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--raiz", type=Path, default=RAIZ)
    pa.add_argument("--condicoes", nargs="+", default=["V", "F", "M", "H", "C"])
    pa.add_argument("--t0", type=int, default=499)
    a = pa.parse_args()

    datasets = sorted(d.name for d in a.raiz.iterdir()
                      if d.is_dir() and d.name.split("_")[0] in a.condicoes
                      and (d / PASTA_RESULTADOS / f"T0_{a.t0}").exists())
    if not datasets:
        raise SystemExit(f"Nenhum resultado das condicoes {a.condicoes} em {a.raiz}")
    linhas, linhas_borda = [], []
    for ds in datasets:
        l2, lb = analisar_dataset(a.raiz, ds, a.t0)
        linhas += l2
        linhas_borda += lb
        print(f"[OK] {ds}", flush=True)
    ordem = ["dataset", "cenario", "bloco", "detector"]
    df = pd.DataFrame(linhas).sort_values(ordem)
    dfb = pd.DataFrame(linhas_borda).sort_values(ordem)
    saida = a.raiz / f"tabela_2set_T0_{a.t0}.csv"
    saida_b = a.raiz / f"tabela_borda_T0_{a.t0}.csv"
    df.to_csv(saida, index=False)
    dfb.to_csv(saida_b, index=False)
    cols = ["u_alpha", "o_omega", "fr", "U_alpha_por_troca", "O_omega_por_troca", "F_por_troca"]
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("=== Fronteira entre classes (2SET) ===")
        print(df.set_index(ordem)[cols])
        print("\n=== Inicio do arquivo: acerto e cobertura ===")
        print(dfb.drop(columns="condicao").set_index(ordem))
    print(f"\nTabelas: {saida}, {saida_b}")


if __name__ == "__main__":
    main()
