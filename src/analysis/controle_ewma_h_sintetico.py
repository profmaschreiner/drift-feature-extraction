"""
controle_ewma_h_sintetico.py
============================
Controle EWMA dos alarmes nos sinteticos fatoriais COM A OTIMIZACAO ISOLADA,
como nos dados reais (controle_ewma.py): a EWMA nao tem busca propria. Em cada
fold externo e para cada detector, ela usa exatamente a configuracao que o
Optuna escolheu para o Algorithm 1 naquele externo (hiperparametros do RF,
scaler do modelo, scaler dos detectores e, em C3, a janela), e so o acumulador
muda:

    S_t = (1 - lam) S_{t-1} + lam a_t,   lam = fator / h(phi_b),

em que h(phi_b) e o horizonte de memoria do Algorithm 1 com o phi_b escolhido
(menor h com sum_{k<=h} -ln(1 - min(1, phi_b ln(1+k))) >= 1, o score cai a 1/e
apos um alarme isolado). Duas variantes, ambas sem tocar nos arquivos de teste
antes do teste:
    h1    fator = 1 (lam = 1/h; nenhuma selecao)
    hsel  fator em {0,5; 1; 2} escolhido pela validacao interna (leave-one-file-
          out nos 8 arquivos de otimizacao do externo, mesmo corte T0 e mesmo
          objetivo do Optuna), como a grade de 3 avaliacoes dos dados reais,
          mas sem olhar o teste (protocolo aninhado)
Diferencas em relacao ao Algorithm 1 passam a refletir so a forma do
acumulador, nao a variancia da busca de hiperparametros.

Reaproveita o rodar_sintetico_aninhado.py (mesmas particoes, corte T0, teste
externo, previsoes por amostra). Requer a rodada principal concluida (as
configuracoes do Algorithm 1 em .../ext<k>/<bloco>/otimizacao/).

Saidas, dentro de cada .../T0_<T0>/ext<k>/<bloco>/:
    otimizacao_ewma_<v>/<cenario>/EWMA_ALARM_<det>/resultado_otimizacao.json
        (configuracao usada, lam, h, fator e grade interna)
    teste_ewma_<v>/<cenario>/EWMA_ALARM_<det>/predicoes.csv.gz, resultado_teste.json
e, na raiz do fatorial:
    ewma_h_por_arquivo_T0_<T0>.csv  F1 pareado por arquivo: Algorithm 1, h1, hsel
                                    (em C, tambem o F1 so das transicoes T_AB e T_BA)
    ewma_h_resumo_T0_<T0>.csv       por dataset, bloco e detector: medias e diferencas de F1,
                                    e U_alpha, fr, nao reconhecidos e (em C) troca, com as
                                    mesmas definicoes do gerar_resultados_artigo.py

--controle var aplica o MESMO protocolo ao EWMA_VAR (variancia exponencial do
sinal normalizado, sem alarmes): para cada detector, a configuracao do Algorithm 1
daquele detector, com lam pelo mesmo horizonte; a comparacao e pareada por
configuracao (alarmes x variancia, mesmo alcance e mesmo classificador). Saidas em
otimizacao_ewmavar_<v>/<cenario>/EWMA_VAR_cfg_<det>/, teste_ewmavar_<v>/... e
ewmavar_h_{por_arquivo,resumo}_T0_<T0>.csv. Os dois controles usam pastas e caches
proprios e podem rodar ao mesmo tempo.

Uso (na mesma pasta do rodar_sintetico_aninhado.py, p. ex. ~/am_reais):
    python3 controle_ewma_h_sintetico.py --condicoes V F M H C --paralelo 8
    python3 controle_ewma_h_sintetico.py --condicoes F --blocos 0 130 --detectores CUSUM SEED
    python3 controle_ewma_h_sintetico.py --so-resumo --condicoes V F M H C
    python3 controle_ewma_h_sintetico.py --controle var --condicoes V F M H C --paralelo 8
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RUNNER = AQUI / "rodar_sintetico_aninhado.py"
FATORES = [0.5, 1.0, 2.0]
VARIANTES = ("h1", "hsel")
TRANSICOES = [2, 3]

# Controles com o mesmo protocolo (configuracao do Algorithm 1 de cada detector,
# lambda pelo horizonte de memoria do phi_b escolhido):
#   alarm: EWMA dos alarmes do proprio detector (troca so o acumulador)
#   var:   EWMA_VAR, variancia exponencial do sinal normalizado, sem alarmes (troca o
#          sinal acumulado); uma avaliacao por configuracao de detector, pareada com ele
CONTROLES = {
    "alarm": dict(tag="ewma", interno="ewma_h_interno", saida="ewma_h"),
    "var": dict(tag="ewmavar", interno="ewmavar_h_interno", saida="ewmavar_h"),
}
CTRL = "alarm"


def nome_pseudo(R, det):
    """Nome que o experimento usa para gerar o score."""
    return R.PREFIXO_EWMA_ALARM + det if CTRL == "alarm" else "EWMA_VAR"


def pasta_det(R, det):
    """Subpasta de cada configuracao (uma por detector nos dois controles)."""
    return R.PREFIXO_EWMA_ALARM + det if CTRL == "alarm" else f"EWMA_VAR_cfg_{det}"


def tag():
    return CONTROLES[CTRL]["tag"]


def carregar_runner(experimento=None, raiz=None):
    spec = importlib.util.spec_from_file_location("rodar_sintetico_aninhado", RUNNER)
    R = importlib.util.module_from_spec(spec)
    sys.modules["rodar_sintetico_aninhado"] = R
    spec.loader.exec_module(R)
    if experimento is not None:
        R.CAMINHO_EXPERIMENTO = Path(experimento)
    if raiz is not None:
        R.RAIZ = Path(raiz)
    return R


def horizonte(phi_b: float) -> int:
    """Menor h com sum_{k=1..h} -ln(1 - min(1, phi_b ln(1+k))) >= 1."""
    from scipy.special import gammaln
    phi_b = float(phi_b)
    if phi_b >= 1e-4:
        s, h = 0.0, 0
        while s < 1.0:
            h += 1
            f = min(1.0, phi_b * math.log1p(h))
            if f >= 1.0:
                return h
            s += -math.log1p(-f)
        return h
    lo, hi = 1, 10 ** 15
    while lo < hi:
        mid = (lo + hi) // 2
        if phi_b * gammaln(mid + 2) >= 1.0:
            hi = mid
        else:
            lo = mid + 1
    return lo


def configs_alg1(pasta_otim: Path, cenario: str, detectores):
    """Configuracao escolhida pelo Optuna para o Algorithm 1 de cada detector."""
    out = {}
    for arq in pasta_otim.glob("**/resultado_otimizacao.json"):
        js = json.loads(arq.read_text())
        det = js.get("detector_name")
        if js.get("scenario") == cenario and det in detectores:
            out[det] = js
    return out


# ============================================================
# JOB (dataset x bloco x externo), dentro do subprocesso
# ============================================================
def rodar_job(R, dataset: str, janela: int, externo: int, t0: int, detectores):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ds = R.RAIZ / dataset
    base = R.pasta_bloco(ds, t0, externo, R.nome_bloco(janela))
    cen = "drift" if janela == 0 else "catch24_drift"
    cfg = configs_alg1(base / "otimizacao", cen, detectores)
    faltam = [d for d in detectores if d not in cfg]
    if faltam:
        print(f"[AVISO] sem configuracao do Algorithm 1 para {faltam} em {base / 'otimizacao'}", flush=True)
    pend = [d for d in detectores if d in cfg and not all(
        (base / f"teste_{tag()}_{v}" / cen / pasta_det(R, d) / "resultado_teste.json").exists()
        for v in VARIANTES)]
    if not pend:
        print(f"[SKIP] {dataset} {R.nome_bloco(janela)} ext{externo}: nada a fazer", flush=True)
        return

    mod = R.carregar_experimento()
    ids_teste_global = R.EXTERNOS[externo]
    ids_otim_global = [i for i in range(R.N_ARQUIVOS) if i not in ids_teste_global]
    mod.PASTA_ENTRADA = str(ds)
    mod.ROTULO = R.ROTULO
    mod.FEATURES_ORIGINAIS = list(R.FEATURES_ORIGINAIS)
    mod.NOME_BASE, mod.END_BASE = R.NOME_BASE, R.END_BASE
    mod.DETECTORES_DISPONIVEIS = (["EWMA_VAR"] if CTRL == "var" else [R.PREFIXO_EWMA_ALARM + d for d in pend])
    # so os scalers que o Algorithm 1 escolheu (evita calcular alarmes com os outros)
    mod.SCALERS_DISPONIVEIS = sorted({cfg[d]["best_params"]["scaler_detectors"] for d in pend})
    R.instalar_pseudodetectores(mod)
    mod.N_JOBS_DRIFT = 1
    mod.N_JOBS_CATCH24 = 1
    mod.N_JOBS_RF = R.N_JOBS_RF
    if janela != 0:
        mod.WINDOW_MIN = mod.WINDOW_MAX = janela
        mod.WINDOW_STEP = 1

    dfs_global = mod.load_dados(str(ds))
    assert len(dfs_global) == R.N_ARQUIVOS, f"{dataset}: {len(dfs_global)} arquivos"
    dfs_otim = [dfs_global[i] for i in ids_otim_global]
    mod.CONJ_TESTE = [[i] for i in range(len(dfs_otim))]
    combinacoes = mod.build_combinacoes(len(dfs_otim))      # LOFO interno, como o Optuna

    build_original = mod.build_dataset_for_item
    R.instalar_corte(mod, t0)
    build_cortado = mod.build_dataset_for_item

    pasta_int = base / CONTROLES[CTRL]["interno"]             # cache da validacao interna
    mod.PASTA_SAIDA = str(pasta_int)
    print(f"\n[JOB {CTRL}] {dataset} | {R.nome_bloco(janela)} | externo {externo} | configuracoes {pend}", flush=True)
    mod.clear_runtime_caches()
    try:
        mod.precompute_drift_files(dfs_otim, combinacoes, str(pasta_int))
        for det in pend:
            js = cfg[det]
            bp = dict(js["best_params"])
            phi_b = float(bp["phi_b"])
            h = horizonte(phi_b)
            lam0 = 1.0 / h
            jan = js.get("fixed_catch24_window_size") or bp.get("window_size") or (janela or None)
            nome = nome_pseudo(R, det)
            grade = []
            for f in FATORES:
                params = dict(bp)
                params["phi_b"] = min(1.0, f * lam0)                 # lambda no lugar do phi_b
                obj = mod.make_objective(dfs_otim, combinacoes, str(pasta_int), cen, nome,
                                         fixed_catch24_window_size=(int(jan) if janela else None))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")                  # FixedTrial fora do intervalo de busca
                    valor = float(obj(optuna.trial.FixedTrial(params)))
                grade.append(dict(fator=f, lam=params["phi_b"], f1_interno=valor))
                print(f"    {det:24s} fator {f:3.1f} lam={params['phi_b']:.3e} (h={h}): F1 interno {valor:.4f}",
                      flush=True)
            sel = max(grade, key=lambda g: g["f1_interno"])
            for v, g in (("h1", next(g for g in grade if g["fator"] == 1.0)), ("hsel", sel)):
                params = dict(bp)
                params["phi_b"] = g["lam"]
                destino = base / f"otimizacao_{tag()}_{v}" / cen / pasta_det(R, det)
                destino.mkdir(parents=True, exist_ok=True)
                (destino / "resultado_otimizacao.json").write_text(json.dumps({
                    "scenario": cen, "detector_name": nome, "configuracao_de": det, "controle": CTRL,
                    "best_params": params,
                    "best_macro_f1": g["f1_interno"], "fixed_catch24_window_size": jan,
                    "variante": v, "fator": g["fator"], "lambda": g["lam"], "horizonte_h": h,
                    "phi_b_algorithm1": phi_b, "f1_interno_algorithm1": js.get("best_macro_f1"),
                    "grade_interna": grade}, indent=2, default=float))
        mod.clear_runtime_caches()
        dfs_all = dfs_otim + [dfs_global[i] for i in ids_teste_global]
        ids_otim = list(range(len(dfs_otim)))
        ids_teste = list(range(len(dfs_otim), len(dfs_all)))
        for v in VARIANTES:
            R.avaliar_teste(mod, build_original, build_cortado, dfs_all, ids_otim, ids_teste,
                            base / f"otimizacao_{tag()}_{v}", base / f"teste_{tag()}_{v}", t0)
            mod.clear_runtime_caches()
    finally:
        mod.clear_runtime_caches()
        for p in [pasta_int] + [base / f"teste_{tag()}_{v}" for v in VARIANTES]:
            mod.remove_temp_dirs(str(p))


# ============================================================
# CONSOLIDACAO
# ============================================================
def _contagens_fronteira(y, p, alvo, outra=None):
    """Contagens de fronteira (Ward et al., 2011), iguais as do
    gerar_resultados_artigo.py: segmentos verdadeiros das classes em `alvo`,
    ignorando o primeiro (comeca no corte do trecho pareado)."""
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


def _f1_por_arquivo(arq_pred: Path, eh_c: bool):
    """F1 pareado por arquivo e contagens de fronteira (em C, so as transicoes)."""
    from sklearn.metrics import f1_score
    p = pd.read_csv(arq_pred)
    col = [c for c in p.columns if c.startswith("pred_")][0]
    out = {}
    for nome, g in p[p["pareada"]].sort_values("t").groupby("arquivo"):
        y, pr = g["y"].to_numpy(), g[col].to_numpy()
        r = {"f1": f1_score(y, pr, average="macro")}
        if eh_c:
            r["f1_transicoes"] = f1_score(y, pr, labels=TRANSICOES, average="macro", zero_division=0)
        alvo = TRANSICOES if eh_c else list(np.unique(y))
        cont = _contagens_fronteira(y, pr, alvo, {2: 3, 3: 2} if eh_c else None)
        r.update({f"n_{k}": v for k, v in cont.items()})
        out[nome] = r
    return out


def consolidar(R, datasets, t0: int, blocos, detectores):
    linhas = []
    for ds in datasets:
        eh_c = ds.startswith("C_")
        for janela in blocos:
            cen = "drift" if janela == 0 else "catch24_drift"
            for externo in range(len(R.EXTERNOS)):
                base = R.pasta_bloco(R.RAIZ / ds, t0, externo, R.nome_bloco(janela))
                alg = {}
                for arq in (base / "teste").glob("**/resultado_teste.json"):
                    js = json.loads(arq.read_text())
                    if js.get("scenario") == cen and js.get("detector_name") in detectores:
                        alg[js["detector_name"]] = arq.parent / "predicoes.csv.gz"
                for det in detectores:
                    fontes = {"alg1": alg.get(det)}
                    for v in VARIANTES:
                        fontes[v] = base / f"teste_{tag()}_{v}" / cen / pasta_det(R, det) / "predicoes.csv.gz"
                    if any(f is None or not Path(f).exists() for f in fontes.values()):
                        continue
                    f1s = {k: _f1_por_arquivo(Path(f), eh_c) for k, f in fontes.items()}
                    lam = {v: json.loads((base / f"otimizacao_{tag()}_{v}" / cen / pasta_det(R, det) /
                                          "resultado_otimizacao.json").read_text()) for v in VARIANTES}
                    for nome_arq in f1s["alg1"]:
                        r = dict(dataset=ds, bloco=R.nome_bloco(janela), cenario="C1" if janela == 0 else "C3",
                                 detector=det, externo=externo, arquivo=nome_arq,
                                 horizonte_h=lam["h1"]["horizonte_h"], fator_hsel=lam["hsel"]["fator"])
                        for k in fontes:
                            for m, val in f1s[k][nome_arq].items():
                                r[f"{m}_{k}"] = val
                        linhas.append(r)
    return pd.DataFrame(linhas)


def resumir(t: pd.DataFrame):
    metricas = ["f1"] + (["f1_transicoes"] if "f1_transicoes_alg1" in t else [])
    linhas = []
    for (ds, cen, det), g in t.groupby(["dataset", "cenario", "detector"], sort=False):
        r = dict(dataset=ds, cenario=cen, detector=det, n_arquivos=len(g),
                 fatores_hsel=" ".join(f"{x:g}" for x in g.groupby("externo").fator_hsel.first()))
        for m in metricas:
            if g[f"{m}_alg1"].isna().all():
                continue
            for k in ("alg1",) + VARIANTES:
                r[f"{m}_{k}"] = 100 * g[f"{m}_{k}"].mean()
            for v in VARIANTES:
                d = 100 * (g[f"{m}_alg1"] - g[f"{m}_{v}"])
                r[f"{m}_dif_{v}"] = d.mean()
                r[f"{m}_arq_alg1_melhor_{v}"] = int((d > 0).sum())
        # fronteira (Ward): somas de todos os arquivos, como no gerar_resultados_artigo.py
        for k in ("alg1",) + VARIANTES:
            soma = {c: g[f"n_{c}_{k}"].sum() for c in ("seg", "rec", "Ua", "frag", "amostras", "troca")}
            r[f"Ua_{k}"] = soma["Ua"] / soma["rec"] if soma["rec"] else np.nan
            r[f"fr_{k}"] = soma["frag"] / soma["amostras"] if soma["amostras"] else np.nan
            r[f"nao_rec_{k}"] = 1 - soma["rec"] / soma["seg"] if soma["seg"] else np.nan
            if ds.startswith("C_"):
                r[f"troca_{k}"] = 100 * soma["troca"] / soma["amostras"] if soma["amostras"] else np.nan
        linhas.append(r)
    return pd.DataFrame(linhas)


# ============================================================
# ORQUESTRACAO
# ============================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--condicoes", nargs="+", default=["V", "F", "M", "H", "C"])
    pa.add_argument("--blocos", nargs="+", type=int, default=[0, 130],
                    help="0 = sem janela (C1); 130 = janela do C3")
    pa.add_argument("--detectores", nargs="+", default=None)
    pa.add_argument("--t0", type=int, default=499)
    pa.add_argument("--paralelo", type=int, default=8)
    pa.add_argument("--controle", choices=list(CONTROLES), default="alarm",
                    help="alarm = EWMA dos alarmes; var = EWMA_VAR (sem alarmes)")
    pa.add_argument("--so-resumo", action="store_true")
    pa.add_argument("--raiz", type=Path, default=None)
    pa.add_argument("--experimento", type=Path, default=None, help="(teste) outro modulo de experimento")
    pa.add_argument("--job-dataset")
    pa.add_argument("--job-janela", type=int)
    pa.add_argument("--job-externo", type=int)
    a = pa.parse_args()
    global CTRL
    CTRL = a.controle
    R = carregar_runner(a.experimento, a.raiz)
    dets = a.detectores or list(R.DETECTORES_PADRAO)

    if a.job_dataset is not None:
        rodar_job(R, a.job_dataset, a.job_janela, a.job_externo, a.t0, dets)
        return

    datasets = sorted(d.name for d in R.RAIZ.iterdir()
                      if d.is_dir() and d.name.split("_")[0] in a.condicoes
                      and (d / R.PASTA_RESULTADOS / f"T0_{a.t0}").exists())
    if not datasets:
        sys.exit(f"Nenhum dataset das condicoes {a.condicoes} com T0_{a.t0} em {R.RAIZ.resolve()}")
    print(f"Datasets: {datasets}\nBlocos: {a.blocos} | T0 = {a.t0} | detectores: {dets}", flush=True)

    if not a.so_resumo:
        jobs = [(ds, j, e) for ds in datasets for j in a.blocos for e in range(len(R.EXTERNOS))]
        logs = R.RAIZ / f"logs_{CONTROLES[CTRL]['saida']}_T0_{a.t0}"
        logs.mkdir(exist_ok=True)
        pendentes, ativos, falhas = list(jobs), [], []
        while pendentes or ativos:
            while pendentes and len(ativos) < a.paralelo:
                ds, j, e = pendentes.pop(0)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--job-dataset", ds, "--job-janela", str(j),
                       "--job-externo", str(e), "--t0", str(a.t0), "--raiz", str(R.RAIZ), "--controle", CTRL, "--detectores", *dets]
                if a.experimento is not None:
                    cmd += ["--experimento", str(a.experimento)]
                nome = f"{ds}_{R.nome_bloco(j)}_ext{e}"
                log = open(logs / f"{nome}.log", "w")
                ativos.append((nome, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
                print(f"[INICIO] {nome}", flush=True)
            for job in list(ativos):
                nome, proc, log = job
                if proc.poll() is not None:
                    log.close()
                    ativos.remove(job)
                    if proc.returncode != 0:
                        falhas.append(nome)
                    print(f"[{'FIM' if proc.returncode == 0 else 'ERRO'}] {nome}"
                          f"{'' if proc.returncode == 0 else f' (ver {logs}/{nome}.log)'}", flush=True)
            time.sleep(5)
        if falhas:
            print(f"[AVISO] jobs com erro: {falhas}", flush=True)

    t = consolidar(R, datasets, a.t0, a.blocos, dets)
    if t.empty:
        sys.exit("[ERRO] nenhum resultado para consolidar")
    saida = CONTROLES[CTRL]["saida"]
    t.to_csv(R.RAIZ / f"{saida}_por_arquivo_T0_{a.t0}.csv", index=False)
    r = resumir(t)
    r.to_csv(R.RAIZ / f"{saida}_resumo_T0_{a.t0}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.max_columns", 40,
                           "display.float_format", "{:.2f}".format):
        print(f"\nControle: {CTRL}. F1 em %; dif = Algorithm 1 - controle (p.p.; positivo favorece o Algorithm 1); "
              "arq_alg1_melhor = arquivos (de 12) em que o Algorithm 1 e melhor")
        print(r.to_string(index=False))
        for v in VARIANTES:
            d = r["f1_dif_" + v]
            print(f"\n[{v}] celulas: {len(d)}; |dif| <= 3 p.p.: {int((d.abs() <= 3).sum())}; "
                  f"Alg. 1 > 5 p.p.: {int((d > 5).sum())}; controle > 5 p.p.: {int((d < -5).sum())}; "
                  f"mediana {d.median():+.2f} p.p.")
            dUa = r["Ua_alg1"] - r["Ua_" + v]
            dfr = r["fr_alg1"] - r["fr_" + v]
            print(f"     fronteira: mediana de U_alpha Alg. 1 - controle {dUa.median():+.1f} amostras; "
                  f"de fr {dfr.median():+.3f} (negativo favorece o Alg. 1)")
            if "f1_transicoes_dif_" + v in r:
                dc = r["f1_transicoes_dif_" + v].dropna()
                print(f"     C, F1 das transicoes: mediana {dc.median():+.2f} p.p.; Alg. 1 > 5: "
                      f"{int((dc > 5).sum())}; controle > 5: {int((dc < -5).sum())} (de {len(dc)})")
    print(f"\nSalvo: {R.RAIZ / f'{saida}_por_arquivo_T0_{a.t0}.csv'} e {R.RAIZ / f'{saida}_resumo_T0_{a.t0}.csv'}")


if __name__ == "__main__":
    main()