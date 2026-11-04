"""
rodar_sintetico_aninhado.py
===========================
Avaliacao dos sinteticos fatoriais com otimizacao e teste separados
(validacao cruzada aninhada), comparacao de janelas do catch24 e gravacao das
previsoes por amostra para as analises de borda e de transicao.

Substitui o rodar_janelas_longas.py.

1) Particao externa (3 folds, sem sobreposicao):
       externo 0: testa nos arquivos 0-3,  otimiza nos arquivos 4-11
       externo 1: testa nos arquivos 4-7,  otimiza nos arquivos 0-3 e 8-11
       externo 2: testa nos arquivos 8-11, otimiza nos arquivos 0-7
   Cada grupo de teste tem 2 arquivos pares e 2 impares, entao a ordem das
   classes fica balanceada. Cada arquivo e teste exatamente uma vez, e os
   arquivos de teste nunca participam da otimizacao.

2) Otimizacao (interna): o experimento original, inalterado, roda o Optuna com
   leave-one-file-out nos 8 arquivos de otimizacao (8 folds internos: 7 de
   treino e 1 de validacao). Custo total por trial: 3 x 8 = 24 ajustes.

3) Teste (externo): com os melhores parametros de cada cenario, o modelo e
   re-treinado nos 8 arquivos de otimizacao (scaler dos detectores ajustado
   neles) e aplicado uma unica vez aos 4 arquivos de teste. As previsoes de
   cada amostra de teste sao salvas (predicoes.csv.gz).

4) Duas avaliacoes no teste:
   - pareada: so as amostras T0 <= t < n - T0, iguais para todos os cenarios
     e janelas (T0 = maior janela - 1). Compara o conteudo de informacao.
   - completa: todas as amostras que cada metodo consegue classificar.
     baseline/drift classificam desde t = 0; catch24 com janela W so a partir
     de t = W - 1. Mede tambem a cobertura (fracao de amostras com previsao),
     que e o custo da janela longa nas bordas.
   A otimizacao interna usa o corte pareado (corte simetrico, mantendo as
   classes balanceadas por arquivo).

5) Pseudodetector EWMA_VAR (controle de ablacao do detector): em vez de
   alarmes, o score de cada variavel e a variancia exponencial do sinal
   normalizado, v_t = (1-lam) v_{t-1} + lam (x_t - m_{t-1})^2, com
   m_t = (1-lam) m_{t-1} + lam x_t e lam = phi_b. Entra no lugar de um
   detector nos cenarios drift (C1) e catch24_drift (C3), com o mesmo
   Optuna, o mesmo espaco de busca, a mesma normalizacao (scaler_detectors
   ajustado no treino de cada fold) e o mesmo protocolo aninhado. Para
   acrescenta-lo a uma rodada ja concluida:
       python3 rodar_sintetico_aninhado.py --condicoes ... --janelas ... --detectores EWMA_VAR
   Nesse modo complementar, baseline e catch24 ja testados nao sao
   re-otimizados (o catch24 so fornece a janela fixa W).

   Pseudodetectores EWMA_ALARM_<det> (ablacao da funcao de score): usa os
   alarmes do detector <det>, mas troca o Algorithm 1 por uma EWMA com
   esquecimento fixo e peso unitario, S_t = (1-lam) S_{t-1} + lam a_t, com
   lam = phi_b. --detectores EWMA_ALARM expande para os 9 detectores.
       python3 rodar_sintetico_aninhado.py ... --detectores EWMA_VAR EWMA_ALARM

6) As previsoes por amostra permitem as analises de borda (inicio do arquivo)
   e de transicao (acerto em funcao da distancia a troca de classe), feitas
   depois, sem rodar nada de novo.

Paralelismo: dentro de cada job o paralelismo interno do experimento (joblib
com processos) e desligado, porque os processos filhos reimportariam o modulo
com a configuracao original do arquivo. Os jobs (dataset x janela x externo)
rodam em paralelo (--paralelo).

Saida, dentro de cada pasta de dataset:
    resultados_aninhado/T0_<T0>/ext<k>/<bloco>/otimizacao/...   (Optuna interno)
    resultados_aninhado/T0_<T0>/ext<k>/<bloco>/teste/<cenario>[/<det>]/
        predicoes.csv.gz, resultado_teste.json
  <bloco> = sem_janela (baseline, drift) ou w<W> (catch24, catch24_drift)
e, na pasta raiz:
    resumo_aninhado_T0_<T0>.csv            (uma linha por cenario/detector/externo)
    resumo_aninhado_por_arquivo_T0_<T0>.csv (macro F1 de teste por arquivo)

Uso (na pasta raiz do fatorial):
    python3 rodar_sintetico_aninhado.py --condicoes F H C --janelas 130 500
    python3 rodar_sintetico_aninhado.py --condicoes V M --janelas 130
    python3 rodar_sintetico_aninhado.py --so-resumo --condicoes F H C --janelas 130 500
"""

from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACAO
# ============================================================
CAMINHO_EXPERIMENTO = Path("experimento.py")
RAIZ = Path(".")
CONDICOES = ["F", "H", "C"]
JANELAS = [130, 500]
DETECTORES = None                      # None = os 9 do experimento
PARALELO = 4
N_JOBS_RF = 2

ROTULO = "y"
FEATURES_ORIGINAIS = ["x1"]
NOME_BASE = "base_"
END_BASE = ".csv"
N_ARQUIVOS = 12
EXTERNOS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]   # arquivos de TESTE de cada externo
PASTA_RESULTADOS = "resultados_aninhado"
DETECTORES_PADRAO = ["ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
                     "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED"]
PREFIXO_EWMA_ALARM = "EWMA_ALARM_"


def eh_pseudo(nome: str) -> bool:
    return nome == "EWMA_VAR" or nome.startswith(PREFIXO_EWMA_ALARM)


def expandir_detectores(dets):
    """EWMA_ALARM sozinho vira EWMA_ALARM_<det> para os 9 detectores."""
    if not dets:
        return dets
    saida = []
    for d in dets:
        saida.extend([PREFIXO_EWMA_ALARM + b for b in DETECTORES_PADRAO] if d == "EWMA_ALARM" else [d])
    return list(dict.fromkeys(saida))


def pasta_bloco(dataset: Path, t0: int, externo: int, bloco: str) -> Path:
    return dataset / PASTA_RESULTADOS / f"T0_{t0}" / f"ext{externo}" / bloco


def nome_bloco(janela: int) -> str:
    return "sem_janela" if janela == 0 else f"w{janela}"


# ============================================================
# MODULO DO EXPERIMENTO
# ============================================================
def carregar_experimento():
    if not CAMINHO_EXPERIMENTO.exists():
        raise FileNotFoundError(f"Experimento nao encontrado: {CAMINHO_EXPERIMENTO}")
    spec = importlib.util.spec_from_file_location("experimento_real", CAMINHO_EXPERIMENTO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experimento_real"] = mod
    spec.loader.exec_module(mod)
    return mod


def inicio_cenario(scenario_name: str, window_size) -> int:
    return 0 if scenario_name in {"baseline", "drift"} else int(window_size) - 1


def instalar_corte(mod, t0: int):
    """Para a otimizacao: todo cenario cobre exatamente t0 <= t < n - t0."""
    original = mod.build_dataset_for_item

    def build_com_corte(item, base_id, scenario_name, window_size, *args, **kwargs):
        X, y = original(item, base_id, scenario_name, window_size, *args, **kwargs)
        if X.empty:
            return X, y
        k = t0 - inicio_cenario(scenario_name, window_size)
        if k < 0:
            raise ValueError(f"T0={t0} menor que o inicio do cenario {scenario_name}.")
        n_fim = len(y) - t0
        X = X.iloc[k:n_fim].reset_index(drop=True)
        y = y[k:n_fim]
        esperado = len(item["df"]) - 2 * t0
        if len(y) != esperado:
            raise AssertionError(f"{scenario_name} w={window_size}: {len(y)} amostras, esperado {esperado}")
        return X, y

    mod.build_dataset_for_item = build_com_corte
    return original


# ============================================================
# PSEUDODETECTOR EWMA_VAR
# ============================================================
def ewma_var(x: np.ndarray, lam: float) -> np.ndarray:
    """v_t = (1-lam) v_{t-1} + lam (x_t - m_{t-1})^2, m_t = (1-lam) m_{t-1} + lam x_t,
    com m_{-1} = 0 e v_{-1} = 1 (sinal normalizado). Causal."""
    from scipy.signal import lfilter
    a = 1.0 - lam
    m = lfilter([lam], [1.0, -a], x, zi=[0.0])[0]
    m_ant = np.r_[0.0, m[:-1]]
    return lfilter([lam], [1.0, -a], (x - m_ant) ** 2, zi=[a * 1.0])[0]


def instalar_pseudodetectores(mod):
    """precompute_drift_files: roda o original so para os detectores reais e,
    para os pseudodetectores, guarda o scaler de cada (pasta, fold, scaler),
    ajustado no treino do fold, como o experimento faz para os detectores.
    get_drift_enriched_df: para um pseudodetector, gera score_<col> com a
    variancia exponencial do sinal normalizado (lam = phi_b)."""
    precompute_original = mod.precompute_drift_files
    enriched_original = mod.get_drift_enriched_df
    scalers = {}

    def precompute(dfs_info, combinacoes, pasta_saida):
        dets = list(mod.DETECTORES_DISPONIVEIS)
        reais = [d for d in dets if not eh_pseudo(d)]
        reais += [d[len(PREFIXO_EWMA_ALARM):] for d in dets if d.startswith(PREFIXO_EWMA_ALARM)]
        reais = list(dict.fromkeys(reais))
        if reais:
            mod.DETECTORES_DISPONIVEIS = reais
            try:
                precompute_original(dfs_info, combinacoes, pasta_saida)
            finally:
                mod.DETECTORES_DISPONIVEIS = dets
        if "EWMA_VAR" in dets:
            for id_fold, comb in enumerate(combinacoes):
                treino = pd.concat([dfs_info[i]["df"] for i in comb["treinamento"]], ignore_index=True)
                for nome in mod.SCALERS_DISPONIVEIS:
                    sc = mod.build_scaler_from_name(nome)
                    sc.fit(treino[mod.FEATURES_ORIGINAIS])
                    scalers[(str(pasta_saida), int(id_fold), nome)] = sc

    def enriched(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida):
        if not eh_pseudo(detector_name):
            return enriched_original(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida)
        if detector_name.startswith(PREFIXO_EWMA_ALARM):
            from scipy.signal import lfilter
            base = detector_name[len(PREFIXO_EWMA_ALARM):]
            csv = mod.get_drift_csv_path(mod.get_drift_cache_dir(pasta_saida), base, scaler_name,
                                         id_fold, split_name, item["nome_arquivo"])
            posicoes = mod.carregar_posicoes_drift(csv)
            df = item["df"].copy()
            n, lam = len(df), float(phi_b)
            for col in mod.FEATURES_ORIGINAIS:
                a = np.zeros(n)
                idx = posicoes[col]
                a[idx[(idx >= 0) & (idx < n)]] = 1.0
                df[f"score_{col}"] = lfilter([lam], [1.0, -(1.0 - lam)], a)
            return df
        sc = scalers[(str(pasta_saida), int(id_fold), scaler_name)]
        df = item["df"].copy()
        z = sc.transform(df[mod.FEATURES_ORIGINAIS])
        for j, col in enumerate(mod.FEATURES_ORIGINAIS):
            df[f"score_{col}"] = ewma_var(np.asarray(z[:, j], dtype=float), float(phi_b))
        return df

    mod.precompute_drift_files = precompute
    mod.get_drift_enriched_df = enriched


class _EstudoReaproveitado:
    """Substitui o estudo do catch24 ja testado: so fornece a janela fixa."""
    def __init__(self, janela):
        import types
        self.best_trial = types.SimpleNamespace(params={"window_size": int(janela)}, user_attrs={})
        self.best_value = float("nan")


def reaproveitar_catch24(mod, janela: int):
    run_original = mod.run_single_study
    save_original = mod.save_study_outputs

    def run_single_study(*args, **kwargs):
        cen = kwargs.get("scenario_name", args[3] if len(args) > 3 else None)
        if cen == "catch24":
            print(f"[REUSO] catch24 ja testado; janela fixa {janela}", flush=True)
            return _EstudoReaproveitado(janela)
        return run_original(*args, **kwargs)

    def save_study_outputs(study, *args, **kwargs):
        if isinstance(study, _EstudoReaproveitado):
            return {"scenario": "catch24", "detector_name": "baseline", "best_macro_f1": float("nan")}
        return save_original(study, *args, **kwargs)

    mod.run_single_study = run_single_study
    mod.save_study_outputs = save_study_outputs


# ============================================================
# TESTE EXTERNO
# ============================================================
def avaliar_teste(mod, build_original, build_cortado, dfs_all, ids_otim, ids_teste,
                  pasta_otim: Path, pasta_teste: Path, t0: int):
    """Re-treina cada cenario otimizado nos arquivos de otimizacao e preve os
    arquivos de teste. dfs_all = arquivos de otimizacao + teste (indices
    locais); o cache de drift/catch24 do teste fica em pasta_teste."""
    import optuna
    from sklearn.metrics import f1_score

    combinacao = [{"treinamento": ids_otim, "teste": ids_teste}]
    if any((pasta_otim / s).exists() for s in ("drift", "catch24_drift")):
        mod.precompute_drift_files(dfs_all, combinacao, str(pasta_teste))

    for arq in sorted(pasta_otim.glob("**/resultado_otimizacao.json")):
        res = json.loads(arq.read_text())
        cen = res["scenario"]
        det = res.get("detector_name")
        params = res["best_params"]
        janela = params.get("window_size") or res.get("fixed_catch24_window_size")
        destino = pasta_teste / arq.parent.relative_to(pasta_otim)
        if (destino / "resultado_teste.json").exists():
            continue
        destino.mkdir(parents=True, exist_ok=True)

        comum = dict(scenario_name=cen, window_size=janela, detector_name=det,
                     scaler_name=params.get("scaler_detectors"), phi_b=params.get("phi_b"),
                     id_fold=0, pasta_saida=str(pasta_teste))
        X_tr, y_tr = [], []
        for i in ids_otim:
            X, y = build_cortado(item=dfs_all[i], base_id=i, split_name="train", **comum)
            X_tr.append(X)
            y_tr.append(y)
        X_te, y_te, meta = [], [], []
        for i in ids_teste:
            X, y = build_original(item=dfs_all[i], base_id=i, split_name="test", **comum)
            n = len(dfs_all[i]["df"])
            t = np.arange(inicio_cenario(cen, janela), n)[:len(y)]
            X_te.append(X)
            y_te.append(y)
            meta.append(pd.DataFrame({"arquivo": dfs_all[i]["nome_arquivo"], "t": t,
                                      "n": n, "pareada": (t >= t0) & (t < n - t0)}))
        X_train, X_test, _, _ = mod.apply_column_policy(
            pd.concat(X_tr, ignore_index=True), pd.concat(X_te, ignore_index=True),
            nan_threshold=mod.NAN_THRESHOLD)
        y_train, y_test = np.concatenate(y_tr), np.concatenate(y_te)

        modelos = mod.build_models_from_trial(optuna.trial.FixedTrial(params))
        pred = pd.concat(meta, ignore_index=True)
        pred["y"] = y_test
        resumo = {"scenario": cen, "detector_name": det, "janela": janela,
                  "best_params": params, "f1_otimizacao": res["best_macro_f1"]}
        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred[f"pred_{nome}"] = modelo.predict(X_test)
            par = pred["pareada"].to_numpy()
            resumo[f"{nome}_f1_pareada"] = float(f1_score(y_test[par], pred.loc[par, f"pred_{nome}"], average="macro"))
            resumo[f"{nome}_f1_completa"] = float(f1_score(y_test, pred[f"pred_{nome}"], average="macro"))
        resumo["cobertura"] = float(len(pred) / sum(len(dfs_all[i]["df"]) for i in ids_teste))
        pred.to_csv(destino / "predicoes.csv.gz", index=False)
        (destino / "resultado_teste.json").write_text(json.dumps(resumo, indent=2, default=float))
        print(f"[TESTE] {destino.relative_to(pasta_teste.parents[2])}: "
              + ", ".join(f"{k}={v:.3f}" for k, v in resumo.items() if k.endswith(("pareada", "completa"))),
              flush=True)


# ============================================================
# JOB (dentro do subprocesso)
# ============================================================
def rodar_job(dataset: str, janela: int, externo: int, t0: int, detectores):
    mod = carregar_experimento()
    ds = RAIZ / dataset
    ids_teste_global = EXTERNOS[externo]
    ids_otim_global = [i for i in range(N_ARQUIVOS) if i not in ids_teste_global]

    mod.PASTA_ENTRADA = str(ds)
    mod.ROTULO = ROTULO
    mod.FEATURES_ORIGINAIS = list(FEATURES_ORIGINAIS)
    mod.NOME_BASE = NOME_BASE
    mod.END_BASE = END_BASE
    if detectores:
        mod.DETECTORES_DISPONIVEIS = list(detectores)
    if any(eh_pseudo(d) for d in mod.DETECTORES_DISPONIVEIS):
        instalar_pseudodetectores(mod)
    mod.N_JOBS_DRIFT = 1
    mod.N_JOBS_CATCH24 = 1
    mod.N_JOBS_RF = N_JOBS_RF
    if janela == 0:
        mod.SCENARIOS_TO_RUN = ["baseline", "drift"]
    else:
        mod.WINDOW_MIN = mod.WINDOW_MAX = janela
        mod.WINDOW_STEP = 1
        mod.SCENARIOS_TO_RUN = ["catch24", "catch24_drift"]

    # todos os arquivos, na ordem global; a otimizacao so ve os de otimizacao
    dfs_global = mod.load_dados(str(ds))
    assert len(dfs_global) == N_ARQUIVOS, f"{dataset}: {len(dfs_global)} arquivos"
    dfs_otim = [dfs_global[i] for i in ids_otim_global]
    mod.load_dados = lambda pasta: dfs_otim
    mod.CONJ_TESTE = [[i] for i in range(len(dfs_otim))]

    base = pasta_bloco(ds, t0, externo, nome_bloco(janela))
    pasta_otim, pasta_teste = base / "otimizacao", base / "teste"
    mod.PASTA_SAIDA = str(pasta_otim)

    # modo complementar (--detectores): baseline/catch24 ja testados nao sao refeitos
    if detectores:
        if janela == 0 and (pasta_teste / "baseline" / "resultado_teste.json").exists():
            mod.SCENARIOS_TO_RUN = ["drift"]
        if janela != 0 and (pasta_teste / "catch24" / "resultado_teste.json").exists():
            reaproveitar_catch24(mod, janela)
    build_original = mod.build_dataset_for_item
    instalar_corte(mod, t0)
    build_cortado = mod.build_dataset_for_item

    print(f"\n[JOB] {dataset} | {nome_bloco(janela)} | externo {externo} "
          f"(teste = arquivos {ids_teste_global}) | T0={t0}", flush=True)
    mod.clear_runtime_caches()
    try:
        mod.run_all_experiments()
        mod.clear_runtime_caches()
        dfs_all = dfs_otim + [dfs_global[i] for i in ids_teste_global]
        ids_otim = list(range(len(dfs_otim)))
        ids_teste = list(range(len(dfs_otim), len(dfs_all)))
        avaliar_teste(mod, build_original, build_cortado, dfs_all, ids_otim, ids_teste,
                      pasta_otim, pasta_teste, t0)
    finally:
        mod.clear_runtime_caches()
        mod.remove_temp_dirs(str(pasta_otim))
        mod.remove_temp_dirs(str(pasta_teste))


# ============================================================
# CONSOLIDACAO
# ============================================================
def consolidar(datasets, t0: int):
    from sklearn.metrics import f1_score
    linhas, por_arquivo = [], []
    for ds in datasets:
        base = RAIZ / ds / PASTA_RESULTADOS / f"T0_{t0}"
        for arq in sorted(base.glob("ext*/*/teste/**/resultado_teste.json")):
            r = json.loads(arq.read_text())
            partes = arq.relative_to(base).parts
            externo, bloco = int(partes[0][3:]), partes[1]
            det = r.get("detector_name") or ""
            linha = {"dataset": ds, "bloco": bloco, "cenario": r["scenario"], "detector": det,
                     "externo": externo, "phi_b": r["best_params"].get("phi_b"),
                     "f1_otimizacao": r["f1_otimizacao"], "cobertura": r["cobertura"]}
            linha.update({k: v for k, v in r.items() if k.endswith(("pareada", "completa"))})
            linhas.append(linha)
            pred = pd.read_csv(arq.parent / "predicoes.csv.gz")
            for nome_arq, g in pred.groupby("arquivo"):
                for col in [c for c in pred.columns if c.startswith("pred_")]:
                    gp = g[g["pareada"]]
                    por_arquivo.append({
                        "dataset": ds, "bloco": bloco, "cenario": r["scenario"], "detector": det,
                        "externo": externo, "arquivo": nome_arq, "modelo": col[5:],
                        "f1_pareada": f1_score(gp["y"], gp[col], average="macro"),
                        "f1_completa": f1_score(g["y"], g[col], average="macro"),
                    })
    return pd.DataFrame(linhas), pd.DataFrame(por_arquivo)


# ============================================================
# ORQUESTRACAO
# ============================================================
def job_concluido(ds: str, janela: int, externo: int, t0: int, detectores) -> bool:
    """Concluido quando o cenario base (baseline ou catch24) e cada detector
    pedido (drift/<det> ou catch24_drift/<det>) tem resultado de teste."""
    dets = list(detectores) if detectores else DETECTORES_PADRAO
    teste = pasta_bloco(RAIZ / ds, t0, externo, nome_bloco(janela)) / "teste"
    base, cen = ("baseline", "drift") if janela == 0 else ("catch24", "catch24_drift")
    esperados = [teste / base] + [teste / cen / d for d in dets]
    return all((p / "resultado_teste.json").exists() for p in esperados)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condicoes", nargs="+", default=CONDICOES)
    p.add_argument("--janelas", nargs="+", type=int, default=JANELAS)
    p.add_argument("--detectores", nargs="+", default=DETECTORES)
    p.add_argument("--paralelo", type=int, default=PARALELO)
    p.add_argument("--so-resumo", action="store_true")
    p.add_argument("--job-dataset")
    p.add_argument("--job-janela", type=int)
    p.add_argument("--job-externo", type=int)
    p.add_argument("--job-t0", type=int)
    a = p.parse_args()
    a.detectores = expandir_detectores(a.detectores)

    if a.job_dataset is not None:
        rodar_job(a.job_dataset, a.job_janela, a.job_externo, a.job_t0, a.detectores)
        return

    t0 = max(a.janelas) - 1
    datasets = sorted(d.name for d in RAIZ.iterdir()
                      if d.is_dir() and d.name.split("_")[0] in a.condicoes
                      and any(d.glob(f"{NOME_BASE}0{END_BASE}")))
    if not datasets:
        sys.exit(f"Nenhum dataset das condicoes {a.condicoes} em {RAIZ.resolve()}")
    print(f"Datasets: {datasets}\nJanelas: {a.janelas} | T0 = {t0} | externos: {len(EXTERNOS)}")

    if not a.so_resumo:
        jobs = [(ds, j, e) for ds in datasets for j in [0] + sorted(a.janelas)
                for e in range(len(EXTERNOS))]
        feitos = [jb for jb in jobs if job_concluido(*jb, t0, a.detectores)]
        jobs = [jb for jb in jobs if jb not in feitos]
        if feitos:
            print(f"[SKIP] {len(feitos)} jobs ja concluidos; {len(jobs)} a rodar", flush=True)
        logs = RAIZ / f"logs_aninhado_T0_{t0}"
        logs.mkdir(exist_ok=True)
        pendentes, ativos, falhas = list(jobs), [], []
        while pendentes or ativos:
            while pendentes and len(ativos) < a.paralelo:
                ds, j, e = pendentes.pop(0)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--job-dataset", ds,
                       "--job-janela", str(j), "--job-externo", str(e), "--job-t0", str(t0)]
                if a.detectores:
                    cmd += ["--detectores", *a.detectores]
                nome = f"{ds}_{nome_bloco(j)}_ext{e}"
                log = open(logs / f"{nome}.log", "w")
                ativos.append((nome, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
                print(f"[INICIO] {nome}", flush=True)
            for job in list(ativos):
                nome, proc, log = job
                if proc.poll() is not None:
                    log.close()
                    ativos.remove(job)
                    ok = proc.returncode == 0
                    if not ok:
                        falhas.append(nome)
                    print(f"[{'FIM' if ok else 'ERRO'}] {nome}"
                          f"{'' if ok else f' (ver {logs}/{nome}.log)'}", flush=True)
            time.sleep(5)
        if falhas:
            print(f"[AVISO] jobs com erro: {falhas}")

    df, df_arq = consolidar(datasets, t0)
    df.to_csv(RAIZ / f"resumo_aninhado_T0_{t0}.csv", index=False)
    df_arq.to_csv(RAIZ / f"resumo_aninhado_por_arquivo_T0_{t0}.csv", index=False)
    print(f"\nResumos: resumo_aninhado_T0_{t0}.csv, resumo_aninhado_por_arquivo_T0_{t0}.csv")
    if not df_arq.empty:
        tab = (df_arq.groupby(["dataset", "cenario", "bloco", "detector"])
               .agg(f1_pareada=("f1_pareada", "mean"), dp=("f1_pareada", "std"),
                    f1_completa=("f1_completa", "mean"), n_arquivos=("arquivo", "nunique")))
        cob = df.groupby(["dataset", "cenario", "bloco", "detector"])["cobertura"].mean()
        tab["cobertura"] = cob
        with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_columns", 20,
                               "display.float_format", "{:.3f}".format):
            print(tab)


if __name__ == "__main__":
    main()