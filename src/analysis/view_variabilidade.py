import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from capymoa.drift.detectors import CUSUM
from capymoa.drift.detectors import EWMAChart
from capymoa.drift.detectors import GeometricMovingAverage
from capymoa.drift.detectors import HDDMAverage
from capymoa.drift.detectors import HDDMWeighted
from capymoa.drift.detectors import SEED

from river.drift import ADWIN
from river.drift import PageHinkley
from river.drift import KSWIN


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path("datasetsNew/sintetico/datasets_sinteticos")

N_BASES = 12
FEATURES = ["x1", "x2", "x3", "x4"]
LABEL_COL = "y"

BASE_FILE_TEMPLATE = "base_{i}.csv"
DRIFT_FILE_TEMPLATE = "base_drift_{i}.csv"

HYPER_JSON_NAME = "hiper_detectores_drift.json"
RESULTADOS_KEY = "resultados"

# Janela usada para dizer que há drift real presente após um drift verdadeiro
JANELA_DRIFT_REAL = 50

# Janela centrada usada para calcular variância local
JANELA_VARIANCIA = 50

DETECTOR_ORDER = [
    "ADWIN",
    "PageHinkley",
    "CUSUM",
    "EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    "SEED",
]

OUT_DIR_NAME = "analise_tp_fp_tn_fn_variancia"


# ============================================================
# SCALERS
# ============================================================

def make_scaler(name: str):
    name = (name or "").lower()

    if name in ("standard", "zscore", "z-score"):
        return StandardScaler()

    if name in ("minmax", "min-max"):
        return MinMaxScaler()

    if name in ("robust",):
        return RobustScaler()

    if name in ("passthrough", "none", ""):
        return None

    warnings.warn(f"Scaler desconhecido '{name}', usando None.")
    return None


# ============================================================
# DETECTORES
# ============================================================

def build_detector(detector_name: str, best_params: dict):
    kwargs = {}

    if detector_name == "ADWIN":
        if "adwin_delta" in best_params:
            kwargs["delta"] = best_params["adwin_delta"]

        for k in ("adwin_max_buckets", "adwin_min_window_length", "adwin_grace_period"):
            if k in best_params:
                kwargs[k.replace("adwin_", "")] = best_params[k]

        return ADWIN(**kwargs)

    elif detector_name == "PageHinkley":
        for k in ("ph_delta", "ph_threshold", "ph_alpha", "ph_min_instances"):
            if k in best_params:
                kwargs[k.replace("ph_", "")] = best_params[k]

        return PageHinkley(**kwargs)

    elif detector_name == "CUSUM":
        for k in ("cusum_threshold", "cusum_delta", "cusum_min_instances"):
            if k in best_params:
                if k == "cusum_threshold":
                    kwargs["lambda_"] = best_params[k]
                elif k == "cusum_min_instances":
                    kwargs["min_n_instances"] = best_params[k]
                else:
                    kwargs[k.replace("cusum_", "")] = best_params[k]

        return CUSUM(**kwargs)

    elif detector_name == "EWMAChart":
        for k in ("ewma_lambda", "ewma_min_instances"):
            if k in best_params:
                if k == "ewma_lambda":
                    kwargs["lambda_"] = best_params[k]
                elif k == "ewma_min_instances":
                    kwargs["min_n_instances"] = best_params[k]
                else:
                    kwargs[k.replace("ewma_", "")] = best_params[k]

        return EWMAChart(**kwargs)

    elif detector_name == "GeometricMovingAverage":
        for k in ("gma_threshold", "gma_min_instances", "gma_alpha"):
            if k in best_params:
                if k == "gma_threshold":
                    kwargs["lambda_"] = best_params[k]
                elif k == "gma_min_instances":
                    kwargs["min_n_instances"] = best_params[k]
                else:
                    kwargs[k.replace("gma_", "")] = best_params[k]

        return GeometricMovingAverage(**kwargs)

    elif detector_name == "HDDMAverage":
        for k in ("hddma_delta", "hddma_warn"):
            if k in best_params:
                if k == "hddma_delta":
                    kwargs["drift_confidence"] = best_params[k]
                elif k == "hddma_warn":
                    kwargs["warning_confidence"] = best_params[k]

        return HDDMAverage(**kwargs)

    elif detector_name == "HDDMWeighted":
        for k in ("hddmw_delta", "hddmw_warn", "hddmw_lambda"):
            if k in best_params:
                if k == "hddmw_delta":
                    kwargs["drift_confidence"] = best_params[k]
                elif k == "hddmw_warn":
                    kwargs["warning_confidence"] = best_params[k]
                elif k == "hddmw_lambda":
                    kwargs["lambda_"] = best_params[k]

        return HDDMWeighted(**kwargs)

    elif detector_name == "SEED":
        for k in ("seed_delta", "seed_block_size", "seed_e_prime", "seed_alpha", "seed_compress_term"):
            if k in best_params:
                kk = k.replace("seed_", "")
                if kk == "e_prime":
                    kk = "epsilon_prime"
                kwargs[kk] = best_params[k]

        return SEED(**kwargs)

    elif detector_name == "KSWIN":
        return KSWIN(
            alpha=float(best_params.get("kswin_alpha", 0.0001)),
            window_size=int(best_params.get("kswin_window_size", 100)),
            stat_size=int(best_params.get("kswin_stat_size", 30)),
            seed=42,
        )

    raise ValueError(f"Detector não implementado: {detector_name}")


def feed_detector(detector_name: str, detector, value: float) -> bool:
    if detector_name in ("ADWIN", "PageHinkley", "KSWIN"):
        detector.update(float(value))
        return bool(getattr(detector, "drift_detected", False))

    detector.add_element(float(value))
    return bool(detector.detected_change())


# ============================================================
# REGRAS DO DATASET SINTÉTICO
# ============================================================

def build_true_ids_per_var(
    df: pd.DataFrame,
    df_drift: pd.DataFrame,
    folder_name: str,
) -> dict[str, np.ndarray]:
    """
    Retorna os pontos de drift real por variável.

    Regra:
    - pastas terminadas em _I_: drift em x1 e x4;
    - demais pastas: classe define a variável afetada:
        classe 0 -> x1
        classe 1 -> x2
        classe 2 -> x3
        classe 3 -> x4
    """

    if "is_drift_point" not in df_drift.columns:
        raise ValueError("Arquivo base_drift_i.csv precisa conter a coluna 'is_drift_point'.")

    drift_idx_all = np.flatnonzero(df_drift["is_drift_point"].to_numpy(dtype=int) == 1)

    out = {f: [] for f in FEATURES}
    is_igual = folder_name.endswith("_I_")
    y = df[LABEL_COL].to_numpy(dtype=int)

    for idx in drift_idx_all:
        if is_igual:
            out["x1"].append(int(idx))
            out["x4"].append(int(idx))
        else:
            classe = int(y[idx])
            if 0 <= classe < len(FEATURES):
                out[FEATURES[classe]].append(int(idx))

    return {k: np.array(v, dtype=int) for k, v in out.items()}


def make_drift_real_mask(true_ids: np.ndarray, n: int, janela: int) -> np.ndarray:
    """
    drift_real[i] = 1 se i está dentro de janela amostras após um drift real.
    """

    mask = np.zeros(n, dtype=bool)

    for idx in true_ids:
        inicio = int(idx)
        fim = min(n, int(idx) + janela + 1)
        mask[inicio:fim] = True

    return mask


# ============================================================
# VARIÂNCIA LOCAL
# ============================================================
"""Calcula a variância local de uma série temporal usando uma janela centrada
def variancia_local_array(series: np.ndarray, janela: int = 50) -> np.ndarray:
    n = len(series)
    out = np.zeros(n, dtype=float)

    half = janela // 2

    for i in range(n):
        inicio = max(0, i - half)
        fim = min(n, i + half)
        out[i] = np.var(series[inicio:fim])

    return out
"""
def variancia_local_array(series: np.ndarray, janela: int = 50) -> np.ndarray:
    """
    Variância local causal.
    Usa somente amostras passadas disponíveis até a posição i.
    """

    n = len(series)
    out = np.zeros(n, dtype=float)

    for i in range(n):
        inicio = max(0, i - janela + 1)
        fim = i + 1

        out[i] = np.var(series[inicio:fim])

    return out

# ============================================================
# CLASSIFICAÇÃO TP, FP, TN, FN POR INSTÂNCIA
# ============================================================

def classificar_instancias(alarmes: np.ndarray, drift_real: np.ndarray):
    tp = alarmes & drift_real
    fp = alarmes & ~drift_real
    tn = ~alarmes & ~drift_real
    fn = ~alarmes & drift_real

    return tp, fp, tn, fn


# ============================================================
# LEITURA E NORMALIZAÇÃO
# ============================================================

def load_pair(folder: Path, i: int):
    p_base = folder / BASE_FILE_TEMPLATE.format(i=i)
    p_drift = folder / DRIFT_FILE_TEMPLATE.format(i=i)

    dfb = pd.read_csv(p_base)
    dfd = pd.read_csv(p_drift)

    min_len = min(len(dfb), len(dfd))
    dfb = dfb.iloc[:min_len].reset_index(drop=True)
    dfd = dfd.iloc[:min_len].reset_index(drop=True)

    mask_valid = ~dfb[FEATURES + [LABEL_COL]].isna().any(axis=1)

    dfb = dfb.loc[mask_valid].reset_index(drop=True)
    dfd = dfd.loc[mask_valid].reset_index(drop=True)

    return dfb, dfd


def carregar_bases(folder: Path):
    bases_df = [None] * N_BASES
    drifts_df = [None] * N_BASES
    bases_X = [None] * N_BASES

    for i in range(N_BASES):
        p_base = folder / BASE_FILE_TEMPLATE.format(i=i)
        p_drift = folder / DRIFT_FILE_TEMPLATE.format(i=i)

        if not p_base.exists() or not p_drift.exists():
            warnings.warn(f"[SKIP] faltando base ou drift em {folder.name}, base_{i}")
            continue

        dfb, dfd = load_pair(folder, i)

        bases_df[i] = dfb
        drifts_df[i] = dfd
        bases_X[i] = dfb[FEATURES].to_numpy(dtype=float)

    return bases_df, drifts_df, bases_X


# ============================================================
# ANÁLISE DE UM STREAM
# ============================================================

def analisar_stream(
    dfb: pd.DataFrame,
    dfd: pd.DataFrame,
    X_train_scaler: np.ndarray | None,
    detector_name: str,
    best_params: dict,
    folder_name: str,
    base_id: int,
) -> pd.DataFrame:

    X = dfb[FEATURES].to_numpy(dtype=float)

    scaler = make_scaler(best_params.get("scaler_detectors", "passthrough"))

    if scaler is not None:
        if X_train_scaler is None or len(X_train_scaler) == 0:
            raise ValueError("Scaler exige dados de treino, mas X_train_scaler está vazio.")

        scaler.fit(X_train_scaler)
        X = scaler.transform(X)

    n = len(dfb)

    true_ids_per_var = build_true_ids_per_var(dfb, dfd, folder_name)

    rows = []

    for j, feature in enumerate(FEATURES):
        serie = X[:, j]

        detector = build_detector(detector_name, best_params)

        alarmes = np.zeros(n, dtype=bool)

        for i in range(n):
            alarmes[i] = feed_detector(detector_name, detector, serie[i])

        drift_real = make_drift_real_mask(
            true_ids=true_ids_per_var[feature],
            n=n,
            janela=JANELA_DRIFT_REAL,
        )

        tp, fp, tn, fn = classificar_instancias(alarmes, drift_real)

        var_local = variancia_local_array(serie, janela=JANELA_VARIANCIA)

        for i in range(n):
            if tp[i]:
                grupo = "TP"
            elif fp[i]:
                grupo = "FP"
            elif tn[i]:
                grupo = "TN"
            else:
                grupo = "FN"

            rows.append({
                "pasta": folder_name,
                "base_id": base_id,
                "detector": detector_name,
                "feature": feature,
                "instancia": i,
                "alarme": int(alarmes[i]),
                "drift_real_presente": int(drift_real[i]),
                "grupo": grupo,
                "variancia_local": float(var_local[i]),
                "valor_serie": float(serie[i]),
                "scaler": best_params.get("scaler_detectors", "passthrough"),
            })

    return pd.DataFrame(rows)


# ============================================================
# TESTES ESTATÍSTICOS
# ============================================================

def mannwhitney_safe(a, b, alternative="greater"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)
    return float(stat), float(p_value)


def cliffs_delta(a, b):
    """
    Effect size não paramétrico.
    delta > 0 indica que valores de a tendem a ser maiores que valores de b.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    maior = 0
    menor = 0

    for x in a:
        maior += np.sum(x > b)
        menor += np.sum(x < b)

    return float((maior - menor) / (len(a) * len(b)))


def resumo_grupos(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pasta", "detector", "feature", "grupo"])["variancia_local"]
        .agg(
            n="count",
            media="mean",
            mediana="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def testes_por_contexto(df: pd.DataFrame) -> pd.DataFrame:
    comparacoes = [
        ("FP", "TN", "greater", "FP > TN"),
        ("FP", "TP", "two-sided", "FP != TP"),
        ("FP", "FN", "greater", "FP > FN"),
        ("TN", "FN", "less", "TN < FN"),
    ]

    rows = []

    group_cols = ["pasta", "detector", "feature"]

    for keys, g in df.groupby(group_cols):
        pasta, detector, feature = keys

        for g1, g2, alternative, hipotese in comparacoes:
            a = g.loc[g["grupo"] == g1, "variancia_local"].to_numpy()
            b = g.loc[g["grupo"] == g2, "variancia_local"].to_numpy()

            stat, p_value = mannwhitney_safe(a, b, alternative=alternative)
            delta = cliffs_delta(a, b)

            rows.append({
                "pasta": pasta,
                "detector": detector,
                "feature": feature,
                "comparacao": f"{g1} vs {g2}",
                "hipotese": hipotese,
                "alternative": alternative,
                "n_grupo_1": len(a),
                "n_grupo_2": len(b),
                "media_grupo_1": np.mean(a) if len(a) else np.nan,
                "media_grupo_2": np.mean(b) if len(b) else np.nan,
                "mediana_grupo_1": np.median(a) if len(a) else np.nan,
                "mediana_grupo_2": np.median(b) if len(b) else np.nan,
                "mannwhitney_stat": stat,
                "p_value": p_value,
                "cliffs_delta": delta,
            })

    return pd.DataFrame(rows)


def testes_agregados(df: pd.DataFrame) -> pd.DataFrame:
    comparacoes = [
        ("FP", "TN", "greater", "FP > TN"),
        ("FP", "TP", "two-sided", "FP != TP"),
        ("FP", "FN", "greater", "FP > FN"),
        ("TN", "FN", "less", "TN < FN"),
    ]

    rows = []

    group_cols = ["detector"]

    for detector, g in df.groupby(group_cols):
        if isinstance(detector, tuple):
            detector = detector[0]

        for g1, g2, alternative, hipotese in comparacoes:
            a = g.loc[g["grupo"] == g1, "variancia_local"].to_numpy()
            b = g.loc[g["grupo"] == g2, "variancia_local"].to_numpy()

            stat, p_value = mannwhitney_safe(a, b, alternative=alternative)
            delta = cliffs_delta(a, b)

            rows.append({
                "detector": detector,
                "comparacao": f"{g1} vs {g2}",
                "hipotese": hipotese,
                "alternative": alternative,
                "n_grupo_1": len(a),
                "n_grupo_2": len(b),
                "media_grupo_1": np.mean(a) if len(a) else np.nan,
                "media_grupo_2": np.mean(b) if len(b) else np.nan,
                "mediana_grupo_1": np.median(a) if len(a) else np.nan,
                "mediana_grupo_2": np.median(b) if len(b) else np.nan,
                "mannwhitney_stat": stat,
                "p_value": p_value,
                "cliffs_delta": delta,
            })

    return pd.DataFrame(rows)


# ============================================================
# EXECUÇÃO POR PASTA
# ============================================================

def analisar_folder(folder: Path, detector_names: list[str]) -> pd.DataFrame:
    hyper_path = folder / HYPER_JSON_NAME

    if not hyper_path.exists():
        raise FileNotFoundError(f"Não achei {HYPER_JSON_NAME} em {folder}")

    with open(hyper_path, "r") as f:
        hyper = json.load(f)

    resultados = hyper.get(RESULTADOS_KEY, {})

    bases_df, drifts_df, bases_X = carregar_bases(folder)

    dfs = []

    for detector_name in detector_names:
        if detector_name not in resultados:
            warnings.warn(f"[SKIP] {folder.name}: detector {detector_name} não está no JSON.")
            continue

        best_params = resultados[detector_name].get("best_params", {})
        scaler = make_scaler(best_params.get("scaler_detectors", "passthrough"))

        for i in range(N_BASES):
            dfb = bases_df[i]
            dfd = drifts_df[i]

            if dfb is None or dfd is None:
                continue

            if scaler is None:
                X_train_scaler = None
            else:
                others = [
                    bases_X[k]
                    for k in range(N_BASES)
                    if k != i and bases_X[k] is not None
                ]

                if len(others) == 0:
                    warnings.warn(f"[SKIP] {folder.name} {detector_name} base_{i}: sem treino para scaler.")
                    continue

                X_train_scaler = np.vstack(others)

            try:
                df_stream = analisar_stream(
                    dfb=dfb,
                    dfd=dfd,
                    X_train_scaler=X_train_scaler,
                    detector_name=detector_name,
                    best_params=best_params,
                    folder_name=folder.name,
                    base_id=i,
                )
                dfs.append(df_stream)

            except Exception as e:
                warnings.warn(f"[FAIL] {folder.name} {detector_name} base_{i}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# MAIN
# ============================================================

def main():
    out_dir = BASE_DIR / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])

    if not folders:
        raise FileNotFoundError(f"Nenhuma pasta encontrada em {BASE_DIR}")

    dfs_all = []

    for folder in folders:
        print(f"\n===== Analisando {folder.name} =====", flush=True)
        if folder.name.startswith("I_") or folder.name.startswith("D_"):
            if folder.name.endswith("_I_"):
                warnings.warn(f"[SKIP] {folder.name} parece ser uma pasta de drift real, pulando.")
                continue
            df_folder = analisar_folder(folder, DETECTOR_ORDER)

            if df_folder.empty:
                warnings.warn(f"[VAZIO] Nenhum resultado para {folder.name}")
                continue

            path_folder = out_dir / f"instancias_tp_fp_tn_fn_{folder.name}.csv"
            #df_folder.to_csv(path_folder, index=False)

            dfs_all.append(df_folder)

            print(f"[OK] análise: {path_folder}", flush=True)

    if len(dfs_all) == 0:
        raise RuntimeError("Nenhum resultado foi gerado.")
    print(f"\nAnálise individual completa. ")
    df_all = pd.concat(dfs_all, ignore_index=True)

    # 1) Base completa por instância
    #out_instancias = out_dir / "instancias_tp_fp_tn_fn_todas.csv"
    #df_all.to_csv(out_instancias, index=False)

    # 2) Resumo descritivo por grupo
    df_resumo = resumo_grupos(df_all)
    out_resumo = out_dir / "resumo_variancia_por_grupo.csv"
    df_resumo.to_csv(out_resumo, index=False)

    # 3) Testes por pasta, detector e feature
    df_testes = testes_por_contexto(df_all)
    out_testes = out_dir / "testes_mannwhitney_por_pasta_detector_feature.csv"
    df_testes.to_csv(out_testes, index=False)

    # 4) Testes agregados por detector
    df_testes_agregados = testes_agregados(df_all)
    out_testes_agregados = out_dir / "testes_mannwhitney_agregados_por_detector.csv"
    df_testes_agregados.to_csv(out_testes_agregados, index=False)

    print("\nArquivos salvos:")
    #print(out_instancias)
    print(out_resumo)
    print(out_testes)
    print(out_testes_agregados)

    print("\nResumo dos testes agregados por detector:")
    print(df_testes_agregados)


if __name__ == "__main__":
    main()