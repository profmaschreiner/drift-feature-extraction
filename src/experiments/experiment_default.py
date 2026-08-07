from scoring import gerar_scores_a_partir_de_posicoes
import os
import json
import math
import shutil
import warnings
from typing import Dict, Tuple, List, Set, Optional

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from numpy.lib.stride_tricks import sliding_window_view

import pycatch22

import sys
from capymoa.drift.detectors import CUSUM
from capymoa.drift.detectors import EWMAChart
from capymoa.drift.detectors import GeometricMovingAverage
from capymoa.drift.detectors import HDDMAverage
from capymoa.drift.detectors import HDDMWeighted
from capymoa.drift.detectors import SEED

from river.drift import ADWIN
from river.drift import PageHinkley
from river.drift import KSWIN

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

warnings.simplefilter("once", FutureWarning)

# =========================================================
# CONFIGURAÇÕES GERAIS
# Altere apenas esta seção para adaptar o experimento.
# =========================================================

"""
PASTA_ENTRADA = "data/real/dados_mhealth"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "Activity"
FEATURES_ORIGINAIS = ["alx", "aly", "alz", "glx", "gly", "glz", "arx", "ary", "arz", "grx", "gry", "grz"]


NOME_BASE = "subject_subject"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]



PASTA_ENTRADA = "data/real/rs"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "rotulo"
FEATURES_ORIGINAIS = ["sdp002", "sdt002", "sdt003", "sdu002_1", "sdu002_2"]
NOME_BASE = "base"
END_BASE = "_final.csv"
CONJ_TESTE = [[3], [4], [5], [0, 1], [0,2], [0,7], [0,8], [6, 1], [6, 2], [6,7], [6,8]]



PASTA_ENTRADA = "data/real/gait"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "label"
FEATURES_ORIGINAIS = ["angle_1_1", "angle_1_2", "angle_1_3", "angle_2_1", "angle_2_2", "angle_2_3"]
NOME_BASE = "base"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]



PASTA_ENTRADA = "data/real/pamap2D"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift_ankle")

ROTULO = "activityID"
#FEATURES_ORIGINAIS = ["hand_temperature", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z", "hand_mag_x", "hand_mag_y", "hand_mag_z", "chest_temperature", "chest_acc16_x", "chest_acc16_y", "chest_acc16_z", "chest_acc6_x", "chest_acc6_y", "chest_acc6_z", "chest_gyro_x", "chest_gyro_y", "chest_gyro_z", "chest_mag_x", "chest_mag_y", "chest_mag_z", "ankle_temperature", "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z", "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z", "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z", "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"]
#FEATURES_ORIGINAIS = ["hand_temperature", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z", "hand_mag_x", "hand_mag_y", "hand_mag_z"]
#FEATURES_ORIGINAIS = ["chest_temperature", "chest_acc16_x", "chest_acc16_y", "chest_acc16_z", "chest_acc6_x", "chest_acc6_y", "chest_acc6_z", "chest_gyro_x", "chest_gyro_y", "chest_gyro_z", "chest_mag_x", "chest_mag_y", "chest_mag_z"]
FEATURES_ORIGINAIS = ["ankle_temperature", "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z", "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z", "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z", "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"]

NOME_BASE = "base_"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7]]



PASTA_ENTRADA = "data/real/occ"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "Room_Occupancy_Count"
FEATURES_ORIGINAIS = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp","S1_Light", "S2_Light", "S3_Light", "S4_Light",
                      "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound", "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR", "hour"]
NOME_BASE = "occupancy_dia_"
END_BASE = ".csv"
CONJ_TESTE = [[0,2],[0,3], [0,4], [0,5], [0,6], [1,2],[1,3], [1,4], [1,5], [1,6]]


PASTA_ENTRADA = "data/real/smartphone"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "label"
FEATURES_ORIGINAIS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
NOME_BASE = "user_"
END_BASE = ".csv"
CONJ_TESTE = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17], [18,19,20], [21,22,23], [24,25,26], [27,28,29]]



PASTA_ENTRADA = "data/real/usc_had"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "activity_code"
FEATURES_ORIGINAIS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
NOME_BASE = "usc_had_subject_"
END_BASE = ".csv"
CONJ_TESTE = [[0,1],  [2,3], [4,5],  [6,7], [8,9], [10,11], [12,13]]


"""

PASTA_ENTRADA = "data/synthetic"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "y"
FEATURES_ORIGINAIS = ["x1", "x2", "x3", "x4"]
NOME_BASE = "base_"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]


"""

PASTA_ENTRADA      = "data/real/ward"
PASTA_SAIDA        = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift_left_wrist")
ROTULO             = "label"
#FEATURES_ORIGINAIS = ['l2eft_wrist_acc_x', 'left_wrist_acc_y', 'left_wrist_acc_z', 'left_wrist_gyro_x', 'left_wrist_gyro_y', 'right_wrist_acc_x', 'right_wrist_acc_y', 'right_wrist_acc_z', 'right_wrist_gyro_x', 'right_wrist_gyro_y', 'waist_acc_x', 'waist_acc_y', 'waist_acc_z', 'waist_gyro_x', 'waist_gyro_y', 'left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z', 'left_ankle_gyro_x', 'left_ankle_gyro_y', 'right_ankle_acc_x', 'right_ankle_acc_y', 'right_ankle_acc_z', 'right_ankle_gyro_x', 'right_ankle_gyro_y']
#FEATURES_ORIGINAIS = ['left_wrist_acc_x', 'left_wrist_acc_y', 'left_wrist_acc_z',  'left_wrist_gyro_x', 'left_wrist_gyro_y',  'right_wrist_acc_x', 'right_wrist_acc_y', 'right_wrist_acc_z', 'right_wrist_gyro_x', 'right_wrist_gyro_y']

#FEATURES_ORIGINAIS = [ 'waist_acc_x', 'waist_acc_y', 'waist_acc_z', 'waist_gyro_x', 'waist_gyro_y']

#FEATURES_ORIGINAIS = ['left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z',   'left_ankle_gyro_x', 'left_ankle_gyro_y', 'right_ankle_acc_x', 'right_ankle_acc_y', 'right_ankle_acc_z',  'right_ankle_gyro_x', 'right_ankle_gyro_y']

# ward_left_wrist
FEATURES_ORIGINAIS = ['left_wrist_acc_x', 'left_wrist_acc_y', 'left_wrist_acc_z',     'left_wrist_gyro_x', 'left_wrist_gyro_y']

# ward_right_wrist
#FEATURES_ORIGINAIS = ['right_wrist_acc_x', 'right_wrist_acc_y', 'right_wrist_acc_z',     'right_wrist_gyro_x', 'right_wrist_gyro_y']

# ward_waist - ja executado
#FEATURES_ORIGINAIS = ['waist_acc_x', 'waist_acc_y', 'waist_acc_z',     'waist_gyro_x', 'waist_gyro_y']

# ward_left_ankle
#FEATURES_ORIGINAIS = ['left_ankle_acc_x', 'left_ankle_acc_y', 'left_ankle_acc_z',     'left_ankle_gyro_x', 'left_ankle_gyro_y']

# ward_right_ankle
#FEATURES_ORIGINAIS = ['right_ankle_acc_x', 'right_ankle_acc_y', 'right_ankle_acc_z',     'right_ankle_gyro_x', 'right_ankle_gyro_y']






PASTA_ENTRADA      = "data/real/sw_sp/geotec_sp"
PASTA_SAIDA        = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")
ROTULO             = "rotulo"
FEATURES_ORIGINAIS = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
NOME_BASE          = "sujeito_"
END_BASE           = ".csv"
CONJ_TESTE = [
    [ 0,  1],  # s01 + s02
    [ 2,  3],  # s03 + s04
    [ 4,  5],  # s05 + s06
    [ 6,  7],  # s07 + s08
    [ 8,  9],  # s09 + s10
    [10, 11],  # s11 + s12
    [12, 13],  # s13 + s14
    [14, 15],  # s15 + s16
    [16, 17],  # s17 + s18
    [18, 19],  # s19 + s20
    [20, 21, 22],  # s21 + s22 + s23
]



# geotec_sw:
PASTA_ENTRADA      = "data/real/sw_sp/geotec_sw"
PASTA_SAIDA        = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")
ROTULO             = "rotulo"
FEATURES_ORIGINAIS = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
NOME_BASE          = "sujeito_"
END_BASE           = ".csv"
CONJ_TESTE = [
    [ 0,  1],  # s01 + s02
    [ 2,  3],  # s03 + s04
    [ 4,  5],  # s05 + s06
    [ 6,  7],  # s07 + s08
    [ 8,  9],  # s09 + s10
    [10, 11],  # s11 + s12
    [12, 13],  # s13 + s14
    [14, 15],  # s15 + s16
    [16, 17],  # s17 + s18
    [18, 19],  # s19 + s20
    [20, 21, 22],  # s21 + s22 + s23
]

"""





DATASET_SINTETICO = True

N_TRIALS = 30
semente= 42
N_JOBS_CATCH24 = 4
N_JOBS_DRIFT = 4
N_JOBS_OPT = 1
N_JOBS_RF = -1
NAN_THRESHOLD = 0.30
WINDOW_MIN = 20
WINDOW_MAX = 130
WINDOW_STEP = 20
USE_CATCH24 = True
SHORT_NAMES = True

REUSAR_JANELA_CATCH24_OTIMIZADA = True
MANTER_APENAS_MELHOR_JANELA_CATCH24_NO_CACHE = True

# Cenários avaliados:
# baseline = somente features originais
# catch24 = somente catch24
# drift = originais + scores de drift
# catch24_drift = originais alinhadas + scores de drift alinhados + catch24
SCENARIOS_TO_RUN = ["catch24_drift"] #["baseline", "drift", "catch24",  "catch24_drift"]  

DETECTORES_DISPONIVEIS = [
    "ADWIN",
    "PageHinkley",
    "KSWIN",
    "CUSUM",
    "EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    "SEED",
]
SCALERS_DISPONIVEIS = ["standard", "minmax", "robust"]

VALORES_AUSENTES = ["?", "N/A", ""]
CATCH_CACHE_DIRNAME = "catch24_cache_tmp"
DRIFT_CACHE_DIRNAME = "drift_cache_tmp"


# =========================================================
# 
# =========================================================
def to_python_types(obj):
    if isinstance(obj, dict):
        return {to_python_types(k): to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_or_empty_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def list_base_files(pasta: str) -> List[str]:
    arquivos = []
    for nome in os.listdir(pasta):
        if not nome.endswith(END_BASE):
            continue
        if not nome.startswith(NOME_BASE):
            continue
        sufixo = nome[len(NOME_BASE):-len(END_BASE)]
        if sufixo.isdigit():
            arquivos.append(nome)
    arquivos = sorted(arquivos, key=lambda x: int(x[len(NOME_BASE):-len(END_BASE)]))
    if not arquivos:
        raise ValueError(f"Nenhum arquivo base encontrado em: {pasta}")
    return arquivos


def build_combinacoes(n_bases: int) -> List[Dict[str, List[int]]]:
    combinacoes = []
    todos = list(range(n_bases))
    for teste in CONJ_TESTE:
        treino = [x for x in todos if x not in teste]
        combinacoes.append({"treinamento": treino, "teste": teste})
    return combinacoes


def build_scaler_from_name(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Scaler desconhecido: {name}")


def suggest_scaler_model(trial):
    scaler_name = trial.suggest_categorical("scaler_model", ["standard", "minmax", "robust"])
    return scaler_name


# =========================================================
# LEITURA
# =========================================================
def load_dados(pasta: str):
    arquivos = list_base_files(pasta)
    dfs_info = []
    for nome in arquivos:
        caminho = os.path.join(pasta, nome)
        df = pd.read_csv(caminho)
        df = df.replace(VALORES_AUSENTES, np.nan)
        df = df.dropna().reset_index(drop=True)

        faltando = set(FEATURES_ORIGINAIS + [ROTULO]) - set(df.columns)
        if faltando:
            raise ValueError(f"Arquivo {nome} sem colunas obrigatórias: {faltando}")

        df = df[FEATURES_ORIGINAIS + [ROTULO]].copy()
        dfs_info.append({"nome_arquivo": nome, "df": df})

    print(f"Foram carregados {len(dfs_info)} blocos.")
    return dfs_info


# =========================================================
# CACHE CATCH24
# =========================================================
_CATCH_CACHE: Dict[Tuple[str, int, int, int, int], Tuple[pd.DataFrame, np.ndarray]] = {}
_COLS_CACHE: Dict[Tuple[str, int, int, int, int], List[str]] = {}


def get_catch_cache_dir(pasta_saida: str) -> str:
    d = os.path.join(pasta_saida, CATCH_CACHE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def get_catch_cache_paths(pasta_saida: str, window_size: int, base_id: int, use_catch24: bool, short_names: bool):
    d = get_catch_cache_dir(pasta_saida)
    stem = f"w{window_size:04d}_b{base_id:03d}_c24{int(use_catch24)}_sn{int(short_names)}"
    return {
        "X": os.path.join(d, f"{stem}_X.parquet"),
        "y": os.path.join(d, f"{stem}_y.npy"),
        "cols": os.path.join(d, f"{stem}_cols.json"),
    }


def purge_catch24_cache_except_window(pasta_saida: str, keep_window_size: int, use_catch24: bool = True, short_names: bool = True):
    d = get_catch_cache_dir(pasta_saida)
    keep_prefix = f"w{keep_window_size:04d}_"
    suffix = f"_c24{int(use_catch24)}_sn{int(short_names)}"
    removed = 0
    for fname in os.listdir(d):
        if not fname.startswith("w") or suffix not in fname:
            continue
        if not fname.startswith(keep_prefix):
            try:
                os.remove(os.path.join(d, fname))
                removed += 1
            except FileNotFoundError:
                pass
    print(f"[INFO] Cache catch24: mantida somente janela {keep_window_size}. Arquivos removidos: {removed}", flush=True)


def get_window_sizes_to_evaluate() -> List[int]:
    """Retorna exatamente os tamanhos de janela que o Optuna pode sugerir."""
    return list(range(WINDOW_MIN, WINDOW_MAX + 1, WINDOW_STEP))


def is_valid_catch24_cache(paths: Dict[str, str]) -> bool:
    """Verifica se os três arquivos do cache existem e conseguem ser lidos."""
    if not (os.path.exists(paths["X"]) and os.path.exists(paths["y"]) and os.path.exists(paths["cols"])):
        return False
    try:
        _ = pd.read_parquet(paths["X"])
        _ = np.load(paths["y"], allow_pickle=False)
        _ = load_cols_json(paths["cols"])
        return True
    except Exception:
        for p in paths.values():
            try:
                if os.path.exists(p):
                    os.remove(p)
            except FileNotFoundError:
                pass
        return False


def precompute_all_catch24_windows(dfs_info, pasta_saida: str):
    """Pré-calcula o catch24 para todas as janelas discretas antes do Optuna."""
    windows = get_window_sizes_to_evaluate()
    print(f"\n===== Pré-cálculo de catch24: janelas={windows} =====", flush=True)

    tasks = []
    for window_size in windows:
        for base_id, item in enumerate(dfs_info):
            paths = get_catch_cache_paths(pasta_saida, window_size, base_id, USE_CATCH24, SHORT_NAMES)
            if is_valid_catch24_cache(paths):
                continue
            tasks.append((item, base_id, window_size))

    if not tasks:
        print("[INFO] Todas as janelas catch24 já existem no cache.", flush=True)
        return

    print(f"[INFO] Tarefas catch24 pendentes: {len(tasks)}", flush=True)

    if Parallel is None or delayed is None or N_JOBS_CATCH24 <= 1:
        for item, base_id, window_size in tasks:
            get_cached_catch24_from_item(
                item=item,
                base_id=base_id,
                window_size=window_size,
                pasta_saida=pasta_saida,
                n_jobs_catch24=N_JOBS_CATCH24,
                use_catch24=USE_CATCH24,
                short_names=SHORT_NAMES,
            )
    else:
        # Paraleliza por (base, janela) e evita paralelismo interno aninhado.
        Parallel(n_jobs=N_JOBS_CATCH24, prefer="processes", batch_size=1)(
            delayed(get_cached_catch24_from_item)(
                item,
                base_id,
                window_size,
                pasta_saida,
                1,
                USE_CATCH24,
                SHORT_NAMES,
            )
            for item, base_id, window_size in tasks
        )

    print("[INFO] Pré-cálculo de catch24 concluído.", flush=True)


def precompute_best_catch24_window_for_all_bases(dfs_info, pasta_saida: str, window_size: int):
    print(f"[INFO] Pré-cálculo/reuso do catch24 para melhor janela: {window_size}", flush=True)
    if Parallel is None or delayed is None or N_JOBS_CATCH24 <= 1:
        for base_id, item in enumerate(dfs_info):
            get_cached_catch24_from_item(item, base_id, window_size, pasta_saida, N_JOBS_CATCH24, USE_CATCH24, SHORT_NAMES)
    else:
        Parallel(n_jobs=min(N_JOBS_CATCH24, len(dfs_info)), prefer="processes")(
            delayed(get_cached_catch24_from_item)(item, base_id, window_size, pasta_saida, 1, USE_CATCH24, SHORT_NAMES)
            for base_id, item in enumerate(dfs_info)
        )


def save_cols_json(path: str, cols: List[str]):
    with open(path, "w") as f:
        json.dump(cols, f)


def load_cols_json(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def catch_for_1d(ts_1d: np.ndarray, use_catch24: bool, short_names: bool) -> Tuple[List[str], List[float]]:
    res = pycatch22.catch22_all(ts_1d, catch24=use_catch24, short_names=short_names)
    if short_names and ("short_names" in res):
        names = res["short_names"]
    else:
        names = res["names"]
    vals = res["values"]
    return list(names), list(vals)


def extract_catch24_features_from_df(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: List[str],
    y_col: str,
    n_jobs: int = 1,
    use_catch24: bool = True,
    short_names: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    data = df[feature_cols].to_numpy(dtype=float, copy=False)
    y = df[y_col].to_numpy(copy=False)

    n, p = data.shape
    if n < window_size:
        return pd.DataFrame(), np.array([], dtype=y.dtype)

    num_windows = n - window_size + 1
    y_win = y[window_size - 1:]
    w = sliding_window_view(data, window_shape=(window_size, p))[:, 0, :, :]

    cols = []
    for j, col in enumerate(feature_cols):
        names, _ = catch_for_1d(w[0, :, j], use_catch24=use_catch24, short_names=short_names)
        cols.extend([f"{col}__{nm}" for nm in names])

    def one_window(i: int) -> np.ndarray:
        feat_vec = []
        for j, _ in enumerate(feature_cols):
            _, vals = catch_for_1d(w[i, :, j], use_catch24=use_catch24, short_names=short_names)
            feat_vec.extend(vals)
        return np.asarray(feat_vec, dtype=np.float64)

    if (n_jobs is None) or (n_jobs <= 1) or (Parallel is None):
        X_mat = np.vstack([one_window(i) for i in range(num_windows)])
    else:
        X_list = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64)(
            delayed(one_window)(i) for i in range(num_windows)
        )
        X_mat = np.vstack(X_list)

    X = pd.DataFrame(X_mat, columns=cols)
    return X, y_win


def get_cached_catch24(
    dfs_info,
    base_id: int,
    window_size: int,
    pasta_saida: str,
    n_jobs_catch24: int = 1,
    use_catch24: bool = True,
    short_names: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    key = (pasta_saida, window_size, base_id, int(use_catch24), int(short_names))
    if key in _CATCH_CACHE:
        return _CATCH_CACHE[key]

    paths = get_catch_cache_paths(pasta_saida, window_size, base_id, use_catch24, short_names)
    if is_valid_catch24_cache(paths):
        X = pd.read_parquet(paths["X"])
        y = np.load(paths["y"], allow_pickle=False)
        _CATCH_CACHE[key] = (X, y)
        return X, y

    X, y = extract_catch24_features_from_df(
        df=dfs_info[base_id]["df"],
        window_size=window_size,
        feature_cols=FEATURES_ORIGINAIS,
        y_col=ROTULO,
        n_jobs=n_jobs_catch24,
        use_catch24=use_catch24,
        short_names=short_names,
    )

    cols = list(X.columns) if not X.empty else []
    X.to_parquet(paths["X"], index=False)
    np.save(paths["y"], y)
    save_cols_json(paths["cols"], cols)

    _CATCH_CACHE[key] = (X, y)
    return X, y


def get_cached_catch24_from_item(
    item,
    base_id: int,
    window_size: int,
    pasta_saida: str,
    n_jobs_catch24: int = 1,
    use_catch24: bool = True,
    short_names: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    key = (pasta_saida, window_size, base_id, int(use_catch24), int(short_names))
    if key in _CATCH_CACHE:
        return _CATCH_CACHE[key]

    paths = get_catch_cache_paths(pasta_saida, window_size, base_id, use_catch24, short_names)
    if is_valid_catch24_cache(paths):
        X = pd.read_parquet(paths["X"])
        y = np.load(paths["y"], allow_pickle=False)
        _CATCH_CACHE[key] = (X, y)
        return X, y

    X, y = extract_catch24_features_from_df(
        df=item["df"],
        window_size=window_size,
        feature_cols=FEATURES_ORIGINAIS,
        y_col=ROTULO,
        n_jobs=n_jobs_catch24,
        use_catch24=use_catch24,
        short_names=short_names,
    )

    cols = list(X.columns) if not X.empty else []
    X.to_parquet(paths["X"], index=False)
    np.save(paths["y"], y)
    save_cols_json(paths["cols"], cols)

    _CATCH_CACHE[key] = (X, y)
    return X, y


def get_cached_columns(
    dfs_info,
    base_id: int,
    window_size: int,
    pasta_saida: str,
    n_jobs_catch24: int = 1,
    use_catch24: bool = True,
    short_names: bool = True,
) -> List[str]:
    key = (pasta_saida, window_size, base_id, int(use_catch24), int(short_names))
    if key in _COLS_CACHE:
        return _COLS_CACHE[key]

    paths = get_catch_cache_paths(pasta_saida, window_size, base_id, use_catch24, short_names)
    if os.path.exists(paths["cols"]):
        cols = load_cols_json(paths["cols"])
        _COLS_CACHE[key] = cols
        return cols

    _ = get_cached_catch24(
        dfs_info=dfs_info,
        base_id=base_id,
        window_size=window_size,
        pasta_saida=pasta_saida,
        n_jobs_catch24=n_jobs_catch24,
        use_catch24=use_catch24,
        short_names=short_names,
    )
    cols = load_cols_json(paths["cols"]) if os.path.exists(paths["cols"]) else []
    _COLS_CACHE[key] = cols
    return cols


def get_common_columns_for_bases(
    dfs_info,
    base_ids: List[int],
    window_size: int,
    pasta_saida: str,
    n_jobs_catch24: int = 1,
    use_catch24: bool = True,
    short_names: bool = True,
) -> List[str]:
    common: Optional[Set[str]] = None
    for base_id in base_ids:
        cols_i = set(get_cached_columns(
            dfs_info=dfs_info,
            base_id=base_id,
            window_size=window_size,
            pasta_saida=pasta_saida,
            n_jobs_catch24=n_jobs_catch24,
            use_catch24=use_catch24,
            short_names=short_names,
        ))
        if not cols_i:
            return []
        common = cols_i if common is None else (common & cols_i)
        if not common:
            return []
    return sorted(common)


# =========================================================
# CACHE DRIFT
# =========================================================
def get_drift_cache_dir(pasta_saida: str) -> str:
    d = os.path.join(pasta_saida, DRIFT_CACHE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def build_default_detector_list(detector_name: str, num_features: int):
    if detector_name == "ADWIN":
        return [ADWIN() for _ in range(num_features)]
    if detector_name == "PageHinkley":
        return [PageHinkley() for _ in range(num_features)]
    if detector_name == "KSWIN":
        return [KSWIN() for _ in range(num_features)]
    if detector_name == "CUSUM":
        return [CUSUM() for _ in range(num_features)]
    if detector_name == "EWMAChart":
        return [EWMAChart() for _ in range(num_features)]
    if detector_name == "GeometricMovingAverage":
        return [GeometricMovingAverage() for _ in range(num_features)]
    if detector_name == "HDDMAverage":
        return [HDDMAverage() for _ in range(num_features)]
    if detector_name == "HDDMWeighted":
        return [HDDMWeighted() for _ in range(num_features)]
    if detector_name == "SEED":
        return [SEED() for _ in range(num_features)]
    raise ValueError(f"Detector desconhecido: {detector_name}")


def update_detector_and_get_flag(detector, value):
    if hasattr(detector, "update"):
        detector.update(float(value))
        if hasattr(detector, "drift_detected"):
            return bool(detector.drift_detected)
        if hasattr(detector, "change_detected"):
            return bool(detector.change_detected)
        return False
    if hasattr(detector, "add_element"):
        detector.add_element(float(value))
        if hasattr(detector, "detected_change"):
            return bool(detector.detected_change())
        return False
    raise ValueError(f"Detector não suportado: {type(detector)}")


def detectar_posicoes_drift_por_feature(np_scaled, detector_name: str) -> pd.DataFrame:
    detectors = build_default_detector_list(detector_name, len(FEATURES_ORIGINAIS))
    data_np = np_scaled

    rows = []
    for linha in range(data_np.shape[0]):
        for j, col_name in enumerate(FEATURES_ORIGINAIS):
            if update_detector_and_get_flag(detectors[j], data_np[linha, j]):
                rows.append({"feature": col_name, "drift_index": int(linha)})
    return pd.DataFrame(rows, columns=["feature", "drift_index"])


def get_drift_csv_path(base_dir, detector_name, scaler_name, id_fold, split_name, nome_arquivo):
    nome_base = os.path.splitext(nome_arquivo)[0]
    pasta = os.path.join(base_dir, detector_name, scaler_name, f"fold_{id_fold:02d}", split_name)
    os.makedirs(pasta, exist_ok=True)
    return os.path.join(pasta, f"{nome_base}_drifts.csv")


def process_single_base_drift(
    item,
    detector_name: str,
    scaler_detectors,
    scaler_name: str,
    id_fold: int,
    split_name: str,
    base_dir: str,
):
    csv_path = get_drift_csv_path(
        base_dir=base_dir,
        detector_name=detector_name,
        scaler_name=scaler_name,
        id_fold=id_fold,
        split_name=split_name,
        nome_arquivo=item["nome_arquivo"],
    )
    if os.path.exists(csv_path):
        return

    df_original = item["df"]
    np_scaled = scaler_detectors.transform(df_original[FEATURES_ORIGINAIS])
    df_drifts = detectar_posicoes_drift_por_feature(np_scaled, detector_name)
    df_drifts.to_csv(csv_path, index=False)


def precompute_drift_files(dfs_info, combinacoes, pasta_saida: str):
    base_dir = get_drift_cache_dir(pasta_saida)
    for detector_name in DETECTORES_DISPONIVEIS:
        print(f"\n===== Pré-cálculo de drifts: {detector_name} =====", flush=True)
        for id_fold, combinacao in enumerate(combinacoes):
            treino_info = [dfs_info[idx] for idx in combinacao["treinamento"]]
            teste_info = [dfs_info[idx] for idx in combinacao["teste"]]
            treino_df_concat = pd.concat([x["df"] for x in treino_info], ignore_index=True)

            for scaler_name in SCALERS_DISPONIVEIS:
                scaler_detectors = build_scaler_from_name(scaler_name)
                scaler_detectors.fit(treino_df_concat[FEATURES_ORIGINAIS])

                for split_name, infos in [("train", treino_info), ("test", teste_info)]:
                    if Parallel is None or delayed is None or N_JOBS_DRIFT <= 1:
                        for item in infos:
                            process_single_base_drift(
                                item=item,
                                detector_name=detector_name,
                                scaler_detectors=scaler_detectors,
                                scaler_name=scaler_name,
                                id_fold=id_fold,
                                split_name=split_name,
                                base_dir=base_dir,
                            )
                    else:
                        Parallel(n_jobs=N_JOBS_DRIFT, prefer="processes")(
                            delayed(process_single_base_drift)(
                                item=item,
                                detector_name=detector_name,
                                scaler_detectors=scaler_detectors,
                                scaler_name=scaler_name,
                                id_fold=id_fold,
                                split_name=split_name,
                                base_dir=base_dir,
                            )
                            for item in infos
                        )


def carregar_posicoes_drift(csv_path):
    drift_map = {col: np.array([], dtype=np.int64) for col in FEATURES_ORIGINAIS}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo de drifts não encontrado: {csv_path}")

    df_drifts = pd.read_csv(csv_path)
    if df_drifts.empty:
        return drift_map

    for col in FEATURES_ORIGINAIS:
        vals = df_drifts.loc[df_drifts["feature"] == col, "drift_index"].to_numpy(dtype=np.int64, copy=False)
        drift_map[col] = vals
    return drift_map





def get_drift_enriched_df(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida: str):
    base_dir = get_drift_cache_dir(pasta_saida)
    df_original = item["df"].copy()
    csv_path = get_drift_csv_path(
        base_dir=base_dir,
        detector_name=detector_name,
        scaler_name=scaler_name,
        id_fold=id_fold,
        split_name=split_name,
        nome_arquivo=item["nome_arquivo"],
    )
    drift_map = carregar_posicoes_drift(csv_path)
    num_rows = len(df_original)

    for col in FEATURES_ORIGINAIS:
        df_original[f"score_{col}"] = gerar_scores_a_partir_de_posicoes(
            num_rows=num_rows,
            drift_indices=drift_map[col],
            phi_b=phi_b,
        )
    return df_original


# =========================================================
# MONTAGEM DAS MATRIZES
# =========================================================
def build_dataset_for_item(
    item,
    base_id: int,
    scenario_name: str,
    window_size: Optional[int],
    detector_name: Optional[str],
    scaler_name: Optional[str],
    phi_b: Optional[float],
    id_fold: int,
    split_name: str,
    pasta_saida: str,
):
    df_base = item["df"]

    if scenario_name == "baseline":
        X = df_base[FEATURES_ORIGINAIS].to_numpy(dtype=np.float32, copy=False)
        y = df_base[ROTULO].to_numpy(copy=False)
        cols = FEATURES_ORIGINAIS.copy()
        return pd.DataFrame(X, columns=cols), y

    if scenario_name == "drift":
        df_drift = get_drift_enriched_df(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida)
        cols = FEATURES_ORIGINAIS + [f"score_{c}" for c in FEATURES_ORIGINAIS]
        X = df_drift[cols].to_numpy(dtype=np.float32, copy=False)
        y = df_drift[ROTULO].to_numpy(copy=False)
        return pd.DataFrame(X, columns=cols), y

    if scenario_name in {"catch24", "catch24_drift"}:
        X_catch, y_catch = get_cached_catch24_from_item(
            item=item,
            base_id=base_id,
            window_size=window_size,
            pasta_saida=pasta_saida,
            n_jobs_catch24=N_JOBS_CATCH24,
            use_catch24=USE_CATCH24,
            short_names=SHORT_NAMES,
        )

        if scenario_name == "catch24":
            return X_catch.reset_index(drop=True), y_catch

        df_drift = get_drift_enriched_df(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida)

        # Catch24 é calculado por base individual.
        # Ao combinar com as demais variáveis do experimento, descartamos
        # as primeiras (window_size - 1) linhas, que não possuem janela completa.
        start = window_size - 1
        drift_cols = FEATURES_ORIGINAIS + [f"score_{c}" for c in FEATURES_ORIGINAIS]
        X_drift = df_drift.iloc[start:].reset_index(drop=True)[drift_cols]

        if len(X_drift) != len(X_catch):
            min_len = min(len(X_drift), len(X_catch))
            X_drift = X_drift.iloc[:min_len].reset_index(drop=True)
            X_catch = X_catch.iloc[:min_len].reset_index(drop=True)
            y_catch = y_catch[:min_len]

        X = pd.concat([X_drift.reset_index(drop=True), X_catch.reset_index(drop=True)], axis=1)
        return X, y_catch

    raise ValueError(f"Cenário desconhecido: {scenario_name}")


# =========================================================
# POLÍTICA DE COLUNAS
# =========================================================
def apply_column_policy(X_train: pd.DataFrame, X_test: pd.DataFrame, nan_threshold: float = 0.30):
    common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
    if not common_cols:
        return pd.DataFrame(), pd.DataFrame(), {"n_features_common": 0, "n_features_after_nan": 0}, []

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_features_common = int(X_train.shape[1])
    nan_frac = X_train.isna().mean(axis=0)
    keep_cols = nan_frac[nan_frac <= nan_threshold].index.tolist()

    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    stats = {
        "n_features_common": n_features_common,
        "n_features_after_nan": int(X_train.shape[1]),
    }
    return X_train, X_test, stats, keep_cols


# =========================================================
# MODELO
# =========================================================
def build_models_from_trial(trial) -> Dict[str, Pipeline]:
    scaler_name = suggest_scaler_model(trial)
    scaler = build_scaler_from_name(scaler_name)
    models = {}
    models["RF"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var0", VarianceThreshold(threshold=0.0)),
        ("scaler", scaler),
        ("clf", RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
            max_depth=trial.suggest_int("rf_max_depth", 2, 5),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("rf_bootstrap", [True, False]),
            n_jobs=N_JOBS_RF,
            random_state=42,
        )),
    ])
    return models


# =========================================================
# OBJECTIVE
# =========================================================
def make_objective(dfs_info, combinacoes, pasta_saida: str, scenario_name: str, detector_name: Optional[str] = None, fixed_catch24_window_size: Optional[int] = None):
    all_labels = np.unique(np.concatenate([item["df"][ROTULO].values for item in dfs_info]))

    def objective(trial: optuna.Trial) -> float:
        window_size = None
        phi_b = None
        scaler_name = None

        if scenario_name == "catch24":
            window_size = trial.suggest_int("window_size", WINDOW_MIN, WINDOW_MAX, step=WINDOW_STEP)
        elif scenario_name == "catch24_drift":
            if REUSAR_JANELA_CATCH24_OTIMIZADA:
                if fixed_catch24_window_size is None:
                    raise ValueError("catch24_drift requer a melhor janela do cenário catch24.")
                window_size = int(fixed_catch24_window_size)
                trial.set_user_attr("fixed_catch24_window_size", window_size)
            else:
                window_size = trial.suggest_int("window_size", WINDOW_MIN, WINDOW_MAX, step=WINDOW_STEP)

        if scenario_name in {"drift", "catch24_drift"}:
            phi_b = trial.suggest_float("phi_b", 1e-6, 1e-1, log=True)
            scaler_name = trial.suggest_categorical("scaler_detectors", SCALERS_DISPONIVEIS)

        models = build_models_from_trial(trial)
        model_names = list(models.keys())

        macro_f1_rows = []
        acc_rows = []
        fold_count_rows = []

        conf_sum_by_model = {m: np.zeros((len(all_labels), len(all_labels)), dtype=np.int64) for m in model_names}
        f1_per_class_sum_by_model = {m: np.zeros(len(all_labels), dtype=float) for m in model_names}
        f1_per_class_count_by_model = {m: 0 for m in model_names}

        used_cols_union: Set[str] = set()
        used_cols_after_nan_union: Set[str] = set()

        valid_folds = 0
        partial_scores = []

        for fold_id, combinacao in enumerate(combinacoes):
            train_ids = combinacao["treinamento"]
            test_ids = combinacao["teste"]

            X_tr_list, y_tr_list = [], []
            X_te_list, y_te_list = [], []

            for idx in train_ids:
                X_i, y_i = build_dataset_for_item(
                    item=dfs_info[idx],
                    base_id=idx,
                    scenario_name=scenario_name,
                    window_size=window_size,
                    detector_name=detector_name,
                    scaler_name=scaler_name,
                    phi_b=phi_b,
                    id_fold=fold_id,
                    split_name="train",
                    pasta_saida=pasta_saida,
                )
                if X_i.empty or len(y_i) == 0:
                    continue
                X_tr_list.append(X_i)
                y_tr_list.append(y_i)

            for idx in test_ids:
                X_i, y_i = build_dataset_for_item(
                    item=dfs_info[idx],
                    base_id=idx,
                    scenario_name=scenario_name,
                    window_size=window_size,
                    detector_name=detector_name,
                    scaler_name=scaler_name,
                    phi_b=phi_b,
                    id_fold=fold_id,
                    split_name="test",
                    pasta_saida=pasta_saida,
                )
                if X_i.empty or len(y_i) == 0:
                    continue
                X_te_list.append(X_i)
                y_te_list.append(y_i)

            if not X_tr_list or not X_te_list:
                continue

            X_train = pd.concat(X_tr_list, ignore_index=True)
            y_train = np.concatenate(y_tr_list)
            X_test = pd.concat(X_te_list, ignore_index=True)
            y_test = np.concatenate(y_te_list)

            X_train, X_test, stats_cols, keep_cols = apply_column_policy(
                X_train=X_train,
                X_test=X_test,
                nan_threshold=NAN_THRESHOLD,
            )
            if X_train.shape[1] == 0 or X_test.shape[1] == 0:
                continue

            used_cols_union.update(list(set(X_train.columns) | set(X_test.columns)))
            used_cols_after_nan_union.update(keep_cols)

            fold_count_rows.append({
                "fold": int(fold_id),
                "test_bases": json.dumps(test_ids),
                "n_train_samples": int(len(y_train)),
                "n_test_samples": int(len(y_test)),
                "n_features_common_train": int(stats_cols["n_features_common"]),
                "n_features_after_nan_train": int(stats_cols["n_features_after_nan"]),
            })

            labels_fold = np.unique(y_test)
            row_f1 = {"fold": int(fold_id), "test_bases": json.dumps(test_ids)}
            row_acc = {"fold": int(fold_id), "test_bases": json.dumps(test_ids)}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1_macro = f1_score(y_test, y_pred, labels=labels_fold, average="macro")
                acc = accuracy_score(y_test, y_pred)

                row_f1[name] = float(f1_macro)
                row_acc[name] = float(acc)

                f1_cls = f1_score(y_test, y_pred, labels=all_labels, average=None, zero_division=0)
                f1_per_class_sum_by_model[name] += f1_cls
                f1_per_class_count_by_model[name] += 1

                cm = confusion_matrix(y_test, y_pred, labels=all_labels)
                conf_sum_by_model[name] += cm

            macro_f1_rows.append(row_f1)
            acc_rows.append(row_acc)
            valid_folds += 1

            partial_val = float(np.mean([row_f1[m] for m in model_names if m in row_f1]))
            partial_scores.append(partial_val)

            if valid_folds >= 3:
                trial.report(float(np.mean(partial_scores)), step=valid_folds)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if valid_folds == 0:
            return 0.0

        df_macro_f1 = pd.DataFrame(macro_f1_rows)
        df_acc = pd.DataFrame(acc_rows)
        df_fold_counts = pd.DataFrame(fold_count_rows)

        mean_per_model = {name: float(df_macro_f1[name].mean()) for name in model_names}
        general_macro_f1 = float(np.mean(list(mean_per_model.values())))

        f1_per_class_summary = {}
        for name in model_names:
            denom = max(1, f1_per_class_count_by_model[name])
            f1_mean = f1_per_class_sum_by_model[name] / denom
            for j, lab in enumerate(all_labels):
                f1_per_class_summary[f"{name}__class_{lab}"] = float(f1_mean[j])

        matriz_confusao_pack = {"labels": all_labels.tolist()}
        for name in model_names:
            matriz_confusao_pack[name] = conf_sum_by_model[name].tolist()

        trial.set_user_attr("macro_f1", df_macro_f1.to_dict(orient="list"))
        trial.set_user_attr("acc", df_acc.to_dict(orient="list"))
        trial.set_user_attr("fold_counts", df_fold_counts.to_dict(orient="list"))
        trial.set_user_attr("f1_per_class_summary", f1_per_class_summary)
        trial.set_user_attr("matriz_confusao", matriz_confusao_pack)
        trial.set_user_attr("features_union", sorted(list(used_cols_union)))
        trial.set_user_attr("features_after_nan_union", sorted(list(used_cols_after_nan_union)))

        return general_macro_f1

    return objective


# =========================================================
# SALVAMENTO
# =========================================================
def save_study_outputs(study, pasta_scenario: str, nome_run: str, scenario_name: str, detector_name: Optional[str]):
    os.makedirs(pasta_scenario, exist_ok=True)
    best_trial = study.best_trial

    path_obj = os.path.join(pasta_scenario, "df_objetive_all.csv")
    path_f1 = os.path.join(pasta_scenario, "df_f1_all.csv")
    path_f1_cls = os.path.join(pasta_scenario, "df_f1_all_classe.csv")
    path_acc = os.path.join(pasta_scenario, "df_acc_all.csv")
    path_cm = os.path.join(pasta_scenario, "df_matriz_all.csv")
    path_features = os.path.join(pasta_scenario, "df_features_used_all.csv")
    path_folds = os.path.join(pasta_scenario, "df_fold_counts_all.csv")

    df_obj = load_or_empty_csv(path_obj)
    df_f1_all = load_or_empty_csv(path_f1)
    df_f1_cls_all = load_or_empty_csv(path_f1_cls)
    df_acc_all = load_or_empty_csv(path_acc)
    df_cm_all = load_or_empty_csv(path_cm)
    df_features_all = load_or_empty_csv(path_features)
    df_folds_all = load_or_empty_csv(path_folds)

    df_obj = pd.concat([df_obj, pd.DataFrame([study.best_value], columns=[nome_run])], axis=1, ignore_index=False)
    df_obj.to_csv(path_obj, index=False)

    melhor_f1 = pd.DataFrame(best_trial.user_attrs["macro_f1"])
    melhor_f1.insert(0, "run", nome_run)
    df_f1_all = pd.concat([df_f1_all, melhor_f1], axis=0, ignore_index=True)
    df_f1_all.to_csv(path_f1, index=False)

    melhor_acc = pd.DataFrame(best_trial.user_attrs["acc"])
    melhor_acc.insert(0, "run", nome_run)
    df_acc_all = pd.concat([df_acc_all, melhor_acc], axis=0, ignore_index=True)
    df_acc_all.to_csv(path_acc, index=False)

    f1_cls_row = best_trial.user_attrs["f1_per_class_summary"].copy()
    f1_cls_row["run"] = nome_run
    df_f1_cls_all = pd.concat([df_f1_cls_all, pd.DataFrame([f1_cls_row])], axis=0, ignore_index=True)
    df_f1_cls_all.to_csv(path_f1_cls, index=False)

    cm_pack = best_trial.user_attrs["matriz_confusao"]
    cm_row = {"run": nome_run, **{k: json.dumps(v) for k, v in cm_pack.items()}}
    df_cm_all = pd.concat([df_cm_all, pd.DataFrame([cm_row])], axis=0, ignore_index=True)
    df_cm_all.to_csv(path_cm, index=False)

    features_union = best_trial.user_attrs.get("features_union", [])
    features_after_nan = best_trial.user_attrs.get("features_after_nan_union", [])
    df_feat_row = pd.DataFrame([{
        "run": nome_run,
        "scenario": scenario_name,
        "detector_name": detector_name if detector_name is not None else "",
        "n_features_union": int(len(features_union)),
        "n_features_after_nan_union": int(len(features_after_nan)),
        "features_union": json.dumps(features_union),
        "features_after_nan_union": json.dumps(features_after_nan),
    }])
    df_features_all = pd.concat([df_features_all, df_feat_row], axis=0, ignore_index=True)
    df_features_all.to_csv(path_features, index=False)

    fold_counts = pd.DataFrame(best_trial.user_attrs["fold_counts"])
    fold_counts.insert(0, "run", nome_run)
    df_folds_all = pd.concat([df_folds_all, fold_counts], axis=0, ignore_index=True)
    df_folds_all.to_csv(path_folds, index=False)

    resumo = {
        "scenario": scenario_name,
        "detector_name": detector_name,
        "best_macro_f1": float(study.best_value),
        "best_params": to_python_types(best_trial.params),
        "fixed_catch24_window_size": to_python_types(best_trial.user_attrs.get("fixed_catch24_window_size")),
    }
    with open(os.path.join(pasta_scenario, "resultado_otimizacao.json"), "w") as f:
        json.dump(resumo, f, indent=4)

    return {
        "scenario": scenario_name,
        "detector_name": detector_name if detector_name is not None else "baseline",
        "best_macro_f1": float(study.best_value),
    }


# =========================================================
# EXECUÇÃO
# =========================================================
def run_single_study(dfs_info, combinacoes, pasta_saida: str, scenario_name: str, detector_name: Optional[str], n_trials: int, seed: int, fixed_catch24_window_size: Optional[int] = None):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective = make_objective(
        dfs_info=dfs_info,
        combinacoes=combinacoes,
        pasta_saida=pasta_saida,
        scenario_name=scenario_name,
        detector_name=detector_name,
        fixed_catch24_window_size=fixed_catch24_window_size,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=N_JOBS_OPT)
    return study


def run_all_experiments():
    print("Carregando dados: ", PASTA_SAIDA, flush=True)
    print("Carregando dados: ", PASTA_SAIDA, flush=True, file=sys.stderr)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    dfs_info = load_dados(PASTA_ENTRADA)
    combinacoes = build_combinacoes(len(dfs_info))

    if any(s in {"drift", "catch24_drift"} for s in SCENARIOS_TO_RUN):
        precompute_drift_files(dfs_info, combinacoes, PASTA_SAIDA)

    if any(s in {"catch24", "catch24_drift"} for s in SCENARIOS_TO_RUN):
        precompute_all_catch24_windows(dfs_info, PASTA_SAIDA)

    summary_rows = []
    run_name = os.path.basename(os.path.normpath(PASTA_ENTRADA))

    if "catch24" in SCENARIOS_TO_RUN:
        print("\n===== Otimizando cenário: catch24 =====", flush=True)
        pasta_scenario = os.path.join(PASTA_SAIDA, "catch24")
        study = run_single_study(dfs_info, combinacoes, PASTA_SAIDA, "catch24", None, N_TRIALS, semente)
        summary_rows.append(save_study_outputs(study, pasta_scenario, run_name, "catch24", None))

        # --- ADICIONE ESTA LINHA ABAIXO ---
        best_catch24_window_size = study.best_trial.params["window_size"]
        print(f"[INFO] Melhor janela encontrada no Catch24: {best_catch24_window_size}")

        if MANTER_APENAS_MELHOR_JANELA_CATCH24_NO_CACHE:
            clear_runtime_caches()
            purge_catch24_cache_except_window(
                PASTA_SAIDA,
                best_catch24_window_size,
                USE_CATCH24,
                SHORT_NAMES,
            )
            precompute_best_catch24_window_for_all_bases(
                dfs_info,
                PASTA_SAIDA,
                best_catch24_window_size,
            )
            clear_runtime_caches()

    elif "catch24_drift" in SCENARIOS_TO_RUN:
        raise ValueError("Para executar catch24_drift com REUSAR_JANELA_CATCH24_OTIMIZADA=True, inclua também o cenário catch24.")

    for detector_name in DETECTORES_DISPONIVEIS:
        if "drift" in SCENARIOS_TO_RUN:

            print(f"\n===== Otimizando cenário: drift | detector={detector_name} =====", flush=True)
            pasta_scenario = os.path.join(PASTA_SAIDA, "drift", detector_name)

            resultado_path = os.path.join(pasta_scenario, "resultado_otimizacao.json")

            if os.path.exists(resultado_path):
                print(f"[SKIP] drift - {detector_name} já executado.")
                continue

            study = run_single_study(dfs_info, combinacoes, PASTA_SAIDA, "drift", detector_name, N_TRIALS, semente)
            summary_rows.append(save_study_outputs(study, pasta_scenario, run_name, "drift", detector_name))

        if "catch24_drift" in SCENARIOS_TO_RUN:
            print(f"\n===== Otimizando cenário: catch24_drift | detector={detector_name} =====")
            pasta_scenario = os.path.join(PASTA_SAIDA, "catch24_drift", detector_name)

            resultado_path = os.path.join(pasta_scenario, "resultado_otimizacao.json")

            if os.path.exists(resultado_path):
                print(f"[SKIP] catch24_drift - {detector_name} já executado.")
                continue
            
            study = run_single_study(dfs_info, combinacoes, PASTA_SAIDA, "catch24_drift", detector_name, N_TRIALS, semente, fixed_catch24_window_size=best_catch24_window_size)
            summary_rows.append(save_study_outputs(study, pasta_scenario, run_name, "catch24_drift", detector_name))


    if "baseline" in SCENARIOS_TO_RUN:
        print("\n===== Otimizando cenário: baseline =====", flush=True)
        pasta_scenario = os.path.join(PASTA_SAIDA, "baseline")
        study = run_single_study(dfs_info, combinacoes, PASTA_SAIDA, "baseline", None, N_TRIALS, semente)
        summary_rows.append(save_study_outputs(study, pasta_scenario, run_name, "baseline", None))


    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(PASTA_SAIDA, "resumo_geral_experimentos.csv"), index=False)
    return df_summary.T


# =========================================================
# LIMPEZA DE CACHE
# =========================================================
def clear_runtime_caches():
    _CATCH_CACHE.clear()
    _COLS_CACHE.clear()


def remove_temp_dirs(pasta_saida: str):
    for dirname in [CATCH_CACHE_DIRNAME, DRIFT_CACHE_DIRNAME]:
        d = os.path.join(pasta_saida, dirname)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"[OK] Cache removida: {d}")


# =========================================================
# MAIN
# =========================================================
def main():
    clear_runtime_caches()
    try:
        if DATASET_SINTETICO == False:
            run_all_experiments()
        else:
            global PASTA_ENTRADA, PASTA_SAIDA
            pasta_entrada_raiz = PASTA_ENTRADA
            pasta_saida_raiz = PASTA_SAIDA
            lista_resultados = []
            for item in os.listdir(pasta_entrada_raiz):
                if os.path.isdir(os.path.join(pasta_entrada_raiz, item)):
                    PASTA_ENTRADA = os.path.join(pasta_entrada_raiz, item)
                    PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")
                    df_summary = run_all_experiments()
                    lista_resultados.append(df_summary)

                    clear_runtime_caches()
                    remove_temp_dirs(PASTA_SAIDA)

            df_consolidado = pd.concat(lista_resultados, ignore_index=True)
            os.makedirs(pasta_saida_raiz, exist_ok=True)
            df_consolidado.to_csv(os.path.join(pasta_saida_raiz, "resumo_consolidado_experimentos.csv"), index=False)

            
        
        print("\nProcesso concluído com sucesso.")
    finally:
        clear_runtime_caches()
        remove_temp_dirs(PASTA_SAIDA)


if __name__ == "__main__":
    main()
