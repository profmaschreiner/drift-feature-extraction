import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# =========================
# CONFIGURAÇÕES
# =========================
DATASET_DIR = Path("datasetsNew/usc_had")

# Arquivos finais já com downsampling
OUTPUT_DIR = DATASET_DIR / "csv_por_sujeito_downsampled"

# Estatísticas
STATS_DIR = DATASET_DIR / "estatisticas_downsampling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# 100 Hz / 5 = 20 Hz
DOWNSAMPLE_STEP = 5

FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

ACTIVITY_NAMES = {
    1: "Walking Forward",
    2: "Walking Left",
    3: "Walking Right",
    4: "Walking Upstairs",
    5: "Walking Downstairs",
    6: "Running Forward",
    7: "Jumping Up",
    8: "Sitting",
    9: "Standing",
    10: "Sleeping",
    11: "Elevator Up",
    12: "Elevator Down",
}


# =========================
# FUNÇÕES AUXILIARES
# =========================
def parse_filename(mat_path: Path):
    match = re.match(r"a(\d+)t(\d+)\.mat$", mat_path.name)

    if not match:
        raise ValueError(f"Nome de arquivo inesperado: {mat_path.name}")

    activity_id = int(match.group(1))
    trial_id = int(match.group(2))

    return activity_id, trial_id


def get_field(mat_data, field_name):
    if field_name in mat_data:
        return mat_data[field_name]

    for key, value in mat_data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.dtype.names:
            if field_name in value.dtype.names:
                return value[field_name][0, 0]

    return None


def load_trial(mat_path: Path, subject_id: int) -> pd.DataFrame:
    activity_id, trial_id = parse_filename(mat_path)

    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    sensor_readings = get_field(mat, "sensor_readings")

    if sensor_readings is None:
        raise ValueError(f"Campo sensor_readings não encontrado em {mat_path}")

    sensor_readings = np.asarray(sensor_readings)

    if sensor_readings.ndim != 2:
        sensor_readings = np.squeeze(sensor_readings)

    if sensor_readings.shape[1] != 6 and sensor_readings.shape[0] == 6:
        sensor_readings = sensor_readings.T

    if sensor_readings.shape[1] != 6:
        raise ValueError(
            f"Formato inesperado em {mat_path}: shape={sensor_readings.shape}"
        )

    df = pd.DataFrame(sensor_readings, columns=FEATURE_COLS)

    df["subject"] = subject_id
    df["activity_code"] = activity_id
    df["activity"] = ACTIVITY_NAMES.get(activity_id, f"Activity {activity_id}")
    df["trial"] = trial_id
    df["sample_index_original"] = np.arange(len(df))

    return df


def downsample_trial(df_trial: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Downsampling sistemático dentro de cada trial.
    Exemplo:
    step=5 -> mantém uma amostra a cada 5 pontos.
    """
    df_down = df_trial.iloc[::step].copy()
    df_down["sample_index_downsampled"] = np.arange(len(df_down))

    return df_down


def dataset_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    n_blocks = df["subject"].nunique()
    n_samples = len(df)
    n_classes = df["activity_code"].nunique()
    n_features = len(FEATURE_COLS)

    return pd.DataFrame([{
        "Datasets": dataset_name,
        "Blocks": n_blocks,
        "Samples": n_samples,
        "Classes": n_classes,
        "Features": n_features,
        "Features catch22": 24 * n_features,
        "Features with our proposal": 2 * n_features,
    }])


def subject_activity_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .pivot_table(
            index="subject",
            columns="activity",
            values="acc_x",
            aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df
        .groupby(["activity_code", "activity"])
        .size()
        .reset_index(name="samples")
        .sort_values("activity_code")
    )

    dist["percentage"] = 100 * dist["samples"] / len(df)

    return dist


def subject_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .groupby(["subject", "activity_code", "activity"])
        .size()
        .reset_index(name="samples")
        .sort_values(["subject", "activity_code"])
    )


# =========================
# PROCESSAMENTO
# =========================
all_subject_dfs_original = []
all_subject_dfs_downsampled = []

subject_dirs = sorted(
    [
        p for p in DATASET_DIR.iterdir()
        if p.is_dir() and p.name.lower().startswith("subject")
    ],
    key=lambda p: int(re.findall(r"\d+", p.name)[0])
)

for subject_dir in subject_dirs:
    subject_id = int(re.findall(r"\d+", subject_dir.name)[0])

    mat_files = sorted(
        subject_dir.glob("*.mat"),
        key=lambda p: parse_filename(p)
    )

    dfs_trials_original = []
    dfs_trials_downsampled = []

    for mat_path in mat_files:
        try:
            df_trial = load_trial(mat_path, subject_id)
            df_trial_down = downsample_trial(df_trial, DOWNSAMPLE_STEP)

            dfs_trials_original.append(df_trial)
            dfs_trials_downsampled.append(df_trial_down)

        except Exception as e:
            print(f"[ERRO] {mat_path}: {e}", flush=True)

    if not dfs_trials_original:
        print(f"[AVISO] Nenhum trial válido para {subject_dir.name}", flush=True)
        continue

    df_subject_original = pd.concat(dfs_trials_original, ignore_index=True)
    df_subject_down = pd.concat(dfs_trials_downsampled, ignore_index=True)

    cols_down = [
        "subject",
        "activity_code",
        "activity",
        "trial",
        "sample_index_original",
        "sample_index_downsampled",
        *FEATURE_COLS,
    ]

    df_subject_down = df_subject_down[cols_down]

    output_file = OUTPUT_DIR / f"usc_had_subject_{subject_id-1}.csv"
    df_subject_down.to_csv(output_file, index=False)

    all_subject_dfs_original.append(df_subject_original)
    all_subject_dfs_downsampled.append(df_subject_down)

    reduction = 100 * (1 - len(df_subject_down) / len(df_subject_original))

    print(
        f"[OK] Subject {subject_id:02d}: "
        f"original={len(df_subject_original)} | "
        f"downsampled={len(df_subject_down)} | "
        f"redução={reduction:.2f}% | "
        f"arquivo={output_file}",
        flush=True
    )


# =========================
# ESTATÍSTICAS GERAIS
# =========================
df_all_original = pd.concat(all_subject_dfs_original, ignore_index=True)
df_all_down = pd.concat(all_subject_dfs_downsampled, ignore_index=True)

summary_original = dataset_summary(df_all_original, "USC-HAD original")
summary_down = dataset_summary(df_all_down, f"USC-HAD downsampled step={DOWNSAMPLE_STEP}")

summary_all = pd.concat([summary_original, summary_down], ignore_index=True)

summary_all["Reduction samples (%)"] = [
    0.0,
    100 * (1 - len(df_all_down) / len(df_all_original))
]

summary_file = STATS_DIR / "usc_had_summary_original_vs_downsampled.csv"
summary_all.to_csv(summary_file, index=False)

print("\nResumo geral:")
print(summary_all.to_string(index=False))


# =========================
# DISTRIBUIÇÃO POR CLASSE
# =========================
dist_class_original = class_distribution(df_all_original)
dist_class_down = class_distribution(df_all_down)

dist_class_original.to_csv(
    STATS_DIR / "usc_had_class_distribution_original.csv",
    index=False
)

dist_class_down.to_csv(
    STATS_DIR / "usc_had_class_distribution_downsampled.csv",
    index=False
)

print("\nDistribuição por classe - original:")
print(dist_class_original.to_string(index=False))

print("\nDistribuição por classe - downsampled:")
print(dist_class_down.to_string(index=False))


# =========================
# DISTRIBUIÇÃO SUJEITO × CLASSE
# =========================
pivot_original = subject_activity_pivot(df_all_original)
pivot_down = subject_activity_pivot(df_all_down)

pivot_original.to_csv(
    STATS_DIR / "usc_had_subject_activity_distribution_original.csv"
)

pivot_down.to_csv(
    STATS_DIR / "usc_had_subject_activity_distribution_downsampled.csv"
)

print("\nSujeito × classe - original:")
print(pivot_original)

print("\nSujeito × classe - downsampled:")
print(pivot_down)


# =========================
# DISTRIBUIÇÃO LONGA SUJEITO × CLASSE
# =========================
subject_class_original = subject_class_distribution(df_all_original)
subject_class_down = subject_class_distribution(df_all_down)

subject_class_original.to_csv(
    STATS_DIR / "usc_had_subject_class_distribution_original_long.csv",
    index=False
)

subject_class_down.to_csv(
    STATS_DIR / "usc_had_subject_class_distribution_downsampled_long.csv",
    index=False
)


# =========================
# COMPARAÇÃO ANTES × DEPOIS POR CLASSE
# =========================
comparison_class = dist_class_original.merge(
    dist_class_down,
    on=["activity_code", "activity"],
    suffixes=("_original", "_downsampled")
)

comparison_class["reduction_samples"] = (
    comparison_class["samples_original"] -
    comparison_class["samples_downsampled"]
)

comparison_class["reduction_percentage"] = (
    100 * comparison_class["reduction_samples"] /
    comparison_class["samples_original"]
)

comparison_class.to_csv(
    STATS_DIR / "usc_had_class_distribution_comparison.csv",
    index=False
)

print("\nComparação por classe:")
print(comparison_class.to_string(index=False))


print(f"\n[OK] CSVs downsampled salvos em: {OUTPUT_DIR}")
print(f"[OK] Estatísticas salvas em: {STATS_DIR}")