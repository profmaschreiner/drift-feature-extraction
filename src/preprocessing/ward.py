"""
WARD — Wearable Action Recognition Database v1.0
Pré-processamento: .mat → CSV por sujeito
=================================================
Estrutura do dataset:
  20 sujeitos (Subject1/ … Subject20/)
  13 atividades × 5 trials = 65 arquivos .mat por sujeito
  Nomenclatura: a<activity>t<trial>.mat  (ex: a9t5.mat = atividade 9, trial 5)

Cada arquivo .mat contém:
  WearableData.Class    → int (classe da atividade 1-13)
  WearableData.Subject  → int (número do sujeito)
  WearableData.Reading  → array de 5 elementos, cada (t x 5):
      cols 0-2 → acc_x, acc_y, acc_z
      cols 3-4 → gyro_x, gyro_y

Sensores (conforme WARDv1.pdf):
  Sensor 1 → left_wrist   (pulso esquerdo)
  Sensor 2 → right_wrist  (pulso direito)
  Sensor 3 → waist        (cintura)
  Sensor 4 → left_ankle   (tornozelo esquerdo)
  Sensor 5 → right_ankle  (tornozelo direito)

Total de features: 5 sensores x 5 canais = 25 features
Frequência: 20 Hz (original — sem downsampling necessário)

Dados ausentes (conforme WARDv1.pdf):
  Valores Inf indicam perda de pacote wireless.
  Tratamento: substitui Inf por NaN, aplica ffill + bfill.

Saída:
  datasetsNew/ward/base_0.csv   ← Subject1 (todos os trials concatenados)
  datasetsNew/ward/base_1.csv   ← Subject2
  ...
  datasetsNew/ward/base_19.csv  ← Subject20
  datasetsNew/ward/all_runs.csv ← todos os sujeitos
  datasetsNew/ward/sample_table.csv

Configuração experimental:
  ROTULO           = "label"
  FEATURES_ORIGINAIS = [25 features — impressas ao final]
  NOME_BASE        = "base_"
  END_BASE         = ".csv"
  CONJ_TESTE = [[0,1],[2,3],[4,5],[6,7],[8,9],
                [10,11],[12,13],[14,15],[16,17],[18,19]]

Classes:
  1=ReSt(Stand) 2=ReSi(Sit) 3=ReLi(Lie) 4=WaFo 5=WaLe 6=WaRi
  7=TuLe 8=TuRi 9=Up 10=Down 11=Jog 12=Jump 13=Push

Uso:
  python ward_preprocessing.py \
      --ward_dir /caminho/WARD \
      --out_dir  datasetsNew/WARD/processed

Referência:
  Yang et al. (2009) JAISE — Distributed Recognition of Human Actions
  Using Wearable Motion Sensor Networks.
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

SENSOR_NAMES  = ["left_wrist", "right_wrist", "waist",
                 "left_ankle", "right_ankle"]
CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y"]

FEATURE_COLS = [
    f"{s}_{c}"
    for s in SENSOR_NAMES
    for c in CHANNEL_NAMES
]  # 25 features

N_SUBJECTS   = 20
N_ACTIVITIES = 13
N_TRIALS     = 5

CLASS_NAMES = {
    1: "Stand", 2: "Sit",  3: "Lie",   4: "WalkFwd", 5: "WalkLeft",
    6: "WalkRight", 7: "TurnLeft", 8: "TurnRight", 9: "Upstairs",
    10: "Downstairs", 11: "Jog", 12: "Jump", 13: "PushWheelchair",
}


# =============================================================================
# FUNÇÕES
# =============================================================================

def load_mat(path: str) -> tuple:
    """
    Carrega arquivo .mat do WARD.
    Retorna (DataFrame com features+label, int classe) ou (None, None).
    """
    try:
        mat = sio.loadmat(path, simplify_cells=True)
        wd  = mat["WearableData"]
    except Exception as e:
        print(f"      [ERRO] {os.path.basename(path)}: {e}")
        return None, None

    activity_class = int(wd["Class"])
    readings = wd["Reading"]

    if len(readings) != 5:
        print(f"      [AVISO] {os.path.basename(path)}: "
              f"esperado 5 sensores, encontrado {len(readings)}")
        return None, None

    sensor_dfs = []
    min_t = None

    for i, r in enumerate(readings):
        arr = np.array(r, dtype=float)

        # Garante shape (t, 5)
        if arr.ndim == 2 and arr.shape[1] == 5:
            pass
        elif arr.ndim == 2 and arr.shape[0] == 5:
            arr = arr.T
        else:
            print(f"      [AVISO] Sensor {i+1} shape: {arr.shape}")
            return None, None

        # Substitui Inf por NaN (perda de pacote wireless)
        arr[np.isinf(arr)] = np.nan

        cols = [f"{SENSOR_NAMES[i]}_{ch}" for ch in CHANNEL_NAMES]
        sensor_dfs.append(pd.DataFrame(arr, columns=cols))
        min_t = len(arr) if min_t is None else min(min_t, len(arr))

    # Trunca ao menor comprimento e concatena
    sensor_dfs = [df.iloc[:min_t].reset_index(drop=True) for df in sensor_dfs]
    df = pd.concat(sensor_dfs, axis=1)

    # Imputa NaN com ffill + bfill
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    df["label"] = activity_class
    return df, activity_class


def build_sample_table(all_dfs: list) -> pd.DataFrame:
    """Tabela: linhas=sujeitos, colunas=classes."""
    cls_cols = list(CLASS_NAMES.values())
    rows = []
    for i, df in enumerate(all_dfs):
        if df is None or df.empty:
            continue
        counts = df["label"].value_counts().to_dict()
        row = {"Sujeito": f"S{i+1}"}
        for cls_id, cls_name in CLASS_NAMES.items():
            row[cls_name] = int(counts.get(cls_id, 0))
        row["Total"] = int(df["label"].count())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_table = pd.DataFrame(rows)
    total_row = {"Sujeito": "TOTAL"}
    for c in cls_cols + ["Total"]:
        total_row[c] = int(df_table[c].sum())
    return pd.concat([df_table, pd.DataFrame([total_row])], ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WARD .mat → CSV por sujeito"
    )
    parser.add_argument("--ward_dir", default="datasetsNew/WARD",
                        help="Pasta raiz do WARD1 (contém Subject1/ … Subject20/)")
    parser.add_argument("--out_dir", default="datasetsNew/WARD/processed",
                        help="Pasta de saída (default: datasetsNew/ward)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  WARD — Pré-processamento")
    print(f"  ward_dir : {args.ward_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  Hz       : 20 (original — sem downsampling)")
    print(f"  features : {len(FEATURE_COLS)} (5 sensores × 5 canais)")
    print(f"  saída    : base_0.csv (S1) … base_19.csv (S20)")
    print(f"{'='*64}\n")

    all_subject_dfs = []

    for subj_idx in range(N_SUBJECTS):
        subj_name = f"Subject{subj_idx + 1}"
        subj_dir  = os.path.join(args.ward_dir, subj_name)

        if not os.path.isdir(subj_dir):
            print(f"  [SKIP] {subj_name} não encontrado.")
            all_subject_dfs.append(None)
            continue

        print(f"  {subj_name} (base_{subj_idx}) ──")
        trial_dfs = []

        for activity in range(1, N_ACTIVITIES + 1):
            for trial in range(1, N_TRIALS + 1):
                fpath = os.path.join(subj_dir, f"a{activity}t{trial}.mat")
                if not os.path.exists(fpath):
                    continue
                df_t, _ = load_mat(fpath)
                if df_t is not None and not df_t.empty:
                    trial_dfs.append(df_t)

        if not trial_dfs:
            print(f"    [AVISO] Nenhum trial válido.")
            all_subject_dfs.append(None)
            continue

        df_subj = pd.concat(trial_dfs, ignore_index=True)
        out_path = os.path.join(args.out_dir, f"base_{subj_idx}.csv")
        df_subj.to_csv(out_path, index=False)

        counts = df_subj["label"].value_counts().sort_index().to_dict()
        print(f"    {len(df_subj):,} amostras | classes={counts}")
        all_subject_dfs.append(df_subj)

    # ── all_runs.csv ──────────────────────────────────────────────────────────
    valid = [df for df in all_subject_dfs if df is not None]
    if valid:
        all_df = pd.concat(valid, ignore_index=True)
        all_df.to_csv(os.path.join(args.out_dir, "all_runs.csv"), index=False)
        print(f"\n  all_runs.csv → {len(all_df):,} amostras totais")

    # ── Tabela ────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("AMOSTRAS POR SUJEITO E CLASSE:")
    print(f"{'='*64}")
    tbl = build_sample_table(all_subject_dfs)
    if not tbl.empty:
        print(tbl.to_string(index=False))
        tbl.to_csv(os.path.join(args.out_dir, "sample_table.csv"), index=False)

    # ── Configurações ─────────────────────────────────────────────────────────
    n_ok = sum(1 for df in all_subject_dfs if df is not None)
    print(f"\n{'='*64}")
    print("CONFIGURAÇÕES PARA O SCRIPT EXPERIMENTAL:")
    print(f"{'='*64}")
    print(f'PASTA_ENTRADA      = "{args.out_dir}"')
    print(f'PASTA_SAIDA        = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")')
    print(f'ROTULO             = "label"')
    print(f'FEATURES_ORIGINAIS = {FEATURE_COLS}')
    print(f'NOME_BASE          = "base_"')
    print(f'END_BASE           = ".csv"')
    print(f'# 10 folds — 2 sujeitos por fold (leave-two-subjects-out):')
    conj = [[i*2, i*2+1] for i in range(10)]
    print(f'CONJ_TESTE         = {conj}')
    print(f'\nSujeitos processados: {n_ok}/{N_SUBJECTS}')


if __name__ == "__main__":
    main()