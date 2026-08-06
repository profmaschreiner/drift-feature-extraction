from pathlib import Path
import re
import pandas as pd
import numpy as np


# =========================================================
# CONFIGURAÇÕES
# =========================================================
RAW_DIR = Path("datasetsNew/smartphone")
OUTPUT_DIR = Path("datasetsNew/smartphone_processed")
FREQ_ORIGINAL_HZ = 50
FREQ_DESEJADA_HZ = 50
OUTPUT_DIR.mkdir(exist_ok=True)

ACC_PATTERN = re.compile(r"acc_exp(\d+)_user(\d+)\.txt")
GYRO_PATTERN = re.compile(r"gyro_exp(\d+)_user(\d+)\.txt")

def downsampling_sequencial_sem_timestamp(df):
    if df.empty:
        return df.copy()

    periodo_original = 1.0 / FREQ_ORIGINAL_HZ
    intervalo_desejado = 1.0 / FREQ_DESEJADA_HZ

    indices = []
    ultimo_tempo = None

    for i in range(len(df)):
        tempo_atual = i * periodo_original

        if ultimo_tempo is None or (tempo_atual - ultimo_tempo) >= intervalo_desejado:
            indices.append(i)
            ultimo_tempo = tempo_atual

    return df.iloc[indices].reset_index(drop=True)
# =========================================================
# LEITURA DOS RÓTULOS
# =========================================================
def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Lê labels.txt com colunas:
    1: exp_id
    2: user_id
    3: activity_id
    4: start
    5: end

    Retorna DataFrame com nomes de colunas padronizados.
    """
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["exp_id", "user_id", "activity_id", "start", "end"],
        engine="python"
    )

    # Garante tipo inteiro
    for col in ["exp_id", "user_id", "activity_id", "start", "end"]:
        labels[col] = labels[col].astype(int)

    return labels


# =========================================================
# LEITURA DOS SINAIS
# =========================================================
def load_signal_file(file_path: Path, prefix: str) -> pd.DataFrame:
    """
    Lê arquivo de sinal triaxial e devolve DataFrame com colunas nomeadas.
    """
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"],
        engine="python"
    )
    return df


def build_labeled_dataframe(acc_file: Path, gyro_file: Path, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina acc + gyro + label para um experimento/usuário.
    Preserva a ordem temporal original.
    Descarta amostras sem rótulo.
    """
    acc_match = ACC_PATTERN.match(acc_file.name)
    gyro_match = GYRO_PATTERN.match(gyro_file.name)

    if not acc_match or not gyro_match:
        raise ValueError(f"Nome de arquivo inválido: {acc_file.name} ou {gyro_file.name}")

    exp_acc, user_acc = map(int, acc_match.groups())
    exp_gyro, user_gyro = map(int, gyro_match.groups())

    if (exp_acc, user_acc) != (exp_gyro, user_gyro):
        raise ValueError(f"acc e gyro não correspondem: {acc_file.name} vs {gyro_file.name}")

    exp_id, user_id = exp_acc, user_acc

    acc_df = load_signal_file(acc_file, "acc")
    gyro_df = load_signal_file(gyro_file, "gyro")

    if len(acc_df) != len(gyro_df):
        raise ValueError(
            f"Número de linhas diferente em acc e gyro para exp={exp_id}, user={user_id}: "
            f"{len(acc_df)} vs {len(gyro_df)}"
        )

    data = pd.concat([acc_df, gyro_df], axis=1)
    data["label"] = np.nan

    # Filtra apenas os rótulos do experimento/usuário atual
    subset = labels_df[
        (labels_df["exp_id"] == exp_id) &
        (labels_df["user_id"] == user_id)
    ].copy()

    # Preenche os rótulos nos intervalos
    # Mantém a ordem sequencial original das amostras
    for row in subset.itertuples(index=False):
        start_idx = int(row.start)
        end_idx = int(row.end)
        activity_id = int(row.activity_id)

        # Ajuste defensivo de limites
        start_idx = max(0, start_idx)
        end_idx = min(len(data) - 1, end_idx)

        if start_idx <= end_idx:
            # Considerando intervalo inclusivo [start, end]
            data.loc[start_idx:end_idx, "label"] = activity_id

    # Remove amostras sem rótulo
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)

    # Guardar metadados úteis
    data["exp_id"] = exp_id
    data["user_id"] = user_id

    return data


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_dataset(raw_dir: Path, output_dir: Path):
    labels_path = raw_dir / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {labels_path}")

    labels_df = load_labels(labels_path)

    acc_files = sorted(raw_dir.glob("acc_exp*_user*.txt"))
    gyro_files = sorted(raw_dir.glob("gyro_exp*_user*.txt"))

    if not acc_files:
        raise FileNotFoundError("Nenhum arquivo acc_exp*_user*.txt encontrado.")
    if not gyro_files:
        raise FileNotFoundError("Nenhum arquivo gyro_exp*_user*.txt encontrado.")

    # Índice de arquivos gyro por (exp_id, user_id)
    gyro_map = {}
    for gf in gyro_files:
        m = GYRO_PATTERN.match(gf.name)
        if m:
            exp_id, user_id = map(int, m.groups())
            gyro_map[(exp_id, user_id)] = gf

    # Junta todos os experimentos por usuário, preservando ordem
    user_data = {}

    for acc_file in acc_files:
        m = ACC_PATTERN.match(acc_file.name)
        if not m:
            continue

        exp_id, user_id = map(int, m.groups())
        key = (exp_id, user_id)

        if key not in gyro_map:
            print(f"[AVISO] Arquivo gyro correspondente não encontrado para {acc_file.name}")
            continue

        gyro_file = gyro_map[key]

        df_labeled = build_labeled_dataframe(acc_file, gyro_file, labels_df)

        if user_id not in user_data:
            user_data[user_id] = []

        # A ordem sequencial entre experimentos é mantida pela ordenação do nome do arquivo:
        # exp01, exp02, ..., e dentro de cada experimento a ordem original das linhas é preservada.
        user_data[user_id].append(df_labeled)

    if not user_data:
        raise RuntimeError("Nenhum dado rotulado foi processado.")

    # Gera arquivo por usuário
    summary_rows = []

    for user_id in sorted(user_data.keys()):
        df_user = pd.concat(user_data[user_id], axis=0, ignore_index=True)
        df_user = downsampling_sequencial_sem_timestamp(df_user)
        # Organiza colunas
        df_user = df_user[
            ["exp_id", "user_id", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label"]
        ]

        output_file = output_dir / f"user_{user_id:02d}.csv"
        df_user.to_csv(output_file, index=False)

        # Contagem por rótulo
        counts = df_user["label"].value_counts().sort_index()
        row = {"user_id": user_id}
        for label_id, count in counts.items():
            row[label_id] = int(count)
        summary_rows.append(row)

        print(f"[OK] Arquivo gerado: {output_file} | amostras = {len(df_user)}")

    # Gera tabela usuário x rótulo
    summary_df = pd.DataFrame(summary_rows).fillna(0)
    summary_df = summary_df.sort_values("user_id").reset_index(drop=True)

    # Renomeia colunas de rótulo
    new_columns = []
    for col in summary_df.columns:
        if col == "user_id":
            new_columns.append("user_id")
        else:
            new_columns.append(f"label_{int(col)}")
    summary_df.columns = new_columns

    summary_file = output_dir / "tabela_usuarios_x_rotulos.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"[OK] Tabela resumo gerada: {summary_file}")
    print("\nResumo:")
    print(summary_df)


# =========================================================
# EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    process_dataset(RAW_DIR, OUTPUT_DIR)