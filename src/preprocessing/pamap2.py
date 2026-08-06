import os
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

INPUT_DIR = "datasetsNew/pamap2"   # ajuste para a pasta onde estão os .dat
OUTPUT_DIR = "datasetsNew/pamap2_filtrado"
SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108]  # descarta o 109
CLASSES_KEEP = [1, 2, 3, 4, 12, 13, 16, 17]
PRE = "subject"
SEPARADOR = r"[,\s;]+"

CLASSE = "activityID"
# Nomes opcionais das atividades para a tabela

LABEL_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
}
"""
LABEL_NAMES = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "11",
    12: "12",
    13: "13",
    14: "14",
    15: "15",
    16: "16",
    17: "17"
}
"""
# ============================================================
# COLUNAS DO PAMAP2
# ============================================================
# 54 colunas: timestamp + activityID + heartRate + 17 IMU hand + 17 chest + 17 ankle
column_names = [
    "timestamp",
    "activityID",
    "heartRate",
    "hand_temperature",
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "hand_acc6_x", "hand_acc6_y", "hand_acc6_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_orient_1", "hand_orient_2", "hand_orient_3", "hand_orient_4",
    "chest_temperature",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "chest_acc6_x", "chest_acc6_y", "chest_acc6_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_orient_1", "chest_orient_2", "chest_orient_3", "chest_orient_4",
    "ankle_temperature",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
    "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_orient_1", "ankle_orient_2", "ankle_orient_3", "ankle_orient_4",
]
assert len(column_names) == 54, "A lista de colunas deve ter 54 elementos."
"""
INPUT_DIR = "datasetsNew/gas"   # ajuste para a pasta onde estão os .dat
OUTPUT_DIR = "datasetsNew/gas_filtrado"
SUBJECTS = [1, 2, 3, 4, 5, 6,7,8,9,10]  
PRE = "batch"
CLASSE= "0"
CLASSES_KEEP = [1, 2, 3, 4,5,6]
SEPARADOR = r"[,\s;:]+"
# Nomes opcionais das atividades para a tabela
LABEL_NAMES = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6"
}

column_names = [str(i) for i in range(0, 258)]  # exemplo: colunas "1", "2", ..., "16"

assert len(column_names) == 258, "A lista de colunas deve ter 258 elementos."
"""
# ============================================================
# FUNÇÕES
# ============================================================
def read_subject_file(file_path: Path) -> pd.DataFrame:
    """
    Lê um arquivo .dat do PAMAP2.
    """
    df = pd.read_csv(
        file_path,
        sep=SEPARADOR,
        header=None,
        names=column_names,
        na_values="NaN",
        engine="python",
    )
    return df


def build_count_table(dfs: dict, classes_keep: list) -> pd.DataFrame:
    """
    Gera tabela classe x sujeito com contagem de amostras.
    """
    rows = []

    for cls in classes_keep:
        row = {}
        subjects_with_class = 0
        total = 0

        for subject_name, df in dfs.items():
            count = int((df[CLASSE] == cls).sum())
            row[subject_name] = count
            total += count
            if count > 0:
                subjects_with_class += 1

        label_text = LABEL_NAMES.get(cls, f"class_{cls}")
        row_name = f"{cls} – {label_text}"

        row["Activity"] = row_name
        row["Sum"] = total
        row["Nr. of subjects"] = subjects_with_class
        rows.append(row)

    count_df = pd.DataFrame(rows)

    subject_cols = [f"{PRE}{s}" for s in SUBJECTS]
    count_df = count_df[["Activity"] + subject_cols + ["Sum", "Nr. of subjects"]]
    return count_df


def verify_classes_in_all_files(dfs: dict, classes_keep: list) -> pd.DataFrame:
    """
    Verifica se cada classe aparece em todos os arquivos.
    """
    records = []

    for cls in classes_keep:
        presence = {}
        for subject_name, df in dfs.items():
            presence[subject_name] = int((df[CLASSE] == cls).sum() > 0)

        record = {"class": cls, "activity": LABEL_NAMES.get(cls, str(cls))}
        record.update(presence)
        record["present_in_all_8_files"] = int(all(v == 1 for v in presence.values()))
        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# PROCESSAMENTO
# ============================================================
def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs_original = {}
    dfs_filtered = {}

    # 1) Ler os 8 arquivos
    for subject_id in SUBJECTS:
        file_name = f"{PRE}{subject_id}.dat"
        file_path = input_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        df = read_subject_file(file_path)
        subject_name = f"{PRE}{subject_id}"
        dfs_original[subject_name] = df

    # 2) Verificar se as classes desejadas aparecem em todos os arquivos
    verification_df = verify_classes_in_all_files(dfs_original, CLASSES_KEEP)
    print("\nVerificação das classes nos 8 arquivos:")
    print(verification_df.to_string(index=False))

    missing_classes = verification_df.loc[
        verification_df["present_in_all_8_files"] == 0, "class"
    ].tolist()

    """
    if missing_classes:
        raise ValueError(
            f"As seguintes classes NÃO aparecem em todos os 8 arquivos: {missing_classes}"
        )
    """

    print("\nTodas as classes escolhidas aparecem nos 8 arquivos.")
    aux = list(range(3,258,2)) + [0]
    # 3) Filtrar cada dataframe e salvar em CSV
    for subject_name, df in dfs_original.items():
        df_f = df[df[CLASSE].isin(CLASSES_KEEP)].copy()
        #df_f = df_f[[str(a) for a in aux]]  # mantém colunas de features + classe (coluna "0")
        dfs_filtered[subject_name] = df_f

        

        out_file = output_dir / f"{subject_name}.csv"
        df_f.to_csv(out_file, index=False)

    # 4) Gerar tabela de contagem
    count_table = build_count_table(dfs_filtered, CLASSES_KEEP)

    print("\nTabela de contagem por classe x sujeito:")
    print(count_table.to_string(index=False))

    # 5) Salvar tabela
    count_table.to_csv(output_dir / "tabela_contagem_classes.csv", index=False)
    verification_df.to_csv(output_dir / "verificacao_classes.csv", index=False)

    print(f"\nArquivos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()