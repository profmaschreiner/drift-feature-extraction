"""
Pré-processamento do GeoTecINIT Smartphone+Smartwatch HAR Dataset
DOI: 10.5281/zenodo.8398688

O que este script faz:
  1. Para cada sujeito (s01 a s23), lê todos os arquivos CSV de execução
     em ordem sequencial, separando smartphone (sp) e smartwatch (sw).
  2. Aplica downsampling sequencial puro iloc[::5]: 100 Hz → 20 Hz.
  3. Codifica o rótulo string → inteiro (ordem alfabética).
  4. Concatena todas as execuções de cada sujeito em um único arquivo CSV.
  5. Gera 2 sub-datasets independentes: geotec_sp e geotec_sw.
  6. Gera tabela de resumo com amostras por sujeito e classe.

Estrutura dos CSVs originais (por linha):
  x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, timestamp, label

Colunas de saída:
  x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, rotulo

Classes (5):
  SEATED, SITTING_DOWN, STANDING_UP, TURNING, WALKING

Configuração do pipeline:
  PASTA_ENTRADA      = "datasetsNew/geotec_sp"   (ou geotec_sw)
  ROTULO             = "rotulo"
  FEATURES_ORIGINAIS = ["x_acc","y_acc","z_acc","x_gyro","y_gyro","z_gyro"]
  NOME_BASE          = "sujeito_"
  END_BASE           = ".csv"
  CONJ_TESTE = [
      [0,  1],   # s01 + s02
      [2,  3],   # s03 + s04
      [4,  5],   # s05 + s06
      [6,  7],   # s07 + s08
      [8,  9],   # s09 + s10
      [10, 11],  # s11 + s12
      [12, 13],  # s13 + s14
      [14, 15],  # s15 + s16
      [16, 17],  # s17 + s18
      [18, 19],  # s19 + s20
      [20, 21],  # s21 + s22
      [22],      # s23 — agrupa com fold anterior: [20, 21, 22]
  ]
"""

import os
import glob
import re
import pandas as pd

# =========================================================
# CONFIGURAÇÃO
# =========================================================
PASTA_DADOS  = "datasetsNew/sw_sp"              # pasta raiz com s01, s02, ... s23
PASTA_BASE   = "datasetsNew/sw_sp/processed"

HZ_ORIGEM    = 100
HZ_SAIDA     = 100
FATOR_DOWN   = HZ_ORIGEM // HZ_SAIDA   # = 5 → iloc[::5]

DISPOSITIVOS = ["sp", "sw"]        # smartphone e smartwatch
FEATURES_RAW = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]

# Mapeamento de rótulos (ordem alfabética → inteiro)
CLASSES_ORDEM = ["SEATED", "SITTING_DOWN", "STANDING_UP", "TURNING", "WALKING"]
MAPA_ROTULOS  = {c: i for i, c in enumerate(CLASSES_ORDEM)}

# =========================================================
# PREPARAÇÃO
# =========================================================
for disp in DISPOSITIVOS:
    os.makedirs(os.path.join(PASTA_BASE, f"geotec_{disp}"), exist_ok=True)

print(f"Downsampling : {HZ_ORIGEM} Hz → {HZ_SAIDA} Hz "
      f"(sequencial puro, fator={FATOR_DOWN}, iloc[::{FATOR_DOWN}])")
print(f"Dispositivos : {DISPOSITIVOS}")
print(f"Mapeamento   : {MAPA_ROTULOS}\n")

# =========================================================
# PROCESSAMENTO
# =========================================================
resumos = {disp: [] for disp in DISPOSITIVOS}

# Lista sujeitos ordenados: s01, s02, ..., s23
pastas_sujeitos = sorted(
    p for p in glob.glob(os.path.join(PASTA_DADOS, "s*"))
    if os.path.isdir(p)
)

if not pastas_sujeitos:
    raise FileNotFoundError(
        f"Nenhuma pasta de sujeito encontrada em '{PASTA_DADOS}'. "
        "Ajuste a variável PASTA_DADOS."
    )

print(f"Sujeitos encontrados: {len(pastas_sujeitos)}\n")

for idx, pasta_subj in enumerate(pastas_sujeitos):
    subj_id = os.path.basename(pasta_subj)   # ex: "s01"

    for disp in DISPOSITIVOS:
        # Lista todos os arquivos deste sujeito e dispositivo, ordenados
        # por número de execução: s01_01_sp.csv, s01_02_sp.csv, ...
        padrao = os.path.join(pasta_subj, f"{subj_id}_*_{disp}.csv")
        arquivos = sorted(
            glob.glob(padrao),
            key=lambda p: int(re.search(r"_(\d+)_", os.path.basename(p)).group(1))
        )

        if not arquivos:
            print(f"  ATENÇÃO: {subj_id}/{disp} — nenhum arquivo encontrado.")
            continue

        # Lê e concatena execuções em ordem sequencial
        partes = []
        for arq in arquivos:
            df_exec = pd.read_csv(arq)
            partes.append(df_exec)

        df_subj = pd.concat(partes, ignore_index=True)

        # Codifica rótulo string → inteiro
        df_subj["rotulo"] = df_subj["label"].map(MAPA_ROTULOS)

        # Remove linhas com rótulo não mapeado (se houver)
        n_antes = len(df_subj)
        df_subj = df_subj.dropna(subset=["rotulo"]).reset_index(drop=True)
        df_subj["rotulo"] = df_subj["rotulo"].astype(int)
        if len(df_subj) < n_antes:
            print(f"  [{subj_id}/{disp}] {n_antes - len(df_subj)} "
                  f"linhas removidas (rótulo desconhecido)")

        # Downsampling sequencial puro
        df_subj = df_subj[FEATURES_RAW + ["rotulo"]].iloc[::FATOR_DOWN].reset_index(drop=True)

        # Grava
        pasta_saida  = os.path.join(PASTA_BASE, f"geotec_{disp}")
        caminho_saida = os.path.join(pasta_saida, f"sujeito_{idx}.csv")
        df_subj.to_csv(caminho_saida, index=False)

        cont = {c: int((df_subj["rotulo"] == i).sum())
                for c, i in MAPA_ROTULOS.items()}
        resumos[disp].append({
            "arquivo":      f"sujeito_{idx}.csv",
            "subject_orig": subj_id,
            "n_total":      len(df_subj),
            **cont,
        })

        print(f"  [{idx:02d}] {subj_id}/{disp} → sujeito_{idx}.csv  "
              f"n={len(df_subj):>6,}  execuções={len(arquivos)}")

# =========================================================
# TABELAS RESUMO
# =========================================================
for disp in DISPOSITIVOS:
    pasta_saida = os.path.join(PASTA_BASE, f"geotec_{disp}")
    df_resumo   = pd.DataFrame(resumos[disp]).set_index("arquivo")

    for c in CLASSES_ORDEM:
        if c not in df_resumo.columns:
            df_resumo[c] = 0

    df_resumo = df_resumo[
        ["subject_orig", "n_total"] + CLASSES_ORDEM
    ].fillna(0)
    df_resumo[["n_total"] + CLASSES_ORDEM] = (
        df_resumo[["n_total"] + CLASSES_ORDEM].astype(int)
    )
    df_resumo.loc["TOTAL"] = df_resumo[["n_total"] + CLASSES_ORDEM].sum()
    df_resumo.loc["TOTAL", "subject_orig"] = "-"

    df_resumo.to_csv(os.path.join(pasta_saida, f"geotec_{disp}_resumo.csv"))

    print()
    print("=" * 80)
    print(f"GEOTEC-{disp.upper()} — amostras por sujeito e classe ({HZ_SAIDA} Hz)")
    print("=" * 80)
    print(df_resumo.to_string())

    # Verificação de classes ausentes
    print()
    sub = df_resumo.drop(index="TOTAL")
    ok  = True
    for arq, row in sub.iterrows():
        ausentes = [c for c in CLASSES_ORDEM if row[c] == 0]
        if ausentes:
            print(f"  ATENÇÃO {arq} ({row['subject_orig']}): ausentes {ausentes}")
            ok = False
    if ok:
        print(f"  OK — todas as 5 classes presentes em todos os sujeitos.")

# =========================================================
# CONFIGURAÇÃO DO PIPELINE
# =========================================================
print()
print("=" * 65)
print("CONFIGURAÇÃO PARA O PIPELINE")
print("=" * 65)
for disp in DISPOSITIVOS:
    pasta_saida = os.path.join(PASTA_BASE, f"geotec_{disp}")
    print(f"\n  # geotec_{disp}:")
    print(f'  PASTA_ENTRADA      = "{pasta_saida}"')
    print('  ROTULO             = "rotulo"')
    print(f'  FEATURES_ORIGINAIS = {FEATURES_RAW}')
    print('  NOME_BASE          = "sujeito_"')
    print('  END_BASE           = ".csv"')
    print('  CONJ_TESTE = [')
    print('      [ 0,  1],  # s01 + s02')
    print('      [ 2,  3],  # s03 + s04')
    print('      [ 4,  5],  # s05 + s06')
    print('      [ 6,  7],  # s07 + s08')
    print('      [ 8,  9],  # s09 + s10')
    print('      [10, 11],  # s11 + s12')
    print('      [12, 13],  # s13 + s14')
    print('      [14, 15],  # s15 + s16')
    print('      [16, 17],  # s17 + s18')
    print('      [18, 19],  # s19 + s20')
    print('      [20, 21, 22],  # s21 + s22 + s23')
    print('  ]')