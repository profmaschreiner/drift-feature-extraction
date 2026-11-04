import pandas as pd
from pathlib import Path

# Caminho do arquivo
input_csv = "datasetsNew/mhealth_raw_data.csv"

# Pasta de saída
output_dir = Path("datasetsNew/dados_mhealth")

# False: só imprime as contagens, sem regravar os arquivos dos sujeitos.
# True : comportamento original (grava subject_*.csv em output_dir).
SALVAR_ARQUIVOS = False


def resumo_amostras(d, titulo):
    n = len(d)
    n0 = int((d[col_classe] == 0).sum())
    print(f"\n===== {titulo} =====")
    print(f"amostras totais          : {n:,}")
    print(f"classe 0                 : {n0:,} ({100 * n0 / n:.2f}%)")
    print(f"atividades (sem classe 0): {n - n0:,}")
    print(f"classes de atividade     : {d.loc[d[col_classe] != 0, col_classe].nunique()}")
    if "subject" in d.columns:
        print(f"sujeitos                 : {d['subject'].nunique()}")

# Ler dados
df = pd.read_csv(input_csv)
col_classe = "Activity"
contagem = df[col_classe].value_counts().sort_index()
proporcao = df[col_classe].value_counts(normalize=True).sort_index() * 100

resultado = pd.DataFrame({
    "quantidade": contagem,
    "percentual (%)": proporcao
})

print(resultado)
resumo_amostras(df, "ANTES")








# Identifica mudanças de classe (segmentos)
df["segment_id"] = (df[col_classe] != df[col_classe].shift()).cumsum()

df_reduzido = []

DROP_CLASS_0 = False # Ajuste conforme necessário
# fator de redução (ajuste aqui)
k = 30  # mantém 1 a cada 30 pontos da classe 0

for seg_id, seg in df.groupby("segment_id"):
    classe = seg[col_classe].iloc[0]
    
    if classe == 0:
        if DROP_CLASS_0:
            continue  # descarta completamente a classe 0
        else:
            # subamostragem sequencial (preserva estrutura)
            seg_reduzido = seg.iloc[::k]
    else:
        # mantém todas as outras classes
        seg_reduzido = seg
    
    df_reduzido.append(seg_reduzido)

df_final = pd.concat(df_reduzido).drop(columns=["segment_id"])

print("Antes:")
print(df[col_classe].value_counts())

print("\nDepois:")
print(df_final[col_classe].value_counts())

resumo_amostras(df_final, f"DEPOIS (classe 0 com 1 a cada {k} amostras)")

df = df_final












# Verificar se coluna existe
if "subject" not in df.columns:
    raise ValueError("A coluna 'subject' não existe no CSV.")

if not SALVAR_ARQUIVOS:
    print("\n[SALVAR_ARQUIVOS = False] nenhum arquivo foi gravado.")
    raise SystemExit(0)

output_dir.mkdir(exist_ok=True)

# Agrupar por subject e salvar arquivos
for subject, df_group in df.groupby("subject"):
    print(f"\nBase: {subject}")
    print(df_group[col_classe].value_counts())
    output_file = output_dir / f"subject_{subject}.csv"
    df_group.to_csv(output_file, index=False)

print("Arquivos gerados com sucesso!")