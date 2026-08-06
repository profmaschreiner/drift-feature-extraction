import pandas as pd
from pathlib import Path

# Caminho do arquivo
input_csv = "datasetsNew/mhealth_raw_data.csv"

# Pasta de saída
output_dir = Path("datasetsNew/dados_mhealth")
output_dir.mkdir(exist_ok=True)

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








# Identifica mudanças de classe (segmentos)
df["segment_id"] = (df[col_classe] != df[col_classe].shift()).cumsum()

df_reduzido = []

DROP_CLASS_0 = True  # Ajuste conforme necessário
# fator de redução (ajuste aqui)
#k = 30  # mantém 1 a cada 10 pontos da classe 0

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

df = df_final












# Verificar se coluna existe
if "subject" not in df.columns:
    raise ValueError("A coluna 'subject' não existe no CSV.")

# Agrupar por subject e salvar arquivos
for subject, df_group in df.groupby("subject"):
    print(f"\nBase: {subject}")
    print(df_group[col_classe].value_counts())
    output_file = output_dir / f"subject_{subject}.csv"
    df_group.to_csv(output_file, index=False)

print("Arquivos gerados com sucesso!")