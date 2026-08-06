# %%

import pandas as pd
import io



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np



import pandas as pd

import matplotlib.pyplot as plt

# Lendo os dados da string simulada para um DataFrame principal
df_principal =  pd.read_csv("datasetsNew/gait.csv")
#df_principal = df_principal.drop(columns=['time', "replication", "leg", "joint"])
df_principal.rename(columns={'condition': 'label'}, inplace=True)
df_principal
#from tabpfn import TabPFNClassifier

# %%


# --- Passo 2: Identificar todos os 'subjects' únicos ---
lista_de_sujeitos = df_principal['subject'].unique()
print(f"Subjects encontrados no arquivo: {lista_de_sujeitos}\n")


lista_de_dataframes = []

for sujeito_id in lista_de_sujeitos:  
    df_sujeito = df_principal[df_principal['subject'] == sujeito_id]
    df_sujeito= df_sujeito.reset_index()
    df_sujeito = df_sujeito.drop(columns=['index', 'subject'])
    

    lista_de_dataframes.append(df_sujeito)



print(f"Foi gerada uma lista com {len(lista_de_dataframes)} DataFrames.\n")
lista_de_dataframes[0]

# %%


# %%
pd.options.mode.copy_on_write = True
lista_nova = []
for i in range(len(lista_de_dataframes)):
    df = lista_de_dataframes[i]

    lista_de_leg = df['leg'].unique()
    lista_joint_leg = []
    for leg_id in lista_de_leg:  
        df_leg = df[df['leg'] == leg_id]        
        lista_de_joint = df_leg['joint'].unique()
    
        for joint_id in lista_de_joint:  
            df_joint = df_leg[df_leg['joint'] == joint_id] 
            df_joint.rename(columns={'angle': 'angle_'+str(leg_id)+"_"+str(joint_id)}, inplace=True)
            df_joint = df_joint[['label', 'angle_'+str(leg_id)+"_"+str(joint_id)]]
            df_joint= df_joint.reset_index(drop=True)
            
            lista_joint_leg.append(df_joint)
            #print(df_joint)

    new_df = pd.concat( lista_joint_leg, ignore_index=False, axis=1)
    aux = new_df[['label']].values[:,0]
    new_df = new_df.drop(columns=['label'])
    new_df['label'] = aux
    
    lista_nova.append(new_df)


lista_de_dataframes = lista_nova    

# %%
for i, base in enumerate(lista_de_dataframes):
    base.to_csv(f"datasetsNew/gait/base{i}.csv", index=False)
