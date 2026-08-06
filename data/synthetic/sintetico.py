# %%
from SCM import CausalGraph as cg
from SCM.Mappers import *
import pandas as pd
import numpy as np
import copy, random, os

# %%

id_drifts_geral = 0
id_drift_abrupto = 1
id_drift_incremental = 2
id_drift_gradual = 3
id_drift_reoccorrente = 4


def get_lista_drift():
    return [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]  

# %%
def dividir_em_partes_desiguais(tamanho_total: int, num_partes: int) -> list[int]:
    if num_partes <= 0:
        return []
    if num_partes == 1:
        return [tamanho_total]

    # 1. Gera N pesos aleatórios (usamos de 1 a 100 para evitar pesos muito pequenos)
    pesos = [random.randint(1, 100) for _ in range(num_partes)]
    soma_pesos = sum(pesos)

    partes = []
    soma_parcial = 0

    # 2. Calcula as N-1 primeiras partes para evitar problemas de arredondamento
    for i in range(num_partes - 1):
        # Calcula o tamanho da parte com base na proporção do seu peso
        tamanho_parte = round((pesos[i] / soma_pesos) * tamanho_total)
        partes.append(tamanho_parte)
        soma_parcial += tamanho_parte

    # 3. A última parte é o que sobrou para garantir que a soma seja exata
    ultima_parte = tamanho_total - soma_parcial
    partes.append(ultima_parte)

    # Opcional: Embaralha a lista para que a parte "de ajuste" não seja sempre a última
    random.shuffle(partes)

    return partes

# %%
def abrupt_drift(graph, size, drift_spec, rotulo, d_num_drifts):
    """Aplica um drift abrupto (uma única chamada) em cada nó informado."""
    nodes = drift_spec[1]
    # Usa apenas a API pública sem parâmetros
    try:
        for node in nodes:
            #print(f"Aplicando drift abrupto no nó {node.name}")
            node.drift_history.append(copy.deepcopy(node.mapper))
            #print(f"Antes do drift: {node.mapper}")
            node.mapper.drift()
            #print(f"Depois do drift: {node.mapper}")

            d_num_drifts[node.name][rotulo][id_drift_abrupto] += 1 # [feature][classe][tipo] 
            d_num_drifts[node.name][rotulo][id_drifts_geral] += 1 # [feature][classe][tipo]
        return pd.DataFrame(graph.generate(dataset_size=size, missing_prob=0.0, intervention_prob=0.0))
    except Exception:
        print(f"Aviso: drift() falhou para o nó {node.name}, tentando novamente."  )


# Aplica etapas de drift até completar o tamanho desejado estabilizar em um novo conceito.
def incremental_drift(graph, size, drift_spec, rotulo, d_num_drifts):
    #print(size)
    size_transition = drift_spec[2]
    steps = random.randint(2, size_transition//3)
    nodes = drift_spec[1]
    for node in nodes:
        node.drift_history.append(copy.deepcopy(node.mapper))
        d_num_drifts[node.name][rotulo][id_drift_incremental] += 1 # [feature][classe][tipo] 
        d_num_drifts[node.name][rotulo][id_drifts_geral] += 1 # [feature][classe][tipo]

    part = dividir_em_partes_desiguais(size_transition, num_partes=steps*len(nodes))
    #print("partes:", part)
    #rem  = size - part * steps
    dfs = []
    i=0
    for s in range(steps):
        for n in nodes:
            try:
                n.mapper.drift()                
            except Exception:
                print(f"Erro ao gerar drift incremental {n.name}")
            # tamanho deste sub-bloco (distribui resto nos primeiros sub-blocos)
            this_size = part[i]
            if this_size > 0:
                sub = pd.DataFrame(graph.generate(dataset_size=this_size, missing_prob=0.0, intervention_prob=0.0))
                dfs.append(sub)
            i += 1
    
    dfs.append(pd.DataFrame(graph.generate(dataset_size=size - size_transition, missing_prob=0.0, intervention_prob=0.0)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()






def gradual_drift(graph, size, drift_spec, rotulo, d_num_drifts):
    """
    Gera um bloco de dados com drift gradual aplicado a nós específicos.
    A probabilidade de usar o novo conceito (B) aumenta linearmente de 0 a 1.
    
    Args:
        graph: O objeto CausalGraph.
        size: Quantidade de amostras no período de transição.
        drift_spec: Tupla no formato ('gradual', [lista_de_objetos_Vertex])
    """
    size_transition = drift_spec[2]
    nodes_to_drift = drift_spec[1]
    sorted_vertices = graph.topological_sort()
    
    # 1. Preparação dos nós afetados
    for node in nodes_to_drift:
        # Salva o conceito atual (A) no histórico do próprio nó
        node.drift_history.append(copy.deepcopy(node.mapper))
        # O mapper atual do nó sofre o drift e se torna o conceito (B)
        node.mapper.drift()
        d_num_drifts[node.name][rotulo][id_drift_gradual] += 1 # [feature][classe][tipo] 
        d_num_drifts[node.name][rotulo][id_drifts_geral] += 1 # [feature][classe][tipo]

    samples = []
    
    # 2. Loop de geração das amostras
    for i in range(size_transition):
        # Probabilidade de escolher o novo conceito (B)
        prob_B = i / size_transition
        choose_B = random.random() < prob_B
        
        # Reset dos valores do grafo para a amostra atual
        for v in graph.vertices.values():
            v.value = None
            
        row = {}
        for v_name in sorted_vertices:
            vtx = graph.vertices[v_name]
            
            if vtx.is_root():
                # Raízes não dependem de pais, chamam o mapper diretamente
                vtx.value = float(vtx.mapper.map(None))
            else:
                if vtx in nodes_to_drift:
                    if choose_B:
                        # Usa o mapper atual (Conceito B)
                        vtx.compute_value() 
                    else:
                        # Usa o conceito que acabamos de salvar no histórico (Conceito A)
                        # O índice -1 acessa o último estado salvo antes do drift
                        vtx.compute_value_last_concept(concept_index=-1)
                else:
                    # Nós que não estão na lista de drift seguem o comportamento normal
                    vtx.compute_value()
            
            row[v_name] = vtx.value
        
        samples.append(row)
    
    df_transition = pd.DataFrame(samples)
    df_stable = pd.DataFrame(graph.generate(dataset_size=size - size_transition, missing_prob=0.0, intervention_prob=0.0))
    return pd.concat([df_transition, df_stable], ignore_index=True)



def recurrent_drift(graph, size, drift_spec, d_num_drifts, concept_index=0, rotulo=0):
    """
    Gera um bloco de dados restaurando um conceito salvo no histórico do próprio nó.
    
    Args:
        graph: O objeto CausalGraph.
        size: Quantidade de amostras.
        drift_spec: Tupla ('recurrent', [lista_de_nos], index_historia)
                    index_historia: qual posição do drift_history restaurar (ex: 0 para o original)
    """
    nodes = drift_spec[1]
    
    # 1. Determina qual versão do passado queremos recuperar
    # Se não for passado um índice, por padrão voltamos ao primeiro (index 0)
    target_idx = concept_index
    
    print(f"--- Recorrência: Restaurando estado do índice {target_idx} dos nós selecionados ---")

    for node in nodes:
        # Verifica se o nó tem histórico suficiente
        if hasattr(node, 'drift_history') and len(node.drift_history) > target_idx:
            
            # Primeiro: Salva o conceito que estava ativo ATÉ AGORA no histórico
            # Isso mantém a linha do tempo do nó completa (A -> B -> A)
            node.drift_history.append(copy.deepcopy(node.mapper))
            
            # Segundo: Restaura o mapper desejado do passado
            node.mapper = copy.deepcopy(node.drift_history[target_idx])
            d_num_drifts[node.name][rotulo][id_drift_reoccorrente] += 1 # [feature][classe][tipo] 
            d_num_drifts[node.name][rotulo][id_drifts_geral] += 1 # [feature][classe][tipo]
        else:
            print(f"Aviso: O nó {node.name} não possui histórico no índice {target_idx}. Mantendo conceito atual.")

    # 2. Gera as amostras com o grafo modificado
    return pd.DataFrame(graph.generate(dataset_size=size, missing_prob=0.0, intervention_prob=0.0))

# %%
def change_mapper_node(phase_label):
    alpha, rho = 0.5, 0.5
    if phase_label == 0:
        
        x1 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x2 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x3 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x4 = NormalMapper(ewma_alpha=alpha, rho=rho)

    elif phase_label == 1:
        
        x1 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x2 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x3 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x4 = NormalMapper(ewma_alpha=alpha, rho=rho)

    elif phase_label == 2:
        
        x1 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x2 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x3 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x4 = NormalMapper(ewma_alpha=alpha, rho=rho)

    elif phase_label == 3:
        
        x1 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x2 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x3 = NormalMapper(ewma_alpha=alpha, rho=rho)
        x4 = NormalMapper(ewma_alpha=alpha, rho=rho)
    return x1, x2, x3, x4

# %%
def get_phases_all(list_of_phases, type_drift,  l_interval_drift, l_variaveis_afetadas, l_comp):
    ph = []
    minimo = 800
    maximo = 900    
    for i in range(len(list_of_phases)):        
        if list_of_phases[i] == 0: 
            ph.append({"label": 0, "size": random.randint(minimo, maximo), "drift": [(type_drift[0],  l_variaveis_afetadas[0], l_comp[0])], "interval": l_interval_drift[0]})
        elif list_of_phases[i] == 1:
            ph.append({"label": 1, "size": random.randint(minimo, maximo), "drift": [(type_drift[1],  l_variaveis_afetadas[1], l_comp[1])], "interval": l_interval_drift[1]   })
        elif list_of_phases[i] == 2:
            ph.append({"label": 2, "size": random.randint(minimo, maximo), "drift": [(type_drift[2],  l_variaveis_afetadas[2], l_comp[2]    )], "interval": l_interval_drift[2]}) 
        elif list_of_phases[i] == 3:
            ph.append({"label": 3, "size": random.randint(minimo, maximo), "drift": [(type_drift[3],  l_variaveis_afetadas[3], l_comp[3])], "interval": l_interval_drift[3]})  
    
    return ph

# %%
# padrão: [frequencia, tipo_drift, variaveis_afetadas]
# frequencia: I ou D
# tipo_drift: abrupto, incremental, gradual, recorrente, D
# variaveis_afetadas: I ou D
def get_phases(list_of_phases, padrao, x1, x2, x3, x4):  
    ph = []  
    interval = random.randint(40, 50)
    interval0 = random.randint(40, 50)
    interval1 = random.randint(110, 120)
    interval2 = random.randint(180, 190)
    interval3 = random.randint(250, 260)
    if len(padrao) == 3:
        if padrao[1] == '' and padrao[0] == '' and padrao[2] == '':
            ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['', '', '', ''], 
                                        l_interval_drift=[0,0, 0, 0], 
                                        l_variaveis_afetadas=[[],[],[],[]],
                                        l_comp=[-1,-1,-1,-1])
        if padrao[1] == 'abrupto':
            if padrao[0] == 'I':
                
                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['abrupto', 'abrupto', 'abrupto', 'abrupto'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[-1,-1,-1,-1])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['abrupto', 'abrupto', 'abrupto', 'abrupto'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[-1,-1,-1,-1])
            elif padrao[0] == 'D': #frequencia diferente

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['abrupto', 'abrupto', 'abrupto', 'abrupto'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[-1,-1,-1,-1])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['abrupto', 'abrupto', 'abrupto', 'abrupto'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[-1,-1,-1,-1])

        elif padrao[1] == 'incremental':
            if padrao[0] == 'I':

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['incremental', 'incremental', 'incremental', 'incremental'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[21,21,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['incremental', 'incremental', 'incremental', 'incremental'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[21,21,21,21])
            elif padrao[0] == 'D': #frequencia diferente

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['incremental', 'incremental', 'incremental', 'incremental'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],    
                                        l_comp=[21,21,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['incremental', 'incremental', 'incremental', 'incremental'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[21,21,21,21])
        elif padrao[1] == 'gradual':
            if padrao[0] == 'I':

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['gradual', 'gradual', 'gradual', 'gradual'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[21,21,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['gradual', 'gradual', 'gradual', 'gradual'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[21,21,21,21])
            elif padrao[0] == 'D': #frequencia diferente

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['gradual', 'gradual', 'gradual', 'gradual'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[21,21,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['gradual', 'gradual', 'gradual', 'gradual'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[21,21,21,21])
        elif padrao[1] == 'recorrente':
            if padrao[0] == 'I':

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'recorrente', 'recorrente', 'recorrente'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[0,0,0,0])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'recorrente', 'recorrente', 'recorrente'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[0,0,0,0])   
            elif padrao[0] == 'D': #frequencia diferente

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'recorrente', 'recorrente', 'recorrente'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[0,0,0,0])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'recorrente', 'recorrente', 'recorrente'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[0,0,0,0])
            
        elif padrao[1] == 'D':
            if padrao[0] == 'I':

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'abrupto', 'incremental', 'gradual'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[0,-1,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'abrupto', 'incremental', 'gradual'], 
                                        l_interval_drift=[interval,interval, interval, interval], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[0,-1,21,21])
            elif padrao[0] == 'D': #frequencia diferente

                if padrao[2] == 'I':
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'abrupto', 'incremental', 'gradual' ], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1,x4],[x1,x4],[x1,x4],[x1,x4 ]],
                                        l_comp=[0,-1,21,21])
                elif padrao[2] == 'D':            
                    ph = get_phases_all(list_of_phases=list_of_phases, 
                                        type_drift=['recorrente', 'abrupto', 'incremental', 'gradual'], 
                                        l_interval_drift=[interval0,interval1, interval2, interval3], 
                                        l_variaveis_afetadas=[[x1],[x2],[x3],[x4]],
                                        l_comp=[0,-1,21,21])
                
        
    return ph

# %%
col_drift = 0
col_drift_win = 1
col_drift_type = 2

def add_metadata(block_size, is_drift, drift_type, l_drifts, win_size=50):
    """Função auxiliar para preencher os metadados do bloco"""
    for i in range(block_size):
        # col_drift: 1 apenas na primeira linha do bloco onde o drift começa
        l_drifts[0].append(1 if (is_drift and i == 0) else 0)
        
        # col_drift_win: 1 se estiver dentro da janela após o ponto de drift
        l_drifts[1].append(1 if (is_drift and i < win_size) else 0)
        
        # col_drift_type: O tipo do drift ou "none"
        l_drifts[2].append(drift_type if is_drift else "none")
    return l_drifts



def generate_block_meta(graph, size, lista_spec, rotulo, interval, d_num_drifts, win_size=50):
    lista_dataframes = []
    # Estrutura para armazenar os metadados: [is_drift_point, acceptance_window, drift_type]
    l_drifts = [[], [], []] 

    

    for spec in lista_spec:
        d_type = spec[0]
        
        # --- BLOCO INICIAL (Sem Drift) ---
        # O primeiro intervalo é sempre estável (ground truth inicial)
        lista_dataframes.append(pd.DataFrame(graph.generate(dataset_size=interval, missing_prob=0.0, intervention_prob=0.0)))
        add_metadata(interval, is_drift=False, drift_type="none", l_drifts=l_drifts, win_size=win_size)

        # --- BLOCOS COM DRIFT ---
        if d_type == "abrupto":
            for i in range(interval, size, interval):
                lista_dataframes.append(abrupt_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
                add_metadata(interval, is_drift=True, drift_type="abrupto", l_drifts=l_drifts, win_size=win_size)

        elif d_type == "incremental":
            for i in range(interval, size, interval):                
                lista_dataframes.append(incremental_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
                add_metadata(interval, is_drift=True, drift_type="incremental", l_drifts=l_drifts, win_size=win_size)

        elif d_type == "gradual":
            for i in range(interval, size, interval):                
                lista_dataframes.append(gradual_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
                add_metadata(interval, is_drift=True, drift_type="gradual", l_drifts=l_drifts, win_size=win_size)

        elif d_type == "recorrente":
            # Caso especial: o código original tem um bloco inicial de 50
            # Ajustando o bloco inicial do recorrente
            rec_initial_size = 50 
            # (Note: se mudar o tamanho do bloco inicial aqui, os metadados devem acompanhar)
            # Como o código original faz um append manual antes do loop:
            
            # Já geramos o baseline 'interval' acima, o recorrente no seu código 
            # adiciona mais um abrupt_drift antes do loop.
            lista_dataframes.append(abrupt_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
            add_metadata(interval, is_drift=True, drift_type="abrupto", l_drifts=l_drifts, win_size=win_size)
            
            new_concept_index = 0
            for i in range(interval, size, interval):                
                lista_dataframes.append(recurrent_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, concept_index=new_concept_index, d_num_drifts=d_num_drifts))
                add_metadata(interval, is_drift=True, drift_type="recorrente", l_drifts=l_drifts, win_size=win_size)
                new_concept_index = 1 - new_concept_index

    # Concatenação dos dados
    newDF = pd.concat(lista_dataframes, ignore_index=True).reset_index(drop=True)
    
    # Construção do DataFrame de Drift
    df_drifts = pd.DataFrame({
        'is_drift_point': l_drifts[0],
        'acceptance_window': l_drifts[1],
        'drift_type': l_drifts[2]
    })

    print(f"Total classe {rotulo}: {newDF.shape}, {newDF.shape} shape, {df_drifts['is_drift_point'].sum()} drifts gerados.")
    
    return newDF, df_drifts

# %%

def generate_block(graph, size, lista_spec, rotulo, interval, d_num_drifts):
    lista = []
    l_drifts = [[],[],[]]
    for spec in lista_spec:
      
        if spec[0] == "abrupto":
            lista_abrupt = [pd.DataFrame(graph.generate(dataset_size=interval, missing_prob=0.0, intervention_prob=0.0))]
            for i in range(interval, size, interval):
                                
                lista_abrupt.append(abrupt_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
            for df in lista_abrupt:
                print(df.shape)
            df = pd.concat(lista_abrupt, ignore_index=True) 
            
            df = df.reset_index(drop=True)
            #print(df.shape)
            lista.append(df) 
        elif spec[0] == "incremental":
            lista_inc = [pd.DataFrame(graph.generate(dataset_size=interval, missing_prob=0.0, intervention_prob=0.0))]
            for i in range(interval, size, interval):                
                lista_inc.append(incremental_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
            df = pd.concat(lista_inc, ignore_index=True) 
            df = df.reset_index(drop=True)
            #print(df.shape)
            lista.append(df) 
        elif spec[0] == "gradual":
            lista_grad = [pd.DataFrame(graph.generate(dataset_size=interval, missing_prob=0.0, intervention_prob=0.0))]
            for i in range(interval, size, interval):                
                lista_grad.append(gradual_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts))
            df = pd.concat(lista_grad, ignore_index=True) 
            df = df.reset_index(drop=True)
            #print(df.shape)
            lista.append(df) 
        elif spec[0] == "recorrente":
            lista_rec = [pd.DataFrame(graph.generate(dataset_size=50, missing_prob=0.0, intervention_prob=0.0))]  # 0            
            lista_rec.append(abrupt_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, d_num_drifts=d_num_drifts)) # 1
            new_concept_index = 0
            for i in range(interval, size, interval):                
                lista_rec.append(recurrent_drift(graph, size=interval, drift_spec=spec, rotulo=rotulo, concept_index=new_concept_index, d_num_drifts=d_num_drifts))
                new_concept_index = 1 - new_concept_index  # alterna entre 0 e 1
            df = pd.concat(lista_rec, ignore_index=True) 
            df = df.reset_index(drop=True)
            #print(df.shape)
            lista.append(df) 


    newDF = pd.concat(lista, ignore_index=True) 
    newDF = newDF.reset_index(drop=True)
    print("total classe ", rotulo)
    print(newDF.shape)
    return  newDF, df_drifts

# %%

def generate_simple_dataset( dataset_id, list_of_phases, padrao, d_num_drifts, output_dir):
    

    # Grafo simples: 5 features independentes -> y
    win_size = 50
    alpha, rho = 0.1, 0.5
    #x0 = cg.Vertex("x0", mapper=NormalMapper(ewma_alpha=alpha, rho=rho))  # nó auxiliar para o bias
    m_x1, m_x2, m_x3, m_x4 = change_mapper_node(list_of_phases[0])

    x1 = cg.Vertex("x1", mapper=m_x1)
    x2 = cg.Vertex("x2", mapper=m_x2)
    x3 = cg.Vertex("x3", mapper=m_x3)
    x4 = cg.Vertex("x4", mapper=m_x4)
    
    y  = cg.Vertex("y",  mapper=PhaseControlledMapper(n_phases=4 ))

    graph = cg.CausalGraph()
    for xi in [x1, x2, x3, x4]: 
        #graph.add_edge(x0, xi)   
        graph.add_edge(xi, y) 
        d_num_drifts[xi.name] = get_lista_drift()    
        #print(xi.name)        
    
    for xi in [x1, x2, x3, x4, y]:        
        graph.add_vertex(xi)
    
    #graph.visualize_graph()

    phases = get_phases(list_of_phases, padrao, x1, x2, x3, x4)

    dataset = []
    meta_drift = []
    df_drifts = None
    f = 0
    
    for i, phase in enumerate(phases):
        
        label = phase["label"]
        size  = phase["size"]
        spec  = phase["drift"]
        interval = phase["interval"]

        y.mapper.drift(p=label)  # muda a fase do alvo])
        m_x1, m_x2, m_x3, m_x4 = change_mapper_node(label)
        graph.vertices["x1"].mapper = m_x1  
        graph.vertices["x2"].mapper = m_x2
        graph.vertices["x3"].mapper = m_x3
        graph.vertices["x4"].mapper = m_x4
        #graph.is_trained = False

        if f > 0:
            print("Atualizando drifts...")
            for xi in [x1, x2, x3, x4]:        
                d_num_drifts[xi.name][label][id_drifts_geral] += 1 # [feature][classe][tipo] 
                d_num_drifts[xi.name][label][id_drift_abrupto] += 1 # [feature][classe][tipo] 

        print(f"\nDataset {dataset_id} → Fase {i+1}: size={size}, phase={label}, drift={spec}")
        if spec[0][0] == '':
            block = pd.DataFrame(graph.generate(dataset_size=size, missing_prob=0.0, intervention_prob=0.0))
        else:
            block, df_drifts = generate_block_meta(graph=graph, size=size, lista_spec=spec, rotulo=label, interval=interval, d_num_drifts=d_num_drifts, win_size=win_size)

        if f > 0 and df_drifts is not None:
            df_drifts.loc[0, 'is_drift_point'] = 1
            df_drifts.loc[0, 'drift_type'] = f"phase_change_{label}"
            # Garante que a janela de aceitação comece no início do bloco
            win_size = win_size # ou seu parâmetro
            df_drifts.loc[0:win_size, 'acceptance_window'] = 1

        dataset.append(block)
        if spec[0][0] != '':
            meta_drift.append(df_drifts)
        f += 1
        

    # Consolida e salva
    os.makedirs(output_dir, exist_ok=True)
    data = pd.concat(dataset, ignore_index=True)
    if len(meta_drift) > 0:
        meta_data = pd.concat(meta_drift, ignore_index=True)
        meta_data.to_csv(f"{output_dir}/base_drift_{dataset_id}.csv", index=False)
    fname = f"{output_dir}/base_{dataset_id}.csv"
    data.to_csv(fname, index=False)
    
    print(f"\n✅ Dataset {dataset_id} salvo em {fname}  ({len(data)} amostras)")
    return data

# %%
random.seed(41)
np.random.seed(41)
states_dataset = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [0, 3, 2, 1],
    [3, 2, 1, 0],
    [2, 1, 0, 3],
    [1, 0, 3, 2],
    [0, 2, 3, 1],
    [1, 3, 0, 2],
    [2, 0, 1, 3],
    [3, 1, 2, 0]
]

padroes = [
    # B1–B4 (D)
    ["I", "D", "I"],   # B1
    ["I", "D", "D"],   # B2
    ["D", "D", "I"],   # B3
    ["D", "D", "D"],   # B4

    # B5–B8 (abrupto)
    
    ["I", "abrupto", "I"],  # B5
    ["I", "abrupto", "D"],  # B6
    ["D", "abrupto", "I"],  # B7
    ["D", "abrupto", "D"],  # B8

    # B9–B12 (Incremental)
    ["I", "incremental", "I"],  # B9
    ["I", "incremental", "D"],  # B10
    ["D", "incremental", "I"],  # B11
    ["D", "incremental", "D"],  # B12

    # B13–B16 (Gradual)
    ["I", "gradual", "I"],  # B13
    ["I", "gradual", "D"],  # B14
    ["D", "gradual", "I"],  # B15
    ["D", "gradual", "D"],  # B16

    # B17–B20 (Recorrente)
    ["I", "recorrente", "I"],  # B17
    ["I", "recorrente", "D"],  # B18
    ["D", "recorrente", "I"],  # B19
    ["D", "recorrente", "D"],  # B20
    
    ["", "", ""],  # B0 (sem drift)
]

output_dir="datasets_sinteticos"





# %%
def soma_drifts_bases(dfs):

    # separa colunas numéricas (exceto 'origem')
    cols_sum = dfs[0].columns.difference(['origem'])

    # soma os dataframes apenas nessas colunas
    df_soma = sum(df[cols_sum] for df in dfs)

    # recupera a coluna origem (por exemplo, do primeiro dataframe)
    df_soma['origem'] = dfs[0]['origem'].values


    return df_soma

# %%
def generate_all_datasets(padrao):
    all_datasets = []
    d_num_drifts = {}

    pasta = output_dir+"/"
    for p in padrao:
        pasta +=str(p)+'_'    
    indices = list(range(0,len(states_dataset)))

    df_num_drifts_classe1 = pd.DataFrame(columns=["0", "1", "2", "3"])
    dfs_classe_var = []
    for i in range(0, len(states_dataset)):
        df = generate_simple_dataset( dataset_id=i, list_of_phases=states_dataset[i], padrao=padrao, output_dir=pasta, d_num_drifts=d_num_drifts)
        dfs = []
        df_num_drifts_geral = pd.DataFrame(columns=["0", "1", "2", "3"])
        for nome, matriz in d_num_drifts.items():            
            df_temp = pd.DataFrame(matriz)
            nova_linha = df_temp[0].values
            df_num_drifts_geral.loc[df_num_drifts_geral.shape[0]] = nova_linha  # adiciona como nova linha
            df_temp["origem"] = nome  # adiciona coluna identificando a origem
            dfs.append(df_temp)
        
        df_num_drifts_classe1.loc[df_num_drifts_classe1.shape[0]] = df_num_drifts_geral.sum().values
        # Concatenar todos os DataFrames
        df_final = pd.concat(dfs, ignore_index=True)
        df_final.to_csv(f"{pasta}/num_drifts_{indices[i-1]}.csv", index=False)
        dfs_classe_var.append(df_final)
        #print(df_final.shape[0])
        all_datasets.append(df)

    print("\n✅ Geração concluída: 10 datasets em ''")
    df_num_drifts_classe1["base"] = indices
    df_num_drifts_classe1 = df_num_drifts_classe1.set_index("base")
    df_num_drifts_classe1.to_csv(f"{pasta}/num_drifts_classe.csv", index=False)

    soma_drifts_bases(dfs=dfs_classe_var).to_csv(f"{pasta}/num_drifts_classe_var.csv", index=False)


# %%
for padrao in padroes:
    print("\n\nGerando datasets para o padrão:", padrao)
    generate_all_datasets(padrao)

# %%


# %%



