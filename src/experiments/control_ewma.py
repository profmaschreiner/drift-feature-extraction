"""
-----------controle_ewma.py
Controles EWMA nos datasets reais, para o MELHOR detector de C1 e de C3:

  EWMA_ALARM: mesmos alarmes do detector, acumulados por uma media movel
              exponencial S_t = (1 - lam) S_{t-1} + lam a_t no lugar do
              Algorithm 1 (esquecimento e peso constantes; caso particular do
              Algorithm 1). Pergunta: o esquecimento adaptativo importa?
  EWMA_VAR:   sem detector; a feature de cada variavel original e a variancia
              exponencialmente ponderada do sinal normalizado,
              v_t = (1 - lam) v_{t-1} + lam (x_t - m_{t-1})^2,
              m_t = (1 - lam) m_{t-1} + lam x_t   (MacGregor e Harris, 1993).
              Pergunta: os detectores sao necessarios?

As features ocupam o lugar das drift scores (colunas score_<var>), no mesmo
cenario do detector de referencia: C1 (originais + score) e C3 (catch22 +
originais + score, janela do C2).

lam: calculado a partir do phi_b do melhor detector, igualando o horizonte de
memoria das duas acumulacoes. Horizonte do Algorithm 1 = numero de amostras h
ate o score cair a 1/e sem alarmes, isto e, o menor h com
sum_{k=1..h} -ln(1 - min(1, phi_b ln(1+k))) >= 1. Horizonte da EWMA = 1/lam,
logo lam_0 = 1/h. Sao avaliados lam_0 x {0.5, 1, 2} (3 avaliacoes por
controle), e o de maior F1 e mantido.

Protocolo: o MESMO do experimento original, sem validacao cruzada aninhada.
Folds (CONJ_TESTE), funcao objetivo (macro F1 medio do RF nos folds de teste),
politica de colunas, RF e metricas identicos. Sem otimizacao do RF: o scaler
dos detectores, o scaler do modelo e os hiperparametros do RF vem do JSON do
melhor detector do cenario (optuna.trial.FixedTrial), como no controle C2.raw.
Esses hiperparametros foram otimizados para o Algorithm 1, o que favorece a
referencia; um empate dos controles e, portanto, conservador.

Leitura dos JSONs (igual ao controle C2.raw):
  C2: <PASTA_ENTRADA>/<PASTA_ORIGINAL>/catch24/resultado_otimizacao.json (janela)
  C1: <PASTA_ENTRADA>/<PASTA_C3 ou PASTA_ORIGINAL>/drift/<det>/resultado_otimizacao.json
  C3: <PASTA_ENTRADA>/<PASTA_C3 ou PASTA_ORIGINAL>/catch24_drift/<det>/resultado_otimizacao.json
  PASTA_C3 = resultados_catch24_drift_rerun para RO, MH e SmPh (C1 e C3
  reexecutados); se o C1 nao estiver na pasta de reexecucao, usa a original e avisa.
Melhor detector = maior best_macro_f1 entre os nove, em cada cenario.

Saidas em <PASTA_ENTRADA>/<PASTA_ORIGINAL>/controle_ewma/:
  <C1|C3>_<EWMA_ALARM|EWMA_VAR>/  df_f1_all.csv, df_acc_all.csv,
      df_f1_all_classe.csv, df_fold_counts_all.csv, grade_lambda.csv,
      resultado_controle.json (lambda escolhido)
  resumo_controle_ewma.csv  uma linha por cenario e controle, com o F1 do
      Algorithm 1 (JSON) ao lado
Execucoes ja concluidas (resultado_controle.json presente) sao puladas, o que
permite reenviar o job sem refazer o que terminou.

Uso:
    python controle_ewma.py <dataset> [c1|c3|ambos] [EWMA_ALARM EWMA_VAR]
    ex.: python controle_ewma.py smph
         python controle_ewma.py gait c3 EWMA_VAR
"""
import os
import json
import shutil
import math

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
PASTA_ENTRADA = "datasetsNew/dados_mhealth"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "Activity"
FEATURES_ORIGINAIS = ["alx", "aly", "alz", "glx", "gly", "glz", "arx", "ary", "arz", "grx", "gry", "grz"]


NOME_BASE = "subject_subject"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]



PASTA_ENTRADA = "datasetsNew/rs_default"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "rotulo"
FEATURES_ORIGINAIS = ["sdp002", "sdt002", "sdt003", "sdu002_1", "sdu002_2"]
NOME_BASE = "base"
END_BASE = "_final.csv"
CONJ_TESTE = [[3], [4], [5], [0, 1], [0,2], [0,7], [0,8], [6, 1], [6, 2], [6,7], [6,8]]



PASTA_ENTRADA = "datasetsNew/gait_default"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "label"
FEATURES_ORIGINAIS = ["angle_1_1", "angle_1_2", "angle_1_3", "angle_2_1", "angle_2_2", "angle_2_3"]
NOME_BASE = "base"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]


PASTA_ENTRADA = "datasetsNew/idosos20hz"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "label"
FEATURES_ORIGINAIS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
NOME_BASE = "base"
END_BASE = ".csv"
#CONJ_TESTE = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13,14]]
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]



PASTA_ENTRADA = "datasetsNew/pamap2D"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift_ankle")

ROTULO = "activityID"
#FEATURES_ORIGINAIS = ["hand_temperature", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z", "hand_mag_x", "hand_mag_y", "hand_mag_z", "chest_temperature", "chest_acc16_x", "chest_acc16_y", "chest_acc16_z", "chest_acc6_x", "chest_acc6_y", "chest_acc6_z", "chest_gyro_x", "chest_gyro_y", "chest_gyro_z", "chest_mag_x", "chest_mag_y", "chest_mag_z", "ankle_temperature", "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z", "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z", "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z", "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"]
#FEATURES_ORIGINAIS = ["hand_temperature", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z", "hand_acc6_x", "hand_acc6_y", "hand_acc6_z", "hand_gyro_x", "hand_gyro_y", "hand_gyro_z", "hand_mag_x", "hand_mag_y", "hand_mag_z"]
#FEATURES_ORIGINAIS = ["chest_temperature", "chest_acc16_x", "chest_acc16_y", "chest_acc16_z", "chest_acc6_x", "chest_acc6_y", "chest_acc6_z", "chest_gyro_x", "chest_gyro_y", "chest_gyro_z", "chest_mag_x", "chest_mag_y", "chest_mag_z"]
FEATURES_ORIGINAIS = ["ankle_temperature", "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z", "ankle_acc6_x", "ankle_acc6_y", "ankle_acc6_z", "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z", "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"]

NOME_BASE = "base_"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7]]



PASTA_ENTRADA = "datasetsNew/cinematica"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")


ROTULO = "activity"
FEATURES_ORIGINAIS = [
    "acceleration_x", "acceleration_y", "acceleration_z",
    "gyro_x", "gyro_y", "gyro_z"
]
NOME_BASE = "occupancy_dia_"
END_BASE = ".csv"

CONJ_TESTE  = [[0], [1], [4],  [6], [7], [8], [9], [11], [2,3], [5,3], [10,3]]

"""

PASTA_ENTRADA = "datasetsNew/occ"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "Room_Occupancy_Count"
FEATURES_ORIGINAIS = ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp","S1_Light", "S2_Light", "S3_Light", "S4_Light",
                      "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound", "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR", "hour"]
NOME_BASE = "occupancy_dia_"
END_BASE = ".csv"
CONJ_TESTE = [[0,2],[0,3], [0,4], [0,5], [0,6], [1,2],[1,3], [1,4], [1,5], [1,6]]


"""

PASTA_ENTRADA = "datasetsNew/smartphone"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "label"
FEATURES_ORIGINAIS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
NOME_BASE = "user_"
END_BASE = ".csv"
CONJ_TESTE = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17], [18,19,20], [21,22,23], [24,25,26], [27,28,29]]



PASTA_ENTRADA = "datasetsNew/wisdm_phone"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "rotulo"
FEATURES_ORIGINAIS = ["x_phone_accel", "y_phone_accel", "z_phone_accel", "x_phone_gyro", "y_phone_gyro", "z_phone_gyro"]
NOME_BASE = "subject_"
END_BASE = ".csv"
CONJ_TESTE = [[0,1], [2,3], [4,5, 18], [6,7], [8,9], [10,11], [12,13],[14,15], [16,17]]

PASTA_ENTRADA = "datasetsNew/wisdm_watch"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "rotulo"
FEATURES_ORIGINAIS = ["x_watch_accel", "y_watch_accel", "z_watch_accel", "x_watch_gyro", "y_watch_gyro", "z_watch_gyro"]
NOME_BASE = "subject_"
END_BASE = ".csv"
CONJ_TESTE = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19], [20,21,22,23,24], [25,26,27,28,29], [30,31,32,33,34], [35,36,37,38,39,45], [40,41,42,43,44]]


PASTA_ENTRADA = "datasetsNew/usc_had_14"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "activity_code"
FEATURES_ORIGINAIS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
NOME_BASE = "usc_had_subject_"
END_BASE = ".csv"
#CONJ_TESTE = [[0,1],  [2,3], [4,5],  [6,7], [8,9], [10,11], [12,13]]
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]



PASTA_ENTRADA = "datasetsNew/sintetico/datasets_sinteticos"
PASTA_SAIDA = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")

ROTULO = "y"
FEATURES_ORIGINAIS = ["x1", "x2", "x3", "x4"]
NOME_BASE = "base_"
END_BASE = ".csv"
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]




PASTA_ENTRADA      = "datasetsNew/ward"
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




NOME_BASE          = "base_"
END_BASE           = ".csv"
# 10 folds — 2 sujeitos por fold (leave-two-subjects-out):
CONJ_TESTE         = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]




PASTA_ENTRADA      = "datasetsNew/recgym_wrist2"
ROTULO             = "rotulo"
FEATURES_ORIGINAIS = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']
NOME_BASE          = "sujeito_"
END_BASE           = ".csv"




# recgym_pocket:
PASTA_ENTRADA      = "datasetsNew/recgym_pocket2"
ROTULO             = "rotulo"
FEATURES_ORIGINAIS = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']
NOME_BASE          = "sujeito_"
END_BASE           = ".csv"


# recgym_leg:
PASTA_ENTRADA      = "datasetsNew/recgym_leg2"
ROTULO             = "rotulo"
FEATURES_ORIGINAIS = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']
NOME_BASE          = "sujeito_"
END_BASE           = ".csv"



PASTA_SAIDA        = os.path.join(PASTA_ENTRADA, "resultados_catch24_drift")
CONJ_TESTE = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]





PASTA_ENTRADA      = "datasetsNew/sw_sp/geotec_sp"
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
PASTA_ENTRADA      = "datasetsNew/sw_sp/geotec_sw"
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
WINDOW_MIN = 30
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
# CONFIGURAÇÃO DOS CONTROLES (registro de datasets igual ao do C2.raw)
# =========================================================
_TODOS = ["ADWIN", "PageHinkley", "KSWIN", "CUSUM", "EWMAChart",
          "GeometricMovingAverage", "HDDMAverage", "HDDMWeighted", "SEED"]
DATASET_SINTETICO = False
REUSAR_JANELA_CATCH24_OTIMIZADA = True
MANTER_APENAS_MELHOR_JANELA_CATCH24_NO_CACHE = False   # não apagar janelas do cache

# Cache próprio do controle: não toca no cache de outros experimentos na mesma
# pasta e é apagado ao final da execução (sucesso ou erro) para liberar disco.
CATCH_CACHE_DIRNAME = "catch24_cache_ewma_tmp"
DRIFT_CACHE_DIRNAME = "drift_cache_ewma_tmp"

# Sem busca: limites largos só para o FixedTrial aceitar os valores dos JSONs.
ESPACO_ATUAL = dict(RF_N_ESTIMATORS=(1, 100000), RF_MAX_DEPTH=(1, 100000), PHI_B=(1e-12, 10.0))

_W_FOLDS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]
_SPSW_FOLDS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21, 22]]
_P2 = ["temperature", "acc16_x", "acc16_y", "acc16_z", "acc6_x", "acc6_y", "acc6_z",
       "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
_W = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y"]


def _ward(loc):
    return dict(PASTA_ENTRADA="datasetsNew/ward", PASTA_ORIGINAL=f"resultados_catch24_drift_{loc}",
                ROTULO="label", FEATURES_ORIGINAIS=[f"{loc}_{c}" for c in _W],
                NOME_BASE="base_", END_BASE=".csv", CONJ_TESTE=_W_FOLDS)


def _pamap(loc):
    return dict(PASTA_ENTRADA="datasetsNew/pamap2D", PASTA_ORIGINAL=f"resultados_catch24_drift_{loc}",
                ROTULO="activityID", FEATURES_ORIGINAIS=[f"{loc}_{c}" for c in _P2],
                NOME_BASE="base_", END_BASE=".csv", CONJ_TESTE=[[i] for i in range(8)])


# Os folds (CONJ_TESTE) são conferidos contra o df_fold_counts_all.csv do C2 e
# as variáveis originais contra as monitoradas no C3 (score_<var> em
# df_features_used_all.csv); se houver diferença, o script usa as do
# experimento e avisa.
CONTROLE = {
    "dr": dict(PASTA_ENTRADA="datasetsNew/rs_default", ROTULO="rotulo",
               FEATURES_ORIGINAIS=["sdp002", "sdt002", "sdt003", "sdu002_1", "sdu002_2"],
               NOME_BASE="base", END_BASE="_final.csv",
               CONJ_TESTE=[[3], [4], [5], [0, 1], [0, 2], [0, 7], [0, 8], [6, 1], [6, 2], [6, 7], [6, 8]]),
    "gait": dict(PASTA_ENTRADA="datasetsNew/gait_default", ROTULO="label",
                 FEATURES_ORIGINAIS=["angle_1_1", "angle_1_2", "angle_1_3", "angle_2_1", "angle_2_2", "angle_2_3"],
                 NOME_BASE="base", END_BASE=".csv", CONJ_TESTE=[[i] for i in range(10)]),
    "ro": dict(PASTA_ENTRADA="datasetsNew/occ", ROTULO="Room_Occupancy_Count",
               FEATURES_ORIGINAIS=["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp", "S1_Light", "S2_Light",
                                   "S3_Light", "S4_Light", "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
                                   "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR", "hour"],
               NOME_BASE="occupancy_dia_", END_BASE=".csv",
               CONJ_TESTE=[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]),
    "mh": dict(PASTA_ENTRADA="datasetsNew/dados_mhealth", ROTULO="Activity",
               FEATURES_ORIGINAIS=["alx", "aly", "alz", "glx", "gly", "glz", "arx", "ary", "arz", "grx", "gry", "grz"],
               NOME_BASE="subject_subject", END_BASE=".csv", CONJ_TESTE=[[i] for i in range(10)]),
    "smph": dict(PASTA_ENTRADA="datasetsNew/smartphone", ROTULO="label",
                 FEATURES_ORIGINAIS=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                 NOME_BASE="user_", END_BASE=".csv",
                 CONJ_TESTE=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17],
                             [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29]]),
    "sp": dict(PASTA_ENTRADA="datasetsNew/sw_sp/geotec_sp", ROTULO="rotulo",
               FEATURES_ORIGINAIS=["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"],
               NOME_BASE="sujeito_", END_BASE=".csv", CONJ_TESTE=_SPSW_FOLDS),
    "sw": dict(PASTA_ENTRADA="datasetsNew/sw_sp/geotec_sw", ROTULO="rotulo",
               FEATURES_ORIGINAIS=["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"],
               NOME_BASE="sujeito_", END_BASE=".csv", CONJ_TESTE=_SPSW_FOLDS),
    # CONFERIR a pasta do USC (no experimento aparece "usc_had_14"; o artigo usa 7 folds)
    "usc": dict(PASTA_ENTRADA="datasetsNew/usc_had", ROTULO="activity_code",
                FEATURES_ORIGINAIS=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                NOME_BASE="usc_had_subject_", END_BASE=".csv",
                CONJ_TESTE=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
    "p2_hand": _pamap("hand"), "p2_chest": _pamap("chest"), "p2_ankle": _pamap("ankle"),
    "w_lwrist": _ward("left_wrist"), "w_rwrist": _ward("right_wrist"), "w_waist": _ward("waist"),
    "w_lankle": _ward("left_ankle"), "w_rankle": _ward("right_ankle"),
}
NOME_PASTA_ORIGINAL = "resultados_catch24_drift"
PASTA_RERUN_C3 = "resultados_catch24_drift_rerun"     # C3 de SmPh, MH e RO
for _k in ("ro", "mh", "smph"):
    CONTROLE[_k]["PASTA_C3"] = PASTA_RERUN_C3
PARAMS_RF = ["scaler_model", "rf_n_estimators", "rf_max_depth", "rf_min_samples_split",
             "rf_min_samples_leaf", "rf_max_features", "rf_bootstrap"]
SCENARIOS_TO_RUN = ["drift", "catch24_drift"]

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


def gerar_scores_a_partir_de_posicoes(num_rows, drift_indices, phi_b):
    scores = np.zeros(num_rows, dtype=np.float64)
    sem_drift = 1.0
    current_score = 0.0

    flags = np.zeros(num_rows, dtype=bool)
    if len(drift_indices) > 0:
        drift_indices = drift_indices[(drift_indices >= 0) & (drift_indices < num_rows)]
        flags[drift_indices] = True

    for linha in range(num_rows):
        flag = flags[linha]
        if flag:
            phi = phi_b * math.log(1 + sem_drift)
            if phi > 1:
                phi = 1
            score_ant = scores[linha - 1] if linha > 1 else 0.0
            current_score = (1 + 1 / math.log(1 + sem_drift)) + (1 - phi) * score_ant
            sem_drift = 0
        else:
            sem_drift += 1
            phi = phi_b * math.log(1 + sem_drift)
            if phi > 1:
                phi = 1
            score_ant = scores[linha - 1] if linha > 1 else 0.0
            current_score = (1 - phi) * score_ant
        scores[linha] = current_score
    return scores


# Controle ativo: "EWMA_ALARM" ou "EWMA_VAR". O phi_b recebido do objetivo e o
# lambda do controle (passado no FixedTrial no lugar do phi_b).
CONTROLE_ATIVO: Optional[str] = None
# Scalers dos detectores ajustados no treino de cada fold: (id_fold, scaler_name) -> scaler
_SCALERS_FOLD: Dict[Tuple[int, str], object] = {}


def ewma_alarmes(flags: np.ndarray, lam: float) -> np.ndarray:
    """S_t = (1 - lam) S_{t-1} + lam a_t, com S_{-1} = 0."""
    from scipy.signal import lfilter
    return lfilter([lam], [1.0, -(1.0 - lam)], flags.astype(np.float64), zi=[0.0])[0]


def ewma_var(x: np.ndarray, lam: float) -> np.ndarray:
    """v_t = (1-lam) v_{t-1} + lam (x_t - m_{t-1})^2, m_t = (1-lam) m_{t-1} + lam x_t,
    com m_{-1} = 0 e v_{-1} = 1 (sinal normalizado). Causal."""
    from scipy.signal import lfilter
    a = 1.0 - lam
    m = lfilter([lam], [1.0, -a], x, zi=[0.0])[0]
    m_ant = np.r_[0.0, m[:-1]]
    return lfilter([lam], [1.0, -a], (x - m_ant) ** 2, zi=[a * 1.0])[0]


def get_drift_enriched_df(item, detector_name, scaler_name, phi_b, id_fold, split_name, pasta_saida: str):
    """Mesma interface do experimento: acrescenta score_<var> a cada variavel
    original. Aqui o score e o do controle ativo, com lambda = phi_b."""
    df_original = item["df"].copy()
    num_rows = len(df_original)
    lam = float(phi_b)
    if CONTROLE_ATIVO == "EWMA_VAR":
        scaler = _SCALERS_FOLD[(id_fold, scaler_name)]
        x = scaler.transform(df_original[FEATURES_ORIGINAIS])
        for j, col in enumerate(FEATURES_ORIGINAIS):
            df_original[f"score_{col}"] = ewma_var(np.asarray(x[:, j], dtype=np.float64), lam)
        return df_original
    if CONTROLE_ATIVO == "EWMA_ALARM":
        base_dir = get_drift_cache_dir(pasta_saida)
        csv_path = get_drift_csv_path(base_dir=base_dir, detector_name=detector_name, scaler_name=scaler_name,
                                      id_fold=id_fold, split_name=split_name, nome_arquivo=item["nome_arquivo"])
        drift_map = carregar_posicoes_drift(csv_path)
        for col in FEATURES_ORIGINAIS:
            flags = np.zeros(num_rows, dtype=bool)
            idx = drift_map[col]
            idx = idx[(idx >= 0) & (idx < num_rows)]
            flags[idx] = True
            df_original[f"score_{col}"] = ewma_alarmes(flags, lam)
        return df_original
    raise ValueError(f"Controle desconhecido: {CONTROLE_ATIVO}")


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

    if scenario_name in {"catch24", "catch24_drift", "catch24_raw"}:
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

        if scenario_name == "catch24_raw":
            # Controle C2.raw: as mesmas variáveis originais alinhadas do C3,
            # sem as drift scores (mesmo descarte das window_size - 1 linhas iniciais).
            start = window_size - 1
            X_raw = df_base.iloc[start:].reset_index(drop=True)[FEATURES_ORIGINAIS]
            if len(X_raw) != len(X_catch):
                min_len = min(len(X_raw), len(X_catch))
                X_raw = X_raw.iloc[:min_len].reset_index(drop=True)
                X_catch = X_catch.iloc[:min_len].reset_index(drop=True)
                y_catch = y_catch[:min_len]
            X = pd.concat([X_raw.reset_index(drop=True), X_catch.reset_index(drop=True)], axis=1)
            return X, y_catch

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
            n_estimators=trial.suggest_int("rf_n_estimators", *ESPACO_ATUAL["RF_N_ESTIMATORS"]),
            max_depth=trial.suggest_int("rf_max_depth", *ESPACO_ATUAL["RF_MAX_DEPTH"]),
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
        elif scenario_name in {"catch24_drift", "catch24_raw"}:
            if REUSAR_JANELA_CATCH24_OTIMIZADA:
                if fixed_catch24_window_size is None:
                    raise ValueError(f"{scenario_name} requer a janela do catch24 fixada.")
                window_size = int(fixed_catch24_window_size)
                trial.set_user_attr("fixed_catch24_window_size", window_size)
            else:
                window_size = trial.suggest_int("window_size", WINDOW_MIN, WINDOW_MAX, step=WINDOW_STEP)

        if scenario_name in {"drift", "catch24_drift"}:
            phi_b = trial.suggest_float("phi_b", *ESPACO_ATUAL["PHI_B"], log=True)
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


# =========================================================
# REEXECUÇÃO DE C1 E C3
# =========================================================
def salvar_todos_trials(study, pasta_scenario: str):
    """Salva todos os trials (inclusive podados) e o F1 por fold dos completos."""
    linhas, por_fold = [], []
    for t in study.trials:
        linhas.append({"trial": t.number, "estado": t.state.name,
                       "valor": t.value if t.value is not None else np.nan,
                       "duracao_s": t.duration.total_seconds() if t.duration else np.nan,
                       **t.params})
        if t.state == optuna.trial.TrialState.COMPLETE:
            f1 = t.user_attrs.get("macro_f1", {})
            acc = t.user_attrs.get("acc", {})
            for i, fold in enumerate(f1.get("fold", [])):
                por_fold.append({"trial": t.number, "fold": fold,
                                 "test_bases": f1.get("test_bases", [None] * (i + 1))[i],
                                 "f1_macro": f1["RF"][i],
                                 "acc": acc.get("RF", [np.nan] * (i + 1))[i],
                                 **t.params})
    pd.DataFrame(linhas).to_csv(os.path.join(pasta_scenario, "trials_todos.csv"), index=False)
    pd.DataFrame(por_fold).to_csv(os.path.join(pasta_scenario, "trials_por_fold.csv"), index=False)




# =========================================================
# AUXILIARES (iguais ao controle C2.raw)
# =========================================================
def ler_json(caminho):
    if not os.path.exists(caminho):
        return None
    with open(caminho) as f:
        return json.load(f)


def conferir_folds_e_features(pasta_c2, pasta_c3):
    """Folds (test_bases) usados no C2 e variáveis originais monitoradas no C3."""
    folds = feats = None
    p = os.path.join(pasta_c2, "df_fold_counts_all.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        if "run" in d and d["run"].nunique() > 1:
            d = d[d["run"] == d["run"].iloc[-1]]
        if "test_bases" in d:
            folds = [json.loads(x) for x in d.sort_values("fold")["test_bases"]]
    for det in _TODOS:
        p = os.path.join(pasta_c3, det, "df_features_used_all.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if not d.empty and "features_union" in d:
                cols = json.loads(d["features_union"].iloc[-1])
                feats = [c[len("score_"):] for c in cols if c.startswith("score_")]
                break
    return folds, feats


def salvar_controle(trial, valor, pasta, run_name, info):
    os.makedirs(pasta, exist_ok=True)
    ua = trial.user_attrs
    f1 = pd.DataFrame(ua["macro_f1"]); f1.insert(0, "run", run_name)
    f1.to_csv(os.path.join(pasta, "df_f1_all.csv"), index=False)
    acc = pd.DataFrame(ua["acc"]); acc.insert(0, "run", run_name)
    acc.to_csv(os.path.join(pasta, "df_acc_all.csv"), index=False)
    cls = dict(ua["f1_per_class_summary"]); cls["run"] = run_name
    pd.DataFrame([cls]).to_csv(os.path.join(pasta, "df_f1_all_classe.csv"), index=False)
    fc = pd.DataFrame(ua["fold_counts"]); fc.insert(0, "run", run_name)
    fc.to_csv(os.path.join(pasta, "df_fold_counts_all.csv"), index=False)
    with open(os.path.join(pasta, "resultado_controle.json"), "w") as f:
        json.dump(to_python_types({**info, "macro_f1": float(valor)}), f, indent=4)



# =========================================================
# CONTROLES EWMA
# =========================================================
FATORES_LAMBDA = [0.5, 1.0, 2.0]
CONTROLES = ["EWMA_ALARM", "EWMA_VAR"]
CENARIOS = {"c1": ("C1", "drift"), "c3": ("C3", "catch24_drift")}
PARAMS_DRIFT = ["phi_b", "scaler_detectors"]


def horizonte_algorithm1(phi_b: float) -> int:
    """Menor h com sum_{k=1..h} -ln(1 - min(1, phi_b ln(1+k))) >= 1 (score cai a
    1/e sem alarmes). Para phi_b pequeno, -ln(1 - x) ~ x e a soma ~ phi_b ln((h+1)!)."""
    from scipy.special import gammaln
    phi_b = float(phi_b)
    if phi_b >= 1e-4:                       # h pequeno: soma exata
        s, h = 0.0, 0
        while s < 1.0:
            h += 1
            f = min(1.0, phi_b * math.log1p(h))
            if f >= 1.0:
                return h
            s += -math.log1p(-f)
        return h
    lo, hi = 1, 10 ** 15                    # busca binaria em phi_b * ln((h+1)!) >= 1
    while lo < hi:
        mid = (lo + hi) // 2
        if phi_b * gammaln(mid + 2) >= 1.0:
            hi = mid
        else:
            lo = mid + 1
    return lo


def ler_json_cenario(pasta_entrada, cfg, cenario_dir, det):
    """JSON do detector no cenario, seguindo a mesma regra do C2.raw: pasta de
    reexecucao (PASTA_C3) quando existir; senao, a pasta original."""
    candidatas = []
    if "PASTA_C3" in cfg:
        candidatas.append(os.path.join(pasta_entrada, cfg["PASTA_C3"]))
    candidatas.append(os.path.join(pasta_entrada, cfg.get("PASTA_ORIGINAL", NOME_PASTA_ORIGINAL)))
    for k, pasta in enumerate(candidatas):
        caminho = os.path.join(pasta, cenario_dir, det, "resultado_otimizacao.json")
        js = ler_json(caminho)
        if js is not None:
            if k > 0 and len(candidatas) > 1:
                print(f"[AVISO] {cenario_dir}/{det}: JSON nao encontrado na pasta de reexecucao; "
                      f"usando {caminho}", flush=True)
            return caminho, js
    return None, None


def melhor_detector(pasta_entrada, cfg, cenario_dir):
    melhores = []
    for det in _TODOS:
        caminho, js = ler_json_cenario(pasta_entrada, cfg, cenario_dir, det)
        if js is not None:
            melhores.append((float(js["best_macro_f1"]), det, caminho, js))
    if not melhores:
        return None
    for f1, det, _, _ in sorted(melhores, reverse=True):
        print(f"    {cenario_dir:14s} {det:24s} F1={f1:.6f}", flush=True)
    return max(melhores, key=lambda x: x[0])


def ajustar_scalers_dos_folds(dfs_info, combinacoes, scaler_name):
    """Scaler dos detectores ajustado no treino de cada fold, como no experimento."""
    for id_fold, comb in enumerate(combinacoes):
        if (id_fold, scaler_name) in _SCALERS_FOLD:
            continue
        treino = pd.concat([dfs_info[i]["df"] for i in comb["treinamento"]], ignore_index=True)
        sc = build_scaler_from_name(scaler_name)
        sc.fit(treino[FEATURES_ORIGINAIS])
        _SCALERS_FOLD[(id_fold, scaler_name)] = sc


def precompute_alarmes(dfs_info, combinacoes, pasta_saida, detector_name, scaler_name):
    """Alarmes do detector (um scaler, todos os folds), no cache de drift."""
    ajustar_scalers_dos_folds(dfs_info, combinacoes, scaler_name)
    base_dir = get_drift_cache_dir(pasta_saida)
    print(f"[INFO] Alarmes: {detector_name} | scaler {scaler_name} | {len(combinacoes)} folds", flush=True)
    for id_fold, comb in enumerate(combinacoes):
        sc = _SCALERS_FOLD[(id_fold, scaler_name)]
        for split_name, ids in [("train", comb["treinamento"]), ("test", comb["teste"])]:
            infos = [dfs_info[i] for i in ids]
            if Parallel is None or delayed is None or N_JOBS_DRIFT <= 1:
                for item in infos:
                    process_single_base_drift(item, detector_name, sc, scaler_name, id_fold, split_name, base_dir)
            else:
                Parallel(n_jobs=min(N_JOBS_DRIFT, len(infos)), prefer="processes")(
                    delayed(process_single_base_drift)(item, detector_name, sc, scaler_name, id_fold,
                                                       split_name, base_dir)
                    for item in infos
                )


def avaliar_controle(dfs_info, combinacoes, pasta_cache, cenario_exp, detector, params, janela):
    trial = optuna.trial.FixedTrial(params)
    obj = make_objective(dfs_info, combinacoes, pasta_cache, cenario_exp, detector,
                         fixed_catch24_window_size=janela)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)    # FixedTrial fora dos limites (nao se aplica)
        valor = obj(trial)
    return trial, valor


def run_controle(nome, cenarios, controles):
    global PASTA_ENTRADA, PASTA_SAIDA, ROTULO, FEATURES_ORIGINAIS, NOME_BASE, END_BASE, CONJ_TESTE, CONTROLE_ATIVO
    cfg = CONTROLE[nome]
    PASTA_ENTRADA = cfg["PASTA_ENTRADA"]
    ROTULO = cfg["ROTULO"]
    FEATURES_ORIGINAIS = list(cfg["FEATURES_ORIGINAIS"])
    NOME_BASE, END_BASE = cfg["NOME_BASE"], cfg["END_BASE"]
    CONJ_TESTE = cfg["CONJ_TESTE"]
    pasta_original = os.path.join(PASTA_ENTRADA, cfg.get("PASTA_ORIGINAL", NOME_PASTA_ORIGINAL))
    pasta_c3 = os.path.join(PASTA_ENTRADA, cfg.get("PASTA_C3", cfg.get("PASTA_ORIGINAL", NOME_PASTA_ORIGINAL)))
    PASTA_SAIDA = pasta_original
    pasta_out_base = os.path.join(pasta_original, "controle_ewma")
    run_name = f"controle_ewma_{nome}"

    folds, feats = conferir_folds_e_features(os.path.join(pasta_original, "catch24"),
                                             os.path.join(pasta_c3, "catch24_drift"))
    if folds is not None and folds != CONJ_TESTE:
        print(f"[AVISO] CONJ_TESTE difere do usado no C2; usando o do C2: {folds}", flush=True)
        CONJ_TESTE = folds
    if feats is not None and set(feats) != set(FEATURES_ORIGINAIS):
        print(f"[AVISO] variaveis originais diferem das monitoradas no C3; usando as do C3: {feats}", flush=True)
        FEATURES_ORIGINAIS = feats

    # janela do catch22: a do C2 (herdada pelo C3)
    janela = None
    if "c3" in cenarios:
        js_c2_path = os.path.join(pasta_original, "catch24", "resultado_otimizacao.json")
        js_c2 = ler_json(js_c2_path)
        if js_c2 is None or "window_size" not in js_c2.get("best_params", {}):
            sys.exit(f"[ERRO] JSON do C2 (com window_size) nao encontrado: {js_c2_path}")
        janela = int(js_c2["best_params"]["window_size"])

    # melhor detector de cada cenario
    tarefas = []
    for cen in cenarios:
        rot, cen_dir = CENARIOS[cen]
        print(f"[INFO] {rot}: melhor detector em {cen_dir}", flush=True)
        m = melhor_detector(PASTA_ENTRADA, cfg, cen_dir)
        if m is None:
            print(f"[ERRO] nenhum JSON de {cen_dir}; {rot} pulado", flush=True)
            continue
        f1_ref, det, caminho, js = m
        bp = js["best_params"]
        faltam = [k for k in PARAMS_RF + PARAMS_DRIFT if k not in bp]
        if faltam:
            print(f"[ERRO] {caminho} sem {faltam}; {rot} pulado", flush=True)
            continue
        if cen == "c3":
            jan_c3 = js.get("fixed_catch24_window_size")
            if jan_c3 is not None and int(jan_c3) != janela:
                print(f"[AVISO] janela do C3 ({jan_c3}) difere da do C2 ({janela}); usando a do C2", flush=True)
        for ctl in controles:
            pasta_out = os.path.join(pasta_out_base, f"{rot}_{ctl}")
            if os.path.exists(os.path.join(pasta_out, "resultado_controle.json")):
                print(f"[SKIP] {rot} {ctl}: ja concluido em {pasta_out}", flush=True)
                continue
            tarefas.append(dict(cen=cen, rot=rot, cen_dir=cen_dir, det=det, caminho=caminho, bp=bp,
                                f1_ref=f1_ref, controle=ctl, pasta_out=pasta_out))
    if not tarefas:
        print("[INFO] nada a executar (tudo concluido ou sem JSONs)", flush=True)
        escrever_resumo(pasta_out_base)
        return

    print(f"Carregando dados: {PASTA_ENTRADA} | resultados e cache: {pasta_original}", flush=True)
    dfs_info = load_dados(PASTA_ENTRADA)
    combinacoes = build_combinacoes(len(dfs_info))
    print(f"[INFO] {len(dfs_info)} blocos, {len(combinacoes)} folds, {len(FEATURES_ORIGINAIS)} variaveis originais"
          + (f", janela do catch22 = {janela}" if janela else ""), flush=True)
    try:
        if any(t["cen"] == "c3" for t in tarefas):
            precompute_best_catch24_window_for_all_bases(dfs_info, pasta_original, janela)
        for t in tarefas:
            CONTROLE_ATIVO = t["controle"]
            bp, det = t["bp"], t["det"]
            phi_b = float(bp["phi_b"])
            h = horizonte_algorithm1(phi_b)
            lam0 = 1.0 / h
            if CONTROLE_ATIVO == "EWMA_ALARM":
                precompute_alarmes(dfs_info, combinacoes, pasta_original, det, bp["scaler_detectors"])
            else:
                ajustar_scalers_dos_folds(dfs_info, combinacoes, bp["scaler_detectors"])
            print(f"\n===== {t['rot']} {CONTROLE_ATIVO} | melhor detector {det} (F1 {t['f1_ref']:.6f}) | "
                  f"phi_b={phi_b:.3e} -> h={h} -> lambda_0={lam0:.3e} =====", flush=True)
            grade, melhor = [], None
            for fator in FATORES_LAMBDA:
                lam = min(1.0, fator * lam0)
                params = {k: bp[k] for k in PARAMS_RF}
                params["phi_b"] = lam                            # lambda no lugar do phi_b
                params["scaler_detectors"] = bp["scaler_detectors"]
                trial, valor = avaliar_controle(dfs_info, combinacoes, pasta_original, t["cen_dir"], det,
                                                params, janela if t["cen"] == "c3" else None)
                print(f"    lambda = {fator} x lambda_0 = {lam:.3e}: F1 = {valor:.6f}", flush=True)
                grade.append(dict(fator=fator, lam=lam, macro_f1=valor))
                if melhor is None or valor > melhor[1]:
                    melhor = (trial, valor, fator, lam)
                clear_runtime_caches()
            trial, valor, fator, lam = melhor
            info = dict(cenario=t["rot"], controle=CONTROLE_ATIVO, detector_referencia=det,
                        fonte_hiperparametros=t["caminho"], phi_b=phi_b, horizonte_h=h, lambda_0=lam0,
                        fator_escolhido=fator, lambda_escolhido=lam, janela_catch24=janela,
                        params_rf={k: bp[k] for k in PARAMS_RF}, scaler_detectors=bp["scaler_detectors"],
                        f1_algorithm1=t["f1_ref"], protocolo="nao aninhado (mesmo do experimento original)")
            salvar_controle(trial, valor, t["pasta_out"], run_name, info)
            pd.DataFrame(grade).to_csv(os.path.join(t["pasta_out"], "grade_lambda.csv"), index=False)
            print(f"[RESULTADO] {t['rot']} {CONTROLE_ATIVO}={valor:.6f} (lambda={lam:.3e}) | Algorithm 1 ({det})="
                  f"{t['f1_ref']:.6f} | Alg.1 - controle = {100 * (t['f1_ref'] - valor):+.2f} p.p.", flush=True)
    finally:
        clear_runtime_caches()
        remove_temp_dirs(pasta_original)
    escrever_resumo(pasta_out_base)


def escrever_resumo(pasta_out_base):
    linhas = []
    if os.path.isdir(pasta_out_base):
        for d in sorted(os.listdir(pasta_out_base)):
            js = ler_json(os.path.join(pasta_out_base, d, "resultado_controle.json"))
            if js is None:
                continue
            linhas.append(dict(cenario=js["cenario"], controle=js["controle"],
                               detector_referencia=js["detector_referencia"], phi_b=js["phi_b"],
                               horizonte_h=js["horizonte_h"], lambda_escolhido=js["lambda_escolhido"],
                               fator=js["fator_escolhido"], f1_controle=js["macro_f1"],
                               f1_algorithm1=js["f1_algorithm1"],
                               alg1_menos_controle_pp=100 * (js["f1_algorithm1"] - js["macro_f1"])))
    if linhas:
        d = pd.DataFrame(linhas)
        caminho = os.path.join(pasta_out_base, "resumo_controle_ewma.csv")
        d.to_csv(caminho, index=False)
        print("\n===== Resumo dos controles EWMA =====")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"Resumo salvo em: {caminho}")


def remove_temp_dirs(pasta_saida: str):
    """Apaga os diretorios de cache do controle (catch22 e drift)."""
    for dirname in [CATCH_CACHE_DIRNAME, DRIFT_CACHE_DIRNAME]:
        d = os.path.join(pasta_saida, dirname)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"[OK] Cache removida: {d}", flush=True)


def clear_runtime_caches():
    _CATCH_CACHE.clear()
    _COLS_CACHE.clear()


def configurar_paralelismo():
    """Usa os nucleos do job SLURM (SLURM_CPUS_PER_TASK) no RF, no catch22 e nos detectores."""
    global N_JOBS_RF, N_JOBS_CATCH24, N_JOBS_DRIFT
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    N_JOBS_RF = n
    N_JOBS_CATCH24 = n
    # Cada processo dos detectores reimporta o script, e com ele capymoa, torch e
    # a JVM (JPype). Muitos processos iniciando juntos abortaram (SIGABRT) nos jobs
    # grandes; o experimento original usava 4.
    N_JOBS_DRIFT = min(n, 4)
    print(f"[INFO] Nucleos: {n} (RF e catch22), {N_JOBS_DRIFT} processos para os detectores", flush=True)


def resumo_geral(saida="resumo_controle_ewma_todos.csv"):
    """Junta os resumos de todos os datasets (rodar na pasta de trabalho, apos os jobs)."""
    partes = []
    for nome, cfg in CONTROLE.items():
        pasta = os.path.join(cfg["PASTA_ENTRADA"], cfg.get("PASTA_ORIGINAL", NOME_PASTA_ORIGINAL), "controle_ewma")
        escrever_resumo(pasta)
        p = os.path.join(pasta, "resumo_controle_ewma.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d.insert(0, "dataset", nome)
            partes.append(d)
        else:
            print(f"[FALTA] {nome}: nenhum controle concluido", flush=True)
    if partes:
        d = pd.concat(partes, ignore_index=True)
        d.to_csv(saida, index=False)
        print(f"\n===== Todos os datasets =====\n{d.to_string(index=False, float_format=lambda v: f'{v:.4g}')}\nSalvo em: {saida}")


def main():
    global CATCH_CACHE_DIRNAME, DRIFT_CACHE_DIRNAME
    args = sys.argv[1:]
    if args and args[0] == "resumo":
        resumo_geral()
        return
    if not args or args[0] not in CONTROLE:
        sys.exit(f"Uso: python {os.path.basename(sys.argv[0])} <{'|'.join(CONTROLE)}> [c1|c3|ambos] "
                 f"[EWMA_ALARM EWMA_VAR]\n     python {os.path.basename(sys.argv[0])} resumo")
    nome = args[0]
    cen_arg = args[1] if len(args) > 1 and args[1] in ("c1", "c3", "ambos") else "ambos"
    controles = [a for a in args[1:] if a in CONTROLES] or list(CONTROLES)
    cenarios = ["c1", "c3"] if cen_arg == "ambos" else [cen_arg]
    # cache proprio de cada execucao: jobs do mesmo dataset (c1 e c3) rodam em
    # paralelo sem que um apague o cache do outro ao terminar
    tag = "_".join([cen_arg] + [c.split("_")[1].lower() for c in controles])
    CATCH_CACHE_DIRNAME = f"catch24_cache_ewma_{tag}_tmp"
    DRIFT_CACHE_DIRNAME = f"drift_cache_ewma_{tag}_tmp"
    configurar_paralelismo()
    clear_runtime_caches()
    try:
        run_controle(nome, cenarios, controles)
        print("\nProcesso concluido com sucesso.")
    finally:
        clear_runtime_caches()


if __name__ == "__main__":
    main()