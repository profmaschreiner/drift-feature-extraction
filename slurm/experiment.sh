#!/bin/bash

#SBATCH --job-name=t_mh
#SBATCH --partition=cpu          # PARTIÇÃO CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1               # 1 processo
#SBATCH --cpus-per-task=16       # ajuste conforme necessário
#SBATCH --mem=16G                # ajuste conforme necessário
#SBATCH --time=10-00:00:00   # 10 dias
#SBATCH --output=log/log_%j.out
#SBATCH --error=log/log_%j.err

# ======================================================================
# Inicializa o ambiente do shell para o conda
# ======================================================================
echo "Inicializando o ambiente do shell..."
source ~/miniconda3/etc/profile.d/conda.sh
echo "Ambiente inicializado."

# ======================================================================
# FORÇA EXECUÇÃO NA CPU (mesmo se houver GPU disponível)
# ======================================================================
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

# Ajuste de threads para bater com --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "CPUs disponíveis: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"

# ======================================================================
# Ativa ambiente Conda
# ======================================================================
echo "Ativando o ambiente Conda: capyRiver"
conda activate capyRiver
echo "Ambiente ativado com sucesso."

# Debug útil
echo "Caminho do Python:"
which python
python --version

# ======================================================================
# Executa o experimento
# ======================================================================
echo "Iniciando o script Python..."
python3 experimento.py  
echo "Script Python finalizado."


