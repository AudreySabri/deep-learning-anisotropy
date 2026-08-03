#!/bin/bash  
#SBATCH --job-name=_BNN_PRED_
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --output=_bnn_pred_results_%j.out
#SBATCH --error=_bnn_pred_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G 

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpsc-hill-ml

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1 

python train.py
