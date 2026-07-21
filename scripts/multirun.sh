#!/bin/bash  
#SBATCH --job-name=MLP_HILL_HPO
#SBATCH --partition=cpu-dedicated
#SBATCH --qos=dedicated
#SBATCH --output=_mlp_hpo_results_%j.out
#SBATCH --error=_mlp_hpo_error_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G 

module purge

# Hyperparameter config:
#HYPERPARAMETERS="model.hidden_dim=choice(21,42,84) trainer.max_epochs=choice(100,150,200)"
HYPERPARAMETERS="model.net.hidden_dim=choice(42,84)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpsc-hill-ml

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1 
# the hyperparam search below will spawn nodes and report to mlflow automatically
python mlp_train.py --multirun $HYPERPARAMETERS &
wait
