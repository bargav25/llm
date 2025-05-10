#!/bin/bash -l

# Set SCC project
#$ -P dnn-motion

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0

#$ -l h_rt=32:00:00

#$ -v WANDB_API_KEY

module load miniconda


conda activate llm_env

python scripts/train.py --config configs/train_config.yaml --use_wandb