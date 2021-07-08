#!/bin/bash

#SBATCH --job-name=cnn_hello_world

#SBATCH --output=/home/hu/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --partition=lopri

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --gres=gpu:1

#SBATCH --time=01:00:00

#SBATCH --signal=TERM@120

WS_DIR="$HOME/Projects/Safe_ML"
SCRIPT="train.py"

cd ${WS_DIR}

# wandb on

srun --unbuffered python ${SCRIPT}
