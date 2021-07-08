#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=cnn_hello_world

#SBATCH --output=/home/kq708907/slurm_logs/%J_%x.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=huzjkevin@gmail.com

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:volta:1

#SBATCH --time=0-01:00:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c18g

#SBATCH --account=rwth0485

#SBATCH --array=1-5



source $HOME/.zshrc
conda activate kevin

WS_DIR="$HOME/Safe_ML"
SCRIPT="train.py"

cd ${WS_DIR}

#file=`ls SPA_cfgs/SPA_*.yaml | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

srun --unbuffered python ${SCRIPT}