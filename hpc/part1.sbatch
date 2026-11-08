#!/bin/bash
#SBATCH --job-name=a4p1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/nlp_a4/logs/%x_%j.out

# Usage (from anywhere on Torch):
#   sbatch /scratch/$USER/nlp_a4/repo/hpc/part1.sbatch <args for main.py>
# Needs SBATCH_ACCOUNT set in your Torch ~/.bashrc (value from my_slurm_accounts).
# Everything after the script name is passed straight to python3 main.py.

module purge
cd /scratch/$USER/nlp_a4/repo/part1

singularity exec --nv \
  --overlay /scratch/$USER/nlp_a4/a4.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; nvidia-smi -L; /ext3/miniforge3/envs/a4p1/bin/python3 main.py $*"
