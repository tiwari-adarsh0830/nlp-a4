#!/bin/bash
#SBATCH --job-name=a4p2
#SBATCH --account=REPLACE_WITH_my_slurm_accounts_OUTPUT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/nlp_a4/logs/%x_%j.out

# Usage (from anywhere on Torch):
#   sbatch /scratch/$USER/nlp_a4/repo/hpc/part2.sbatch <args for train_t5.py>
# Everything after the script name is passed straight to python3 train_t5.py.

module purge
cd /scratch/$USER/nlp_a4/repo/part2

singularity exec --nv \
  --overlay /scratch/$USER/nlp_a4/a4.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate a4p2; nvidia-smi -L; python3 train_t5.py $*"
