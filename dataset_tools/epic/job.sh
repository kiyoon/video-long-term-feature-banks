#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Hannah
#SBATCH --gres=gpu:0
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-00:00:00
#SBATCH --job-name=extract

export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/bin/miniconda3/bin/activate
bash ./extract_epic_frames.sh "test/P$(printf '%02d' ${SLURM_ARRAY_TASK_ID})"
