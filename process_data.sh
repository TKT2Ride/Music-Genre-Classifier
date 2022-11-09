#!/bin/bash

#SBATCH --job-name="Music Preprocessing"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=trant@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4

SCRIPT_NAME="Music Preprocessing"
PYTHON_FILE="/home/trant/CS4981/Music-Genre-Classifier/data_preprocessing.py"
ENV="music-classifier"

## SCRIPT START
echo "SBATCH SCRIPT: ${SCRIPT_NAME}"
srun python3 ${PYTHON_FILE}
echo "END: ${SCRIPT_NAME}"

## SCRIPT END