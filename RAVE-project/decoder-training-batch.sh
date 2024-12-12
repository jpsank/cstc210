#!/bin/bash

#SBATCH -J rave-train
#SBATCH -p education_gpu
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL

module purge
module load miniconda
conda activate ~/palmer_scratch/envs/RAVE4
python process_video.py
python latent-to-image.py
