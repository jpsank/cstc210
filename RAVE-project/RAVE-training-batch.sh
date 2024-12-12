#!/bin/bash

#SBATCH -J rave-train
#SBATCH -p education_gpu
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL

module purge
module load miniconda
conda activate RAVE
python RAVE/scripts/train.py --name hiphop --db_path preprocessed/hiphop/ \
	--out_path trainings/hiphop --config v2_small --config noise \
	--config wasserstein --augment compress --augment gain \
	--channels 1 --save_every 100000 --workers 1
