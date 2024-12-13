#!/bin/bash

#SBATCH -J decoder-train
#SBATCH -p education_gpu
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

python_path="$HOME/palmer_scratch/envs/RAVE4/bin/python"  # Change this to your environment's python path
$python_path preprocess_video.py datasets/kdot/kdot.mov datasets/kdot/kdot.wav exports/hiphop_streaming.ts datasets/kdot
$python_path preprocess_video.py datasets/allmylife/allmylife.mp4 datasets/allmylife/allmylife.wav exports/hiphop_streaming.ts datasets/allmylife
$python_path decoder.py train datasets/kdot/latents datasets/kdot/frames --epochs 80
$python_path decoder.py train datasets/allmylife/latents datasets/allmylife/frames --epochs 20 --resume

$python_path decoder.py demo decoder_model.pth exports/hiphop_streaming.ts --reproduce datasets/kdot/kdot.wav