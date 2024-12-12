# CSTC 210 Final Project: Music Video Generator

## Project Description
Music videos are maybe the ideal music visualizers; they are artfully crafted to provide interesting visuals that go along with the music. However, they can't be auto-generated in real time. My aim is to make a model that generates artificial music videos from live audio input, using RAVE as a feature extractor for the audio.

I train a custom decoder model to recreate image frames from their accompanying audio chunks in a music video. It would be difficult to do this with raw waveform data, so that's where RAVE comes in: I pretrain RAVE on five hours of hip hop mixes, so that I can intercept its latent encodings to use as input to my custom decoder model. So the pipeline is audio sample -> pretrained RAVE model (encoder portion) -> latent vector -> custom decoder model -> 256x256 image frame.

## Data
The data for this project is a collection of music videos from YouTube. I used the `yt-dlp` package to download the videos and `ffmpeg` to extract the audio. In `process_video.py`, the audio is then split into `.wav` chunks using the `librosa` library, the image frames are extracted using `opencv-python`, and the audio chunks are encoded into `.npy` files using the RAVE model.

## RAVE
RAVE or Realtime Audio Variational autoEncoder is a model for synthesizing audio by compressing it into a latent space and then expanding it back out. I'm using it as a feature extractor for the audio in the music videos. The model is trained on the High-Performance Computing Cluster at Yale University on five hours of hip hop mixes. The results are saved in `trainings/`, and the exported model is saved in `exports/`.

## Custom Decoder Model
The custom decoder model is a simple fully connected neural network in `latent-to-image.py` that expands the latent vector to a shape that can be fed into a series of deconvolutional layers. The deconvolutional layers then generate the image. The model is trained on the latent vectors extracted from a pretrained RAVE model and the corresponding image frames, as prepared in `process_video.py`.

## Pipeline
The pipeline for this project is as follows:
1. Download hip hop mixes from YouTube, e.g. `yt-dlp https://www.youtube.com/watch?v=IvXE-aDGOaU`, and put them in `datasets/hiphop`.
2. Extract audio from videos using `ffmpeg -i "xxxxx.webm" -vn "xxxxx.wav"` in the `datasets/hiphop` directory.
3. On the cluster, create a conda environment with the necessary RAVE dependencies:
```
salloc -p education_gpu -t 2:00:00 --gpus=1
module load miniconda
conda create -n RAVE python=3.9
conda activate RAVE
conda install ffmpeg sox -y
pip install acids-rave
```
4. Clone RAVE from GitHub with `git clone https://github.com/acids-ircam/RAVE.git`.
5. Preprocess audio using RAVE with `rave preprocess --input_path datasets/hiphop --output_path preprocessed/hiphop`.
6. Start an HPC slurm job to train RAVE on the preprocessed audio with `sbatch RAVE-training-batch.sh`.
7. After training, export the RAVE model with `rave export --name hiphop --run trainings/hiphop/hiphop_43ef472243/ --streaming True --output exports/`.
8. For the custom decoder model, make a new conda environment with the necessary dependencies on HPC for CUDA support:
```
conda create -n RAVE2 python=3.11
conda activate RAVE2
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install librosa opencv-python pygame torchvision tqdm pillow
```
9. Process data with `python process_video.py` to split audio into chunks using `librosa`, extract image frames with `python-opencv`, and encode audio chunks into latent vectors using the exported RAVE model.
10. Train custom decoder model on frames and latent vectors with `python latent-to-image.py`. This will save as a `.pth` file and generate images displayed in a live `pygame` window from audio input.

## Future Work
The next steps for this project are to:
- Train the custom decoder model longer and on more data to improve the quality of the generated images. I just trained it on my local computer on Kendrick Lamar's "Alright" music video, but I ran into memory limits on the HPC (because I designed the dataloader to load everything into memory, which won't scale to larger training data). Also, "Alright" is all black and white, so I may want to use a more colorful music video.
- Train RAVE longer on the hip hop mixes to improve the quality of the latent vectors. I trained it for several hours, but when I ran into those memory issues from the custom decoder model, the RAVE training was cut short.
- Refactor the code to be more modular and scalable. I have a lot of hard-coded paths and parameters in my Python scripts that could be abstracted into a config file or command line arguments. It's just that the pipeline makes things complicated with all the different steps and dependencies. For example, training RAVE requires a different conda environment than the custom decoder model.