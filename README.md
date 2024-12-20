# CSTC 210 Final Project: Music Video Generator

## Project Description
Music videos are maybe the ideal music visualizers; they are artfully crafted to provide interesting visuals that go along with the music. However, they can't be auto-generated in real time. My aim is to make a model that generates artificial music videos from live audio input, using RAVE as a feature extractor for the audio.

I train a custom decoder model to recreate image frames from their accompanying audio chunks in a music video. It would be difficult to do this with raw waveform data, so that's where RAVE comes in: I pretrain RAVE on five hours of hip hop mixes, so that I can intercept its latent encodings to use as input to my custom decoder model. So the pipeline is audio sample -> pretrained RAVE model (encoder portion) -> latent vector -> custom decoder model -> 256x256 image frame.

## Data
The data for this project is a collection of music videos from YouTube. I used the `yt-dlp` package to download the videos and `ffmpeg` to extract the audio. In `process_video.py`, the audio is then split into `.wav` chunks using the `librosa` library, the image frames are extracted using `opencv-python`, and the audio chunks are encoded into `.npy` files using the RAVE model.

## RAVE
RAVE or Realtime Audio Variational autoEncoder is a model for synthesizing audio by compressing it into a latent space and then expanding it back out. I'm using it as a feature extractor for the audio in the music videos. The model is trained on the High-Performance Computing Cluster at Yale University on five hours of hip hop mixes. The results are saved in `trainings/`, and the exported model is saved in `exports/`.

## Custom Decoder Model
The custom decoder model is a simple fully connected neural network in `decoder.py` that expands the latent vector to a shape that can be fed into a series of deconvolutional layers. The deconvolutional layers then generate the image. The model is trained on the latent vectors extracted from a pretrained RAVE model and the corresponding image frames, as prepared in `process_video.py`.

## Pipeline
All code must be run on the HPC Cluster. You can train the models on your local machine, but it'll take forever. The pipeline for this project is as follows:

### Part 1: Training RAVE
1. Download hip hop mixes from YouTube, e.g. `yt-dlp https://www.youtube.com/watch?v=IvXE-aDGOaU`, and put them in `datasets/hiphop/`.
2. Extract audio from videos, i.e., by running `ffmpeg -i "xxxxxxx.mp4" -vn "xxxxxxx.wav"` in the `datasets/hiphop/` directory.
3. Create a conda environment with the necessary RAVE dependencies:
```
module load miniconda
conda create -n RAVE python=3.9
conda activate RAVE
conda install ffmpeg sox -y
pip install acids-rave
```
4. Clone RAVE from GitHub with `git clone https://github.com/acids-ircam/RAVE.git`.
5. Preprocess audio using RAVE with `rave preprocess --input_path datasets/hiphop --output_path preprocessed/hiphop/`.
6. Start a slurm job to train RAVE on the preprocessed audio with `sbatch RAVE-training-batch.sh`.
7. After the job finishes, export the RAVE model with `rave export --name hiphop --run trainings/hiphop/hiphop_xxxxxxxxxx/ --streaming True --output exports/`. The exported model will be saved as a `.ts` file.

### Part 2: Training Custom Decoder Model
1. Download music videos from YouTube, e.g. `yt-dlp https://www.youtube.com/watch?v=Z-48u_uWMHY`, and put them in a folder in `datasets/`.
2. Make a new conda environment with the necessary dependencies for CUDA support (I just installed everything with pip because it's easier):
```
conda create -n RAVE2 python=3.11
conda activate RAVE2
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install librosa opencv-python pygame torchvision tqdm pillow click
```
1. Edit `decoder-training-batch.sh` to point to the correct dataset, environment, and RAVE model paths, then run `sbatch decoder-training-batch.sh` to start a slurm job. This will run `preprocess_video.py` and `decoder.py train` in sequence. The `preprocess_video.py` script will extract image frames and encode the corresponding audio chunks into latent vectors using the exported RAVE model. The `decoder.py train` script will train the custom decoder model on the latent vectors and image frames. The model will be saved as a `.pth` file.
2. After the job finishes, to generate images from audio input, run `python decoder.py demo` with the path to the `.pth` file and the path to the exported RAVE model from Part 1.

## Preliminary Results
After training RAVE on the hip hop mixes and the custom decoder model on Kendrick Lamar's "Alright" music video, I was able to generate images from audio input. The quality of the images is not great, but it's a start. The custom decoder model is not trained for very long (only 20 epochs). The images are generated in real time from audio input, so it's a proof of concept that the pipeline works.

![Live Music Video Generator](screenshots/live-audio.gif)

As a control, I had it generate images from the original audio of the "Alright" music video. The images show some smooth transitions between frames and recognizable forms.

![Control Music Video Generator](screenshots/control.gif)

I tested the decoder model separately on a test set of random 128-length latent vectors that map to images of sine waves based on the latent features. The model was able to reproduce the sine waves with some noise. On the right is one of the original sine wave images, and on the left is the corresponding image generated from the latent vector.

![Sine Wave Test](screenshots/sine-wave-test.png)

## Future Work
The next steps for this project are to:
- Train the custom decoder model longer and on more data to improve the quality of the generated images. I just trained it on my local computer on Kendrick Lamar's "Alright" music video, but I ran into memory limits on the HPC (because I designed the dataloader to load everything into memory, which won't scale to larger training data). Also, "Alright" is all black and white, so I may want to use a more colorful music video.
- Train RAVE longer on the hip hop mixes to improve the quality of the latent vectors. I trained it for several hours, but when I ran into those memory issues from the custom decoder model, the RAVE training was cut short.
- Refactor the code to be more modular and scalable. I have a lot of hard-coded paths and parameters in my Python scripts that could be abstracted into a config file or command line arguments. It's just that the pipeline makes things complicated with all the different steps and dependencies. For example, training RAVE requires a different conda environment than the custom decoder model.
- Convert models to ONNX format for deployment to a web app for real-time music video generation in JavaScript in the browser.
