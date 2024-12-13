import os
import torch
import torchaudio
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import click

# Click
@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("audio_path", type=click.Path(exists=True))
@click.argument("rave_model_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--only-latents", is_flag=True, help="Skip frame extraction and audio splitting", default=False)
def main(video_path, audio_path, rave_model_path, output_dir, only_latents=False):
    """
    Preprocess video and audio data for decoder model training.
    """
    # Ensure output directories exist
    frames_dir = os.path.join(output_dir, "frames")
    latents_dir = os.path.join(output_dir, "latents")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    # Load RAVE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rave_model = torch.jit.load(rave_model_path).to(device)
    target_sr = rave_model.sr

    # Open video and get metadata
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not only_latents:
        # Multithreaded frame extraction
        print("Extracting video frames...")
        frame_idx = 0
        success, frame = cap.read()
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            while success:
                # Save the current frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
                tasks.append(executor.submit(cv2.imwrite, frame_path, frame))
                frame_idx += 1

                # Read the next frame
                success, frame = cap.read()
                print(f"Extracted {frame_idx}/{total_frames} frames.", end="\r")
            
            print(f"Extracted {frame_idx}/{total_frames} frames.")

    # Release video capture  
    cap.release()

    # Extract audio using torchaudio
    print("Extracting audio...")
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if audio has multiple channels
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)  # Average the channels
    
    # Resample audio if necessary
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    
    # Calculate samples per frame
    chunk_size = int(target_sr / fps)  # Number of audio samples per frame
    num_chunks = audio.size(1) // chunk_size  # Number of audio chunks
    if num_chunks > total_frames:
        print("Warning: More audio chunks than video frames. Truncating audio chunks.")
        num_chunks = total_frames

    # Generate latents for audio chunks
    print("Generating latents...")
    for frame_idx in tqdm(range(num_chunks), desc="Generating latents"):
        # Extract corresponding audio chunk
        chunk = audio[:, frame_idx * chunk_size:(frame_idx + 1) * chunk_size]
        chunk = chunk.unsqueeze(0)  # Add batch dimension
        chunk = chunk.to(device)

        # Encode audio into latents
        latent = rave_model.encode(chunk)
        latent = latent.squeeze(0)  # Remove batch dimension
        latent_path = os.path.join(latents_dir, f"latent_{frame_idx:04d}.npy")
        np.save(latent_path, latent.cpu().detach().numpy())

    print("Done!")

if __name__ == "__main__":
    main()

"""
Usage:
python preprocess_video.py datasets/kdot/kdot.mov datasets/kdot/kdot.wav exports/hiphop_streaming.ts datasets/kdot
python preprocess_video.py datasets/allmylife/allmylife.mp4 datasets/allmylife/allmylife.wav exports/hiphop_streaming.ts datasets/allmylife
"""