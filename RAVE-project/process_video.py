import os
import torch
import torchaudio
import librosa
import numpy as np
import cv2
from tqdm import tqdm
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
video_path = "datasets/kdot/kdot.mov"
audio_path = "datasets/kdot/kdot.wav"
output_dir = "datasets/kdot"
model_path = "exports/hiphop_streaming.ts"

# Ensure output directories exist
frames_dir = os.path.join(output_dir, "frames")
audio_chunks_dir = os.path.join(output_dir, "audio_chunks")
latents_dir = os.path.join(output_dir, "latents")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(audio_chunks_dir, exist_ok=True)
os.makedirs(latents_dir, exist_ok=True)

# Split video into frames
def extract_frames(video_path, output_dir):
    """
    Extract frames from a video and save them as individual images using multithreading.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    success, frame = cap.read()

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        while success:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            tasks.append(executor.submit(cv2.imwrite, frame_path, frame))
            success, frame = cap.read()
            frame_idx += 1
        
        # Use tqdm to track progress
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Processing frames"):
            pass

    cap.release()

    return video_fps

# Split audio into chunks corresponding to frames
def split_audio(audio_path, output_dir, target_sr, fps=30):
    """
    Split audio into chunks corresponding to video frames.
    """
    audio, sr = librosa.load(audio_path, sr=None)  # Load audio with original sampling rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    chunk_duration = int(target_sr / fps)  # Samples per frame
    num_chunks = len(audio) // chunk_duration
    
    for i in tqdm(range(num_chunks), desc="Splitting audio into chunks"):
        chunk = audio[i * chunk_duration:(i + 1) * chunk_duration]
        chunk_path = os.path.join(output_dir, f"audio_{i:04d}.wav")
        sf.write(chunk_path, chunk, target_sr)

# Generate latents using RAVE model
def generate_latents(rave_model, audio_dir, output_dir, device):
    """
    Generate latents for each audio chunk using the RAVE model.
    """
    rave_model.eval()
    for audio_file in tqdm(sorted(os.listdir(audio_dir)), desc="Generating latents"):
        audio_path = os.path.join(audio_dir, audio_file)
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            audio = audio.to(device)
            if sr != rave_model.sr:
                audio = torchaudio.functional.resample(audio, sr, rave_model.sr)
            
            # Encode audio into latents
            latent = rave_model.encode(audio[None])  # Add batch dimension
            latent = latent.squeeze(0).flatten() # Remove batch dimension and flatten frequency bands
            latent_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.npy")
            np.save(latent_path, latent.cpu().detach().numpy())
        except Exception as e:
            print(f"Failed to process {audio_file}: {e}")

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load RAVE model
    rave_model = torch.jit.load(model_path).to(device)

    if False:
        # Step 1: Extract video frames
        print("Extracting frames from video...")
        fps = extract_frames(video_path, frames_dir)
        print("Video FPS:", fps)

        # Step 2: Split audio into chunks
        print("Splitting audio into chunks...")
        target_sr = rave_model.sr  # Expected sampling rate for RAVE
        split_audio(audio_path, audio_chunks_dir, target_sr, fps=fps)

    # Step 3: Generate latents
    print("Generating latents with RAVE...")
    generate_latents(rave_model, audio_chunks_dir, latents_dir, device)

    print("Processing complete. Data saved in:", output_dir)