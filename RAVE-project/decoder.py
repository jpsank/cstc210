import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from torchvision.models import vgg16, VGG16_Weights
import subprocess


# Define the RNN-based encoder
class StatefulRNNDecoder(nn.Module):
    def __init__(self, latent_dim=16, img_size=256, img_channels=3, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(latent_dim, self.hidden_dim, batch_first=True)
        self.hidden_state = None

        # Fully connected layers and deconv
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * (img_size // 16) * (img_size // 16)),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Transpose to [batch_size, time_steps, features] for RNN
        x = x.transpose(1, 2)

        # Adjust hidden state dynamically for batch size
        if self.hidden_state is None or self.hidden_state.size(1) != x.size(0):
            self.hidden_state = x.new_zeros(1, x.size(0), self.hidden_dim)
        else:
            self.hidden_state = self.hidden_state[:, :x.size(0), :]
        
        # Pass the hidden state for stateful behavior
        output, self.hidden_state = self.rnn(x, self.hidden_state)
        # Detach the hidden state to prevent backpropagation through time
        self.hidden_state = self.hidden_state.detach()
        # Use the last output as the latent representation
        output = output[:, -1, :]

        # Fully connected layers and deconv
        x = self.fc(output)
        x = x.view(x.size(0), 256, self.img_size // 16, self.img_size // 16) # Reshape for deconv
        x = self.deconv(x)
        return x

    def reset_hidden_state(self):
        self.hidden_state = None


# Define the dataset for latent vectors and images
class LatentImageDataset(Dataset):
    def __init__(self, latent_dir, image_dir, transform=None):
        """
        Preload all latent vectors and corresponding images into memory.
        Args:
            latent_dir (str): Directory containing latent .npy files.
            image_dir (str): Directory containing target images.
            transform (callable, optional): Transformation applied to images.
        """
        self.latents = {}
        self.images = {}
        self.transform = transform

        latent_files = sorted(os.listdir(latent_dir))
        image_files = sorted(os.listdir(image_dir))
        if len(image_files) < len(latent_files):
            # Ensure that there are no extra latent vectors
            print("Warning: More latent vectors than images. Truncating latent vectors.")
            latent_files = latent_files[:len(image_files)]

        def load_data(idx, latent_path, image_path):
            # Load latent vector
            latent = np.load(latent_path)
            latent = torch.tensor(latent, dtype=torch.float32)
            self.latents[idx] = latent

            # Load image
            image = Image.open(image_path).convert("RGB")
            # # Center crop to minimum dimension
            # min_dim = min(image.size)
            # image = transforms.CenterCrop(min_dim)(image)
            if self.transform:
                image = self.transform(image)
            self.images[idx] = image

        print("Pre-loading data into memory...")
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for idx, (latent_file, image_file) in enumerate(zip(latent_files, image_files)):
                latent_path = os.path.join(latent_dir, latent_file)
                image_path = os.path.join(image_dir, image_file)
                tasks.append(executor.submit(load_data, idx, latent_path, image_path))
            
            for _ in tqdm(as_completed(tasks), total=len(latent_files), desc="Loading data"):
                pass

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.images[idx]


# Define the evaluation function
def evaluate(model, latent_vector, device):
    model.eval()
    with torch.no_grad():
        latent_vector = latent_vector.to(device)
        generated_image = model(latent_vector)
        return generated_image


# Define the command-line interface
@click.group()
def cli():
    pass

@cli.command("train")
@click.argument("latent_dir", type=click.Path(exists=True))
@click.argument("image_dir", type=click.Path(exists=True))
@click.option("--latent_dim", type=int, default=16, help="Dimension of RAVE latent space")
@click.option("--img_channels", type=int, default=3, help="Number of image channels (e.g., 3 for RGB)")
@click.option("--img_size", type=int, default=256, help="Image size (e.g., 256x256)")
@click.option("--batch_size", type=int, default=16, help="Batch size for training")
@click.option("--epochs", type=int, default=20, help="Number of training epochs")
@click.option("--save_path", type=click.Path(), default="decoder_model.pth", help="Path to save trained model")
@click.option("--resume", is_flag=True, help="Resume training from existing model", default=False)
def train(latent_dir, image_dir, latent_dim, img_channels, img_size, batch_size, epochs, save_path, resume):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = StatefulRNNDecoder(latent_dim, img_size=img_size, img_channels=img_channels).to(device)
    if resume and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # Transform for images
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    # Dataset and DataLoader
    dataset = LatentImageDataset(latent_dir, image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Display some sample images
    # sample_latent, sample_image = dataset[300]
    # sample_image = sample_image.permute(1, 2, 0).numpy()
    # sample_image = (sample_image + 1) / 2
    # plt.imshow(sample_image)
    # plt.axis("off")
    # plt.show()

    # Train model
    print("Training model...")
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # # Normalization for VGG16
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # # Perceptual loss using VGG16 features
    # weights = VGG16_Weights.DEFAULT
    # vgg = vgg16(weights=weights).features[:23].to(device).eval()
    # for param in vgg.parameters():
    #     param.requires_grad = False

    # def perceptual_loss(pred, target):
    #     pred = normalize(pred)
    #     target = normalize(target)
    #     pred_features = vgg(pred)
    #     target_features = vgg(target)
    #     return nn.MSELoss()(pred_features, target_features)

    # # Combined loss function
    # mse_loss = nn.MSELoss()
    # def loss_fn(pred, target):
    #     return 0.7 * mse_loss(pred, target) + 0.5 * perceptual_loss(pred, target)
    loss_fn = nn.MSELoss()
    
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for latent_vectors, target_images in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            latent_vectors, target_images = latent_vectors.to(device), target_images.to(device)

            # Forward pass
            generated_images = model(latent_vectors)
            loss = loss_fn(generated_images, target_images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Reset hidden state after each epoch
        model.reset_hidden_state()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        # Save model checkpoint
        torch.save(model.state_dict(), save_path)
    
    print("Model trained.")


@cli.command("demo")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("rave_model_path", type=click.Path(exists=True))
@click.option("--img_size", type=int, default=256, help="Image size (e.g., 256x256)")
@click.option("--latent_dim", type=int, default=16, help="Dimension of RAVE latent space")
@click.option("--img_channels", type=int, default=3, help="Number of image channels (e.g., 3 for RGB)")
@click.option("--latent_dir", type=click.Path(exists=True), help="Directory containing latent vectors", required=False)
@click.option("--audio_file", type=click.Path(exists=True), help="Audio file for live demo", required=False)
@click.option("--reproduce", type=click.Path(exists=True), help="Reproduce video from a specific audio file", required=False)
def demo(model_path, rave_model_path, img_size, latent_dim, img_channels, latent_dir, audio_file, reproduce):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model for evaluation
    model = StatefulRNNDecoder(latent_dim, img_size=img_size, img_channels=img_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load RAVE model
    rave_model = torch.jit.load(rave_model_path).to(device)
    rave_model.eval()

    # Sample one latent vector, generate image, and save to file
    latent = torch.randn(1, latent_dim, 1, device=device)
    generated_image = evaluate(model, latent, device)
    generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated_image = (1 + generated_image) / 2 # Denormalize
    generated_image = (generated_image * 255).astype(np.uint8)
    with open("demo_image.jpg", "wb") as f:
        Image.fromarray(generated_image).save(f, format="JPEG")
    
    if reproduce:
        import torchaudio
        import cv2

        # Load audio file
        audio, sr = torchaudio.load(reproduce)
        audio = audio.to(device)
        if sr != rave_model.sr:
            audio = torchaudio.functional.resample(audio, sr, rave_model.sr)

        # Flatten audio channels
        audio = audio.mean(dim=0, keepdim=True)
        # audio = audio[:, :rave_model.sr * 30]

        # Chunk size for audio processing (1 second per chunk)
        fps = 24

        # Generate images from audio
        print("Generating images...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
        video_writer = cv2.VideoWriter("tmp.avi", fourcc, fps, (img_size, img_size))
        for frame_idx in tqdm(range(audio.size(1) // (rave_model.sr // fps)), desc="Generating images"):
            # Extract corresponding audio chunk
            chunk = audio[:, frame_idx * (rave_model.sr // fps):(frame_idx + 1) * (rave_model.sr // fps)]
            chunk = chunk.unsqueeze(0)
        
            # Encode audio into latents
            latent = rave_model.encode(chunk)
            latent = latent.to(device)
            with torch.no_grad():
                img = evaluate(model, latent, device)
                img = img.squeeze(0)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
            img = (1 + img) / 2
            img = (255 * img).astype(np.uint8)
            video_writer.write(img)

        video_writer.release()

        # Combine video and audio
        print("Combining video and audio...")
        subprocess.run(["ffmpeg", "-y", "-i", "tmp.avi", "-i", reproduce, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "reproduced_video.mp4"])

        # Clean up
        os.remove("tmp.avi")
        print("Reproduced video saved as 'reproduced_video.mp4'.")

    # Run interactive demo
    import sounddevice as sd
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((img_size, img_size))
    clock = pygame.time.Clock()

    if latent_dir:
        # Test on a range of latent vectors
        latent_files = sorted(os.listdir(latent_dir))
        done = False
        while latent_files and not done:
            for event in pygame.event.get():
                # Press q to quit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True
            
            # Load latent vector
            latent_file = latent_files.pop(0)
            latent_path = os.path.join(latent_dir, latent_file)
            latent = np.load(latent_path)
            latent = torch.tensor(latent, dtype=torch.float32)
            latent = latent.unsqueeze(0)  # Add batch dimension

            # Generate image using latent decoder
            generated_image = evaluate(model, latent, device)
            generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_image = (1 + generated_image) / 2
            generated_image = (generated_image * 255).astype(np.uint8)

            # # Generate audio using RAVE model
            # audio = rave_model.decode(latent)
            # audio = audio.squeeze(0).cpu().numpy()
            # audio = np.clip(audio, -1, 1)
            
            # # Play audio
            # sd.play(audio, samplerate=rave_model.sr)

            # Display image
            pygame.display.set_caption(f"Latent: {latent_file}")
            pygame.surfarray.blit_array(screen, generated_image)
            pygame.display.flip()
            clock.tick(100)

    if audio_file:
        import torchaudio
        # Load audio file
        audio, sr = torchaudio.load(audio_file)
        audio = audio.to(device)
        if sr != rave_model.sr:
            audio = torchaudio.functional.resample(audio, sr, rave_model.sr)

        # Flatten audio channels
        audio = audio.mean(dim=0, keepdim=True) # Average the channels

        # Chunk size for audio processing (1 second per chunk)
        fps = 29

        # Play audio and process simultaneously
        print("Playing audio and generating images...")
        done = False
        start_time = time.time()
        last_time = start_time
        sd.play(audio.squeeze(0).cpu().numpy(), samplerate=rave_model.sr, blocking=False)
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            last_elapsed_time = last_time - start_time

            # Extract audio chunk
            audio_chunk = audio[:, int(last_elapsed_time * rave_model.sr):int(elapsed_time * rave_model.sr)]
            audio_chunk = audio_chunk.to(device).unsqueeze(0) # Add batch dimension
            # print("Audio chunk shape:", audio_chunk.shape)

            # Encode audio chunk to latent
            latent = rave_model.encode(audio_chunk)
            # print("Latent shape:", latent.shape)

            # Generate image from latent
            generated_image = model(latent).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            generated_image = (generated_image + 1) / 2
            generated_image = (generated_image * 255).astype(np.uint8)

            # Display image
            pygame.surfarray.blit_array(screen, generated_image)
            pygame.display.flip()

            # Update last time
            last_time = time.time()

            # Update indices and maintain sync
            clock.tick(fps)

    # Test on live audio
    fps = 29
    done = False
    while True:
        for event in pygame.event.get():
            # Press q to quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True
        
        if done:
            break
        
        # Record audio from microphone
        indata = sd.rec(rave_model.sr // fps, samplerate=rave_model.sr, channels=1, dtype='float32', blocking=True)
        
        # Preprocess audio
        audio = torch.tensor(indata, dtype=torch.float32).to(device).t()
        audio = audio.unsqueeze(0)  # Add batch dimension
        # print("Test audio shape:", audio.shape)
        
        # Encode audio into latents
        latent = rave_model.encode(audio)
        # print("Test latent shape:", latent.shape)

        # Generate image
        generated_image = evaluate(model, latent, device)
        generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_image = (1 + generated_image) / 2
        generated_image = (generated_image * 255).astype(np.uint8)

        # Display image
        pygame.display.set_caption("Live Audio")
        pygame.surfarray.blit_array(screen, generated_image)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


@cli.command("export")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("rave_model_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--img_size", type=int, default=256, help="Image size (e.g., 256x256)")
@click.option("--latent_dim", type=int, default=16, help="Dimension of RAVE latent space")
@click.option("--img_channels", type=int, default=3, help="Number of image channels (e.g., 3 for RGB)")
def export(model_path, rave_model_path, output_path, img_size, latent_dim, img_channels):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model for evaluation
    model = StatefulRNNDecoder(latent_dim, img_size=img_size, img_channels=img_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load RAVE model
    rave_model = torch.jit.load(rave_model_path).to(device)
    rave_model.eval()

    # Export model
    example_latent = torch.randn(1, latent_dim, 1, device=device)
    traced_model = torch.jit.trace(model, example_latent)
    traced_model.save(output_path)

    print(f"Model exported to {output_path}.")

if __name__ == "__main__":
    cli()

"""
Usage:
python decoder.py train datasets/kdot/latents datasets/kdot/frames --latent_dim 16 --img_channels 3 --img_size 256 --batch_size 16 --epochs 20 --save_path decoder_model.pth
python decoder.py demo decoder_model.pth exports/hiphop_streaming.ts --latent_dir datasets/kdot/latents
python decoder.py demo decoder_model.pth exports/hiphop_streaming.ts --audio_file datasets/allmylife/allmylife.wav
"""