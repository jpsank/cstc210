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


# Define the RNN-based encoder
class StatefulRNNDecoder(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_dim = latent_dim * 2 # Hidden state size

        self.rnn = nn.GRU(latent_dim, self.hidden_dim, batch_first=True)
        self.hidden_state = None

        # Fully connected layers and deconv (as before)
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
            # Center crop to minimum dimension
            min_dim = min(image.size)
            image = transforms.CenterCrop(min_dim)(image)
            # Resize to 256x256 and make tensor
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
@click.option("--save_path", type=click.Path(), default="latent_to_image_model.pth", help="Path to save trained model")
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
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Dataset and DataLoader
    dataset = LatentImageDataset(latent_dir, image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Display some sample images
    sample_latent, sample_image = dataset[300]
    sample_image = sample_image.permute(1, 2, 0).numpy()
    sample_image = (sample_image + 1) / 2
    plt.imshow(sample_image)
    plt.axis("off")
    plt.show()

    # Train model
    print("Training model...")
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()  # Reconstruction loss
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
@click.option("--img_size", type=int, default=256, help="Image size (e.g., 256x256)")
@click.option("--latent_dim", type=int, default=16, help="Dimension of RAVE latent space")
@click.option("--img_channels", type=int, default=3, help="Number of image channels (e.g., 3 for RGB)")
@click.option("--latent_dir", type=click.Path(exists=True), help="Directory containing latent vectors", required=False)
@click.option("--audio_file", type=click.Path(exists=True), help="Audio file for live demo", required=False)
def demo(model_path, img_size, latent_dim, img_channels, latent_dir, audio_file):
    import sounddevice as sd
    import pygame

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model for evaluation
    model = StatefulRNNDecoder(latent_dim, img_size=img_size, img_channels=img_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load RAVE model
    model_path = "exports/hiphop_streaming.ts"
    rave_model = torch.jit.load(model_path).to(device)
    rave_model.eval()

    # Run interactive demo
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
            generated_image = (generated_image + 1) / 2
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
        fps = 60
        chunk_size = rave_model.sr // fps

        # Play audio and process simultaneously
        print("Playing audio and generating images...")
        done = False
        start_idx = 0
        sd.play(audio.squeeze(0).cpu().numpy(), samplerate=rave_model.sr, blocking=False)
        while start_idx < audio.shape[1] and not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True

            # Extract audio chunk
            end_idx = min(start_idx + chunk_size, audio.shape[1])
            audio_chunk = audio[:, start_idx:end_idx]
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

            # Update indices and maintain sync
            start_idx += chunk_size
            clock.tick(fps)

    # Test on live audio
    fps = 60
    done = False
    while not done:
        for event in pygame.event.get():
            # Press q to quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                done = True
        
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
        generated_image = (generated_image + 1) / 2
        generated_image = (generated_image * 255).astype(np.uint8)

        # Display image
        pygame.display.set_caption("Live Audio")
        pygame.surfarray.blit_array(screen, generated_image)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    cli()

"""
Usage:
python latent-to-image.py train datasets/kdot/latents datasets/kdot/frames --latent_dim 16 --img_channels 3 --img_size 256 --batch_size 16 --epochs 20 --save_path latent_to_image_model.pth
python latent-to-image.py demo latent_to_image_model.pth --img_size 256 --latent_dir datasets/kdot/latents
python latent-to-image.py demo latent_to_image_model.pth --audio_file datasets/allmylife/allmylife.wav
"""