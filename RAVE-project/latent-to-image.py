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

# --- 1. Define the Model ---
class LatentToImageModel(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Fully connected layers to expand the latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * (img_size // 16) * (img_size // 16)),  # Match shape for deconv
            nn.ReLU()
        )

        # Deconvolutional layers to generate the image
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
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, latent_vector):
        x = self.fc(latent_vector)  # Expand latent vector
        x = x.view(x.size(0), 256, self.img_size // 16, self.img_size // 16)  # Reshape for deconv
        x = self.deconv(x)  # Generate image
        return x

# --- 2. Define Dataset ---
class LatentImageDataset(Dataset):
    def __init__(self, latent_dir, image_dir, img_size, transform=None):
        """
        Preload all latent vectors and corresponding images into memory.
        Args:
            latent_dir (str): Directory containing latent .npy files.
            image_dir (str): Directory containing target images.
            transform (callable, optional): Transformation applied to images.
        """
        self.latents = []
        self.images = []
        self.transform = transform

        latent_files = sorted(os.listdir(latent_dir))
        image_files = sorted(os.listdir(image_dir))
        if len(image_files) < len(latent_files):
            # Ensure that there are no extra latent vectors
            latent_files = latent_files[:len(image_files)]

        def load_data(latent_path, image_path):
            latent = np.load(latent_path)
            latent = torch.tensor(latent, dtype=torch.float32)
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.latents.append(latent)
            self.images.append(image)

        print("Pre-loading data into memory...")
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for latent_file, image_file in zip(latent_files, image_files):
                latent_path = os.path.join(latent_dir, latent_file)
                image_path = os.path.join(image_dir, image_file)
                tasks.append(executor.submit(load_data, latent_path, image_path))
            
            for _ in tqdm(as_completed(tasks), total=len(latent_files), desc="Loading data"):
                pass

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.images[idx]

# --- 3. Define Training Loop ---
def train(model: nn.Module, dataloader: DataLoader, epochs: int, device: torch.device, save_path: str):
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        # Save model checkpoint
        torch.save(model.state_dict(), save_path)

# --- 4. Define Evaluation ---
def evaluate(model, latent_vector, device):
    model.eval()
    with torch.no_grad():
        latent_vector = latent_vector.to(device)
        generated_image = model(latent_vector)
        return generated_image

# --- 5. Main Function ---
if __name__ == "__main__":
    # Directories
    latent_dir = "datasets/kdot/latents"  # Directory containing RAVE latent vectors
    image_dir = "datasets/kdot/frames"           # Directory containing corresponding images

    # Hyperparameters
    latent_dim = 64          # Dimension of RAVE latent space
    img_channels = 3          # RGB images
    img_size = 256            # Image size (e.g., 256x256)
    batch_size = 16
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "latent_to_image_model.pth"

    # Initialize model
    model = LatentToImageModel(latent_dim, img_channels, img_size).to(device)

    if False:
        # Transform for images
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Dataset and DataLoader
        dataset = LatentImageDataset(latent_dir, image_dir, img_size, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train model
        print("Training model...")
        train(model, dataloader, epochs, device, save_path)
        print("Model trained.")

    # Load model for evaluation
    model.load_state_dict(torch.load(save_path))

    # Load RAVE model
    model_path = "exports/hiphop_streaming.ts"
    rave_model = torch.jit.load(model_path).to(device)
    rave_model.eval()

    # Run interactive demo
    import sounddevice as sd
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((img_size, img_size))
    clock = pygame.time.Clock()

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
        latent = latent[None]  # Add batch dimension

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

    # Test on live audio
    done = False
    while not done:
        for event in pygame.event.get():
            # Press q to quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                done = True
        
        # Record audio from microphone
        indata = sd.rec(1839, samplerate=rave_model.sr, channels=1, dtype='float32', blocking=True)
        
        # Preprocess audio
        audio = torch.tensor(indata, dtype=torch.float32).to(device).t()
        audio = audio[None]
        print("Test audio shape:", audio.shape)
        
        # Encode audio into latents
        latent = rave_model.encode(audio)
        latent = latent.squeeze(0).flatten() # Remove batch dimension and flatten frequency bands
        latent = latent[None] # Add batch dimension
        print("Test latent shape:", latent.shape)

        # Generate image
        generated_image = evaluate(model, latent, device)
        generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_image = (generated_image + 1) / 2
        generated_image = (generated_image * 255).astype(np.uint8)

        # Display image
        pygame.display.set_caption("Live Audio")
        pygame.surfarray.blit_array(screen, generated_image)
        pygame.display.flip()
        clock.tick(30)

    
