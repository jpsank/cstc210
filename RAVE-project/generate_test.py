import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
latent_dim = 64  # Dimensionality of the latent space
img_size = 256  # Image size (e.g., 256x256)
img_channels = 3  # RGB images
num_samples = 100  # Number of samples in the test dataset
output_dir = "datasets/test"  # Output directory

# Create directories
latent_dir = os.path.join(output_dir, "latents")
image_dir = os.path.join(output_dir, "images")
os.makedirs(latent_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Generate dataset where images are sine waves defined by latent vectors
x = np.linspace(0, 2 * np.pi, img_size)  # Horizontal axis for sine wave

for i in range(num_samples):
    # Create a random latent vector
    latent_vector = np.random.randn(latent_dim).astype(np.float32)
    latent_path = os.path.join(latent_dir, f"latent_{i:04d}.npy")
    np.save(latent_path, latent_vector)

    # Generate a sine wave image using latent vector for frequencies and colors
    sine_image = np.zeros((img_size, img_size, img_channels), dtype=np.uint8)
    for c in range(img_channels):
        frequency = abs(latent_vector[c % latent_dim]) + 1  # Ensure positive frequency
        sine_wave = np.sin(frequency * x)  # Generate sine wave
        sine_wave_normalized = ((sine_wave + 1) / 2 * 255).astype(np.uint8)  # Normalize to [0, 255]
        sine_image[:, :, c] = np.tile(sine_wave_normalized, (img_size, 1))  # Apply sine wave to image

    # Save the image
    image_path = os.path.join(image_dir, f"image_{i:04d}.png")
    Image.fromarray(sine_image).save(image_path)

# Show an example of the corresponding pair
example_latent = np.load(os.path.join(latent_dir, "latent_0000.npy"))
example_image = Image.open(os.path.join(image_dir, "image_0000.png"))

print("Generated latent vector shape:", example_latent.shape)
plt.imshow(example_image)
plt.title("Sine Wave Image Corresponding to Latent Vector")
plt.axis("off")
plt.show()