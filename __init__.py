#something in here
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cnn
from cnn import *
import datasets
from datasets import *
import layers
from layers import *
import utils
from utils import *
from torch.utils.data import DataLoader


X, y = get_cnn_dataset("coil20", 64)

transform = transforms.Compose([
    transforms.Grayscale(),          # Ensure it's in grayscale (single channel)
    transforms.Resize((128, 128)),   # Resize to 128x128 (COIL-20 images are usually 128x128)
    transforms.ToTensor()            # Convert to tensor
])

# Load the dataset (replace with your dataset path)
data_dir = r"C:\Users\dariu\Desktop\Studium\Vorlesungen und Übungsblätter\10.Semester\Bachelorthesis\dataset\coil-20-proc\coil-20-proc"  # Path to the COIL-20 dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Get the first image and its label
#image, label = dataset[200]  # Get the first image and label

# Convert the image tensor to a NumPy array (for displaying)
#img_np = image.squeeze().numpy()  # Remove the single channel dimension

# Plot the image
#plt.imshow(img_np, cmap='gray')
#plt.title(f"Label: {label}")  # Show the label of the first image
#plt.axis('off')  # Hide axes for better view
#plt.show()