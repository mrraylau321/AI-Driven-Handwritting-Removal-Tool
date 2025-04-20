import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# --- Step 1: Load CSV Data ---
# The CSV should have columns: "filename" and "angle"
csv_file = 'rotation_angles.csv'
df = pd.read_csv(csv_file)

# Folder containing the images
image_folder = "raw_noisy_full"

# --- Step 2: Split Data into Training and Validation Sets ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Step 3: Create a Custom Dataset ---
class RotationDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]['image_name']
        image_path = os.path.join(self.image_dir, filename)
        # Open the image and ensure it is in RGB mode
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert the angle to a float
        angle = float(self.df.iloc[index]['rotation_angle'])
        return image, angle

# Define image transformations (resize to 128x128 and convert to tensor with pixel values [0, 1])
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the datasets
train_dataset = RotationDataset(train_df, image_folder, transform=transform)
val_dataset = RotationDataset(val_df, image_folder, transform=transform)

# Create DataLoaders for batching and shuffling
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Step 4: Build the CNN Regression Model ---
class RotationPredictor(nn.Module):
    def __init__(self):
        super(RotationPredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                 # 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                 # 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 128 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # 128 x 16 x 16
        )
        # Calculate the flattened feature size: 128 * 16 * 16 = 32768
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # One output neuron for regression
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set device to GPU if available, otherwise CPU
    device = 'mps'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RotationPredictor().to(device)
    print(model)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.load_state_dict(torch.load('DocumentAngleRegressionModel_temp.pth'))
    print (model)

    # --- Step 5: Train the Model ---
    EPOCHS = 20
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, angles in train_loader:
            images = images.to(device)
            # Ensure angles have shape [batch_size, 1] and are of type float32
            angles = torch.tensor(angles, dtype=torch.float32).to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluate on the validation set
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                images = images.to(device)
                angles = torch.tensor(angles, dtype=torch.float32).to(device).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, angles)
                running_val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # --- Step 6: Make Predictions on a Batch from the Validation Set ---
    model.eval()

    torch.save(model.state_dict(), f'DocumentAngleRegressionModel.pth')

    # --- (Optional) Plot Training and Validation Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()