import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the DoubleConv module for U-Net blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Define the U-Net model with self-attention
class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck with self-attention
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.attention = nn.MultiheadAttention(embed_dim=features[-1] * 2, num_heads=8)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck with self-attention
        x = self.bottleneck(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # Reshape to (seq_len, batch, features)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Back to (batch, channels, h, w)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        return torch.sigmoid(x)  # Output in [0, 1]

# Define the custom dataset
transform_count = 0
load_count = 0
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None, max_workers=20):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.num_clean = 1005
        self.num_noisy_per_clean = 10
        self.max_workers = max_workers  # Store max_workers

    def __len__(self):
        return self.num_clean * self.num_noisy_per_clean

    def __getitem__(self, idx):
        i = idx // self.num_noisy_per_clean
        j = (idx % self.num_noisy_per_clean) + 1
        noisy_path = os.path.join(self.noisy_dir, f"raw_{i}_{j}.jpg")
        clean_path = os.path.join(self.clean_dir, f"raw_clean_{i}.jpg")
        #global load_count
        #global transform_count

        # Use ThreadPoolExecutor for concurrent image loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # Use max_workers
            noisy_img_future = executor.submit(self.load_and_transform_image, noisy_path)
            clean_img_future = executor.submit(self.load_and_transform_image, clean_path)

            noisy_img = noisy_img_future.result()
            clean_img = clean_img_future.result()

        # Use ThreadPoolExecutor for concurrent tensor transformation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # Use max_workers
            noisy_tensor_future = executor.submit(self.transform_to_tensor, noisy_img)
            clean_tensor_future = executor.submit(self.transform_to_tensor, clean_img)

            noisy_tensor = noisy_tensor_future.result()
            clean_tensor = clean_tensor_future.result()

        return noisy_tensor, clean_tensor

    def load_and_transform_image(self, path):
        img = Image.open(path).convert('RGB')

        # Rotate noisy image if landscape
        if img.width > img.height:
            img = img.rotate(90, expand=True)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img

    def transform_to_tensor(self, img):
        # Remove the second transformation
        return img  # Return the already transformed tensor

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((880, 880), interpolation=Image.BICUBIC),  # Bicubic for text clarity
    transforms.ToTensor(),  # Scales to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #Normalized: [-1, 1]
])

def visualize_results(model, dataloader, device, num_images=3):
    model.eval()  # Set the model to evaluation mode
    images, clean_images = next(iter(dataloader))  # Get a batch of images
    images, clean_images = images.to(device), clean_images.to(device)

    with torch.no_grad():
        generated_images = model(images)  # Generate clean images

    # Convert tensors to numpy arrays for visualization
    images = images.cpu().numpy()
    clean_images = clean_images.cpu().numpy()
    generated_images = generated_images.cpu().numpy()

    # Set num_images to the minimum of the requested and available images
    num_images = min(num_images, images.shape[0])

    # Set up the plot
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images > 1:
        for i in range(num_images):
            index = random.randint(0, len(images) - 1)
            # Dirty input
            axes[i, 0].imshow(np.transpose(images[index], (1, 2, 0)))
            axes[i, 0].set_title("Dirty Input")
            axes[i, 0].axis('off')

            # Real clean image
            axes[i, 1].imshow(np.transpose(clean_images[index], (1, 2, 0)))
            axes[i, 1].set_title("Real Clean Image")
            axes[i, 1].axis('off')

            # Generated clean image
            axes[i, 2].imshow(np.transpose(generated_images[index], (1, 2, 0)))
            axes[i, 2].set_title("Generated Clean Image")
            axes[i, 2].axis('off')
    else:
        index = random.randint(0, len(images) - 1)
        # For single image case, axes is 1D
        axes[0].imshow(np.transpose(images[index], (1, 2, 0)))
        axes[0].set_title("Dirty Input")
        axes[0].axis('off')

        axes[1].imshow(np.transpose(clean_images[index], (1, 2, 0)))
        axes[1].set_title("Real Clean Image")
        axes[1].axis('off')

        axes[2].imshow(np.transpose(generated_images[index], (1, 2, 0)))
        axes[2].set_title("Generated Clean Image")
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Set up dataset and dataloaders
    noisy_dir = 'raw_images_addtextonly'
    clean_dir = 'raw_clean'
    transform = transforms.Compose([
        transforms.Resize((880, 880), interpolation=Image.BICUBIC),
        transforms.ToTensor()  # Scales to [0, 1]
    ])
    
    # Create dataset
    full_dataset = DenoisingDataset(noisy_dir, clean_dir, transform=transform)
    
    # Split dataset: 70% train, 10% validation, 20% test
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNetWithAttention(in_channels=3, out_channels=3).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
        
        train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item() * noisy_imgs.size(0)
        
        val_loss = val_loss / len(val_dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save temporary model
        torch.save(model.state_dict(), "Drop_text_model_temp.pth")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "Drop_text_model_best.pth")
            print(f"New best model saved with validation loss: {val_loss:.6f}")
    
    # Test the final model
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            test_loss += loss.item() * noisy_imgs.size(0)
    
    test_loss = test_loss / len(test_dataset)
    print(f"Final test loss: {test_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), "Drop_text_model_final.pth")
    print("Training completed. Final model saved.")