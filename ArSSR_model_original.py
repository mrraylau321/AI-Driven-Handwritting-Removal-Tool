import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ArSSRDataset(Dataset):
    def __init__(self, hr_dir, scale_factor, transform=None):
        self.hr_paths = glob.glob(os.path.join(hr_dir, "*"))
        self.scale_factor = scale_factor
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # Load HR image
        hr_image = Image.open(self.hr_paths[idx]).convert("RGB")
        
        # Check if image is landscape (width > height) and rotate if needed
        width, height = hr_image.size
        if width > height:
            # Rotate landscape image by 90 degrees
            hr_image = hr_image.rotate(90, expand=True)
        
        hr_tensor = self.transform(hr_image)  # shape: (C, H, W)
        
        # Make sure dimensions are divisible by scale_factor
        C, H_hr, W_hr = hr_tensor.shape
        H_hr_new = H_hr - (H_hr % int(self.scale_factor))
        W_hr_new = W_hr - (W_hr % int(self.scale_factor))
        hr_tensor = hr_tensor[:, :H_hr_new, :W_hr_new]

        # Use bicubic interpolation to generate the LR image
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0),
            scale_factor=1.0/self.scale_factor,
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        return lr_tensor, hr_tensor


# ============================================
# Encoder: A simple CNN to extract latent features
# ============================================
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x  # shape: (B, latent_dim, H_lr, W_lr)


# ============================================
# Decoder: A simple MLP to learn the implicit function
# ============================================
class Decoder(nn.Module):
    def __init__(self, in_dim=130, hidden_dim=256, out_dim=3, num_layers=5):
        """
        Args:
            in_dim (int): Dimension of concatenated latent feature and coordinate (here latent_dim + 2).
            hidden_dim (int): Hidden layer dimension.
            out_dim (int): Output dimension (e.g., 3 channels).
            num_layers (int): Number of fully-connected layers.
        """
        super(Decoder, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================
# ArSSRNet: Combines encoder and decoder
# ============================================
class ArSSRNet(nn.Module):
    def __init__(self, scale_factor, latent_dim=128):
        super(ArSSRNet, self).__init__()
        self.scale_factor = scale_factor
        self.encoder = Encoder(in_channels=3, out_channels=latent_dim)
        # Decoder input: latent feature (latent_dim) + coordinate (2)
        self.decoder = Decoder(in_dim=latent_dim + 2, hidden_dim=256, out_dim=3, num_layers=5)

    def forward(self, lr_img, query_coords, hr_shape):
        B = lr_img.shape[0]
        # 1. Extract latent features from LR image
        feat_map = self.encoder(lr_img)  # shape: (B, latent_dim, H_lr, W_lr)
        H_hr, W_hr = hr_shape
        # The encoder's output spatial dimensions
        H_lr, W_lr = feat_map.shape[2], feat_map.shape[3]

        device = query_coords.device
        x_hr = query_coords[..., 0]  # (B, N) normalized
        y_hr = query_coords[..., 1]

        # Convert normalized HR coordinates to HR pixel coordinates
        x_hr_pix = ((x_hr + 1) / 2) * (W_hr - 1)
        y_hr_pix = ((y_hr + 1) / 2) * (H_hr - 1)

        # Map HR pixel coordinates to LR (divide by scale factor)
        x_lr_pix = x_hr_pix / self.scale_factor
        y_lr_pix = y_hr_pix / self.scale_factor

        # Normalize LR pixel coordinates to range [-1,1] for grid_sample
        x_lr_norm = 2 * (x_lr_pix / (W_lr - 1)) - 1
        y_lr_norm = 2 * (y_lr_pix / (H_lr - 1)) - 1
        lr_grid = torch.stack([x_lr_norm, y_lr_norm], dim=-1)  # (B, N, 2)
        # Reshape for grid_sample: expected shape (B, N, 1, 2)
        lr_grid = lr_grid.view(B, -1, 1, 2)

        # 3. Sample latent features from the encoder's feature map (using bilinear interpolation)
        sampled_feat = F.grid_sample(feat_map, lr_grid, mode='bilinear', align_corners=True)
        # sampled_feat shape: (B, latent_dim, N, 1) → reshape to (B, N, latent_dim)
        sampled_feat = sampled_feat.squeeze(-1).permute(0, 2, 1)

        # 4. Concatenate the latent feature with the query coordinate (still using HR normalized coord)
        decoder_input = torch.cat([sampled_feat, query_coords], dim=-1)  # (B, N, latent_dim+2)
        B, N, D = decoder_input.shape
        decoder_input_flat = decoder_input.view(B * N, D)
        pred = self.decoder(decoder_input_flat)  # (B*N, 3)
        pred = pred.view(B, N, 3)
        return pred


# ============================================
# Utility to sample random HR query coordinates
# ============================================
def sample_random_queries(hr_shape, num_samples, device):
    H_hr, W_hr = hr_shape
    xs = torch.randint(0, W_hr, (num_samples,), device=device).float()
    ys = torch.randint(0, H_hr, (num_samples,), device=device).float()
    xs_norm = 2 * (xs / (W_hr - 1)) - 1
    ys_norm = 2 * (ys / (H_hr - 1)) - 1
    queries = torch.stack([xs_norm, ys_norm], dim=-1)  # shape: (num_samples, 2)
    return queries


# ============================================
# Validation Loop
# ============================================
def validate(model, dataloader, device, num_queries=1024):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            B, C, H_hr, W_hr = hr_img.shape

            # Sample random HR query coordinates
            queries = sample_random_queries((H_hr, W_hr), num_queries, device)
            queries_batch = queries.unsqueeze(0).repeat(B, 1, 1)

            # Forward pass
            pred = model(lr_img, queries_batch, hr_shape=(H_hr, W_hr))

            # Get ground truth values
            hr_grid = queries_batch.view(B, -1, 1, 2)
            gt = F.grid_sample(hr_img, hr_grid, mode='bilinear', align_corners=True)
            gt = gt.squeeze(-1).permute(0, 2, 1)

            loss = F.l1_loss(pred, gt)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================
# Test Loop
# ============================================
def test(model, dataloader, device, num_queries=1024):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            B, C, H_hr, W_hr = hr_img.shape

            # Sample all pixels for comprehensive evaluation
            all_queries = []
            for h in range(H_hr):
                for w in range(W_hr):
                    x_norm = 2 * (w / (W_hr - 1)) - 1
                    y_norm = 2 * (h / (H_hr - 1)) - 1
                    all_queries.append([x_norm, y_norm])
            
            all_queries = torch.tensor(all_queries, device=device)
            
            # Process in chunks to avoid OOM
            chunk_size = num_queries
            num_chunks = (len(all_queries) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(all_queries))
                chunk_queries = all_queries[start_idx:end_idx]
                
                chunk_queries_batch = chunk_queries.unsqueeze(0).repeat(B, 1, 1)
                
                # Forward pass
                pred = model(lr_img, chunk_queries_batch, hr_shape=(H_hr, W_hr))
                
                # Get ground truth values
                hr_grid = chunk_queries_batch.view(B, -1, 1, 2)
                gt = F.grid_sample(hr_img, hr_grid, mode='bilinear', align_corners=True)
                gt = gt.squeeze(-1).permute(0, 2, 1)
                
                loss = F.l1_loss(pred, gt)
                test_loss += loss.item() / num_chunks
    
    return test_loss / len(dataloader)


# ============================================
# Training Loop (Updated)
# ============================================
def train(model, train_loader, val_loader, optimizer, device, 
          num_queries=1024, epochs=100, save_dir='checkpoints'):
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            B, C, H_hr, W_hr = hr_img.shape

            # Sample random HR query coordinates
            queries = sample_random_queries((H_hr, W_hr), num_queries, device)
            queries_batch = queries.unsqueeze(0).repeat(B, 1, 1)

            # Forward pass
            pred = model(lr_img, queries_batch, hr_shape=(H_hr, W_hr))

            # Get ground truth values
            hr_grid = queries_batch.view(B, -1, 1, 2)
            gt = F.grid_sample(hr_img, hr_grid, mode='bilinear', align_corners=True)
            gt = gt.squeeze(-1).permute(0, 2, 1)

            loss = F.l1_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_loss = validate(model, val_loader, device, num_queries)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save checkpoint for each epoch
        checkpoint_path = os.path.join(save_dir, f"arssr_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "arssr_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")


# ============================================
# Main: Configuration and Training (Updated)
# ============================================
if __name__ == '__main__':
    # Configuration parameters
    hr_dir = 'raw_clean'  # Directory containing high-resolution images
    scale_factor = 8      # Downsampling factor to generate LR images
    epochs = 20           # Number of training epochs
    batch_size = 64      # Batch size
    num_queries = 64      # Number of random query coordinates per image
    val_split = 0.1        # Validation split ratio
    test_split = 0.1       # Test split ratio
    checkpoint_dir = 'arssr_checkpoints'  # Directory to save model checkpoints

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use similar transforms as in SRmodel.py
    transform = transforms.Compose([
        transforms.Resize((2496, 3520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create full dataset
    full_dataset = ArSSRDataset(hr_dir=hr_dir, scale_factor=scale_factor, transform=transform)
    
    # Split dataset into train, validation, and test
    dataset_size = len(full_dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = ArSSRNet(scale_factor=scale_factor, latent_dim=128).to(device)
    
    # Fix: Load checkpoint properly
    checkpoint = torch.load("arssr_epoch_20.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Also load optimizer state if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, device, 
          num_queries=num_queries, epochs=epochs, save_dir=checkpoint_dir)

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "arssr_best_model.pth")))
    
    # Test the model
    print("Testing the best model...")
    test_loss = test(model, test_loader, device, num_queries=num_queries)
    print(f"Test Loss: {test_loss:.6f}")

    # Save the final model
    torch.save(model.state_dict(), "arssr_final_model.pth")
    print("Training complete. Final model saved to arssr_final_model.pth")