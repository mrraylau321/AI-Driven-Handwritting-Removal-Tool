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
import cv2
from collections import Counter
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

# Define the angle regression model
class RotationPredictor(nn.Module):
    def __init__(self):
        super(RotationPredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def sample_all_pixels(hr_shape, device):
    """Generate query points for all pixels in the high-resolution image."""
    H_hr, W_hr = hr_shape
    # Create grid of coordinates
    y_coords = torch.linspace(-1, 1, H_hr, device=device)
    x_coords = torch.linspace(-1, 1, W_hr, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Reshape to [H*W, 2] where each row is [x, y]
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return query_points

def detect_border_color(image):
    """Detect the most likely border color by sampling the edges of the image"""
    # Get dimensions
    h, w = image.shape[:2]
    
    # Collect border pixels (from top, bottom, left and right edges)
    border_pixels = []
    border_pixels.extend([tuple(map(int, pixel)) for pixel in image[0, :]])  # top edge
    border_pixels.extend([tuple(map(int, pixel)) for pixel in image[h-1, :]]) # bottom edge
    border_pixels.extend([tuple(map(int, pixel)) for pixel in image[:, 0]])  # left edge 
    border_pixels.extend([tuple(map(int, pixel)) for pixel in image[:, w-1]]) # right edge
    
    # Find the most common color
    color_counter = Counter(border_pixels)
    most_common_color = color_counter.most_common(1)[0][0]
    
    # Make sure we return integers for OpenCV
    return [int(c) for c in most_common_color]

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Automatically detect border color
    border_value = detect_border_color(image)
    
    # Make sure the border value is properly formatted
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=border_value)
    return rotated

def predict_and_fix_rotation(image_path, model, device):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load the original image for rotation
    original_img = Image.open(image_path).convert('RGB')
    
    # Transform for prediction
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Predict rotation angle
    with torch.no_grad():
        pred_angle = model(img_tensor).item()
    
    # Convert PIL image to numpy for OpenCV functions
    img_np = np.array(original_img)
    
    # Rotate the image using smart border detection
    rotated_np = rotate_image(img_np, -pred_angle)
    
    # Convert back to PIL image
    rotated_img = Image.fromarray(rotated_np)
    
    return rotated_img, pred_angle

def denoise_image(image, model, device):
    # Transform image for the U-Net model
    transform = transforms.Compose([
        transforms.Resize((880, 880), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)
    original_np = np.array(image.resize((880, 880), Image.BICUBIC))
    
    # Generate clean image
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    
    # Convert tensor back to PIL image
    denoised_tensor = denoised_tensor.cpu().squeeze(0)
    denoised_img = transforms.ToPILImage()(denoised_tensor)
    denoised_np = np.array(denoised_img)
    
    # Detect background color from original image
    bg_color = detect_document_background(original_np)
    print(f"Detected background color: {bg_color}")
    
    # Create background mask (where pixels are very close to white in the denoised image)
    # Adjust threshold as needed (higher = less preservation, lower = more preservation)
    threshold = 0.95
    white_mask = np.all(denoised_np > 255 * threshold, axis=2)
    
    # Blend the original background color into white areas
    for c in range(3):  # RGB channels
        denoised_np[:,:,c][white_mask] = bg_color[c]
    
    # Convert back to PIL
    result_img = Image.fromarray(denoised_np)
    return result_img

def detect_document_background(image):
    """Detect the background color of a document using histogram analysis"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create a mask of likely background pixels (high value/brightness)
    # This assumes background is lighter than text
    bg_mask = (hsv[:,:,2] > 200)  # V channel > 200 (bright areas)
    
    # If there aren't enough background pixels, relax the threshold
    if np.sum(bg_mask) < (image.shape[0] * image.shape[1] * 0.2):
        bg_mask = (hsv[:,:,2] > 180)
    
    # Get the average color of the background pixels
    bg_color = [
        int(np.mean(image[:,:,0][bg_mask])),
        int(np.mean(image[:,:,1][bg_mask])),
        int(np.mean(image[:,:,2][bg_mask]))
    ]
    
    return bg_color

def super_resolve_Real_ESRGAN(image, device, target_size=(2490, 3520), scale_factor=4):
    import torch
    import torchvision.transforms.functional as F
    
    try:
        # Try to use pretrained Real-ESRGAN if available (need to pip install basicsr)
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        from PIL import Image
        
        print("Using Real-ESRGAN for super-resolution")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        model_path = load_file_from_url('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth')
        upsampler = RealESRGANer(scale=scale_factor, model_path=model_path, model=model, device=device)
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Apply super-resolution
        output, _ = upsampler.enhance(img_array)
        
        # Convert back to PIL image
        sr_img = Image.fromarray(output)
    
    except (ImportError, ModuleNotFoundError):
        # Fallback to using pretrained Super Resolution model from torch hub
        try:
            print("Falling back to SRResNet model from torch hub")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'srresnet', pretrained=True)
            model = model.to(device).eval()
            
            # Convert PIL image to tensor
            input_tensor = F.to_tensor(image).unsqueeze(0).to(device)
            
            # Apply super-resolution
            with torch.no_grad():
                output = model(input_tensor)
            
            # Convert back to PIL image
            output_tensor = output.squeeze(0).cpu().clamp(0, 1)
            sr_img = F.to_pil_image(output_tensor)
            
        except (ImportError, RuntimeError):
            # Final fallback to traditional upsampling
            print("Falling back to traditional upsampling methods")
            from PIL import Image
            import torch
            import torchvision.transforms as transforms
            
            # First use bicubic interpolation
            sr_img = image.resize((image.width * scale_factor, image.height * scale_factor), 
                                  Image.BICUBIC)
    
    # Step 2: Rescale to target dimensions
    final_img = sr_img.resize((target_size[1], target_size[0]), Image.LANCZOS)  # PIL uses (width, height)
    
    return final_img

import cv2
from PIL import Image

def super_resolve(image,device, target_size=(2490, 3520), scale_factor=4):
    # Load OpenCV's ESPCN model (x4)
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel('ESPCN_x4.pb')  # Download from OpenCV's repo
    model.setModel('espcn', scale_factor)

    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Super-resolve
    sr_img = model.upsample(img_np)
    
    # Resize to target dimensions
    final_img = Image.fromarray(sr_img).resize(
        (target_size[1], target_size[0]),  # (width, height)
        Image.LANCZOS
    )
    return final_img

def process_and_visualize():
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Pick a random noisy image
    noisy_dir = 'raw_noisy_full'
    clean_dir = 'raw_clean'
    
    # Get all noisy image paths
    noisy_files = [f for f in os.listdir(noisy_dir) if f.startswith('raw_') and f.endswith('.jpg')]
    
    # Randomly select a noisy image
    random_noisy = random.choice(noisy_files)
    print(f"Selected noisy image: {random_noisy}")
    
    # Extract the clean index
    parts = random_noisy.split('_')
    clean_idx = parts[1]
    clean_filename = f"raw_clean_{clean_idx}.jpg"
    
    noisy_path = os.path.join(noisy_dir, random_noisy)
    clean_path = os.path.join(clean_dir, clean_filename)
    
    # Create list to store intermediate results
    results = []
    
    original_img = Image.open(noisy_path).convert('RGB')
    results.append(("Original Noisy", original_img))
    
    # 2. Process with rotation correction model
    print("Loading rotation correction model...")
    rotation_model = RotationPredictor().to(device)
    rotation_model.load_state_dict(torch.load('DocumentAngleRegressionModel.pth', map_location=device))
    rotation_model.eval()
    
    rotated_img, angle = predict_and_fix_rotation(noisy_path, rotation_model, device)
    results.append((f"Rotation Corrected ({angle:.2f}°)", rotated_img))
    
    # Clear the model from memory
    del rotation_model
    try:torch.cuda.empty_cache()
    except:pass
    print("Rotation correction completed.")
    
    # 3. Process with denoising model
    print("Loading denoising model...")
    denoising_model = UNetWithAttention(in_channels=3, out_channels=3).to(device)
    denoising_model.load_state_dict(torch.load('Drop_text_model_temp.pth', map_location=device))
    denoising_model.eval()
    
    denoised_img = denoise_image(rotated_img, denoising_model, device)
    #for ct in range(50):denoised_img = denoise_image(denoised_img, denoising_model, device)
    results.append(("Denoised", denoised_img))
    
    # Clear the model from memory
    del denoising_model
    try:torch.cuda.empty_cache()
    except:pass
    print("Denoising completed.")
    
    # 4. Process with super-resolution model
    print("Loading super-resolution model...")
    
    # Super-resolve to full target size (2490x3520)
    target_size = (3520, 2490)
    
    # Skip loading the ArSSR model entirely
    # sr_model = ArSSRNet(scale_factor=4, latent_dim=128).to(device)
    # checkpoint = torch.load('Arssr_epoch_20.pth', map_location=device)
    # sr_model.load_state_dict(checkpoint['model_state_dict'])
    # sr_model.eval()
    
    sr_img = super_resolve(denoised_img, device, target_size=target_size)
    
    # For display purposes, resize to a reasonable size while preserving aspect ratio
    display_sr_img = sr_img.resize((623, 880), Image.LANCZOS)
    results.append(("Super-Resolved", display_sr_img))
    
    try:
        torch.cuda.empty_cache()
    except:
        pass
    print("Super-resolution completed.")
    
    # 5. Original clean image for comparison
    clean_img = Image.open(clean_path).convert('RGB')
    # Resize clean image for display while preserving aspect ratio
    display_clean_img = clean_img.resize((623, 880), Image.BICUBIC)  # Changed dimensions to maintain portrait orientation
    #results.append(("Original Clean", display_clean_img))

    display_sr_img.save('generated_image_pretrainedSR.jpg')
    
    # Visualize all results
    fig, axes = plt.subplots(1, len(results), figsize=(20, 6))  # Made figure taller to accommodate portrait images
    
    for i, (title, img) in enumerate(results):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    process_and_visualize()