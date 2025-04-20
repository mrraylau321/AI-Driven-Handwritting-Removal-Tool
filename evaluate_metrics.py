import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the necessary functions from the original script
from visualization_combined_result_Public_pretrained import (
    RotationPredictor, UNetWithAttention, predict_and_fix_rotation,
    denoise_image, super_resolve, super_resolve_Real_ESRGAN
)

def evaluate_ssim():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define directories
    noisy_dir = 'raw_images'
    clean_dir = 'raw_clean'
    
    # Get all noisy image paths
    noisy_files = [f for f in os.listdir(noisy_dir) if f.startswith('raw_') and f.endswith('.jpg')]
    
    # Initialize results storage
    results = []
    
    # Load models
    print("Loading models...")
    rotation_model = RotationPredictor().to(device)
    rotation_model.load_state_dict(torch.load('DocumentAngleRegressionModel.pth', map_location=device))
    rotation_model.eval()
    
    denoising_model = UNetWithAttention(in_channels=3, out_channels=3).to(device)
    denoising_model.load_state_dict(torch.load('Drop_text_model_temp.pth', map_location=device))
    denoising_model.eval()
    
    # Process each image
    for noisy_file in tqdm(noisy_files, desc="Processing images"):
        try:
            # Extract the clean index
            parts = noisy_file.split('_')
            clean_idx = parts[1]
            clean_filename = f"raw_clean_{clean_idx}.jpg"
            
            noisy_path = os.path.join(noisy_dir, noisy_file)
            clean_path = os.path.join(clean_dir, clean_filename)
            
            # Load images
            noisy_img = Image.open(noisy_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
            
            # Process noisy image through pipeline
            # 1. Rotation correction
            rotated_img, angle = predict_and_fix_rotation(noisy_path, rotation_model, device)
            
            # 2. Denoising
            denoised_img = denoise_image(rotated_img, denoising_model, device)
            
            # 3. Super-resolution
            target_size = (3520, 2490)
            sr_img = super_resolve_Real_ESRGAN(denoised_img, device, target_size=target_size)
            
            # Resize clean image to match super-resolved image size
            clean_img_resized = clean_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
            
            # Convert images to numpy arrays for metric calculation
            sr_np = np.array(sr_img)
            clean_np = np.array(clean_img_resized)
            
            # Calculate SSIM
            ssim_value = ssim(clean_np, sr_np, channel_axis=2, data_range=255)
            
            # Store results
            results.append({
                'image_name': noisy_file,
                'rotation_angle': angle,
                'ssim': ssim_value
            })
            
            # Clear memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {noisy_file}: {str(e)}")
            continue
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('ssim_evaluation_results.csv', index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average SSIM: {df['ssim'].mean():.4f}")
    print(f"Median SSIM: {df['ssim'].median():.4f}")
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['ssim'], bins=20, alpha=0.7)
    plt.title('SSIM Distribution')
    plt.xlabel('SSIM Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('ssim_distribution.png')
    plt.close()

if __name__ == '__main__':
    evaluate_ssim() 