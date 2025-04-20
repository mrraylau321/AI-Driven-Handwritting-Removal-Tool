import cv2
import numpy as np
import os
from pathlib import Path
import random
import csv
from collections import Counter

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

def random_rotate(image, angle):
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

def main():
    # Create raw_images directory
    raw_dir = Path('raw_noisy_full')
    raw_dir.mkdir(exist_ok=True)
    
    # Create CSV file to store rotation angles
    csv_path = 'rotation_angles.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'rotation_angle'])
        
        # Process each image in output_jpg
        output_path = Path('raw_images_adjusted_addtextonly')
        for i, img_path in enumerate(output_path.glob('*')):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Read original image
                original = cv2.imread(str(img_path))
                angle = random.uniform(-6, 6)
                augmented = random_rotate(original, angle)
                aug_filename = img_path.name
                aug_path = raw_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented)
                
                # Save rotation angle to CSV
                csv_writer.writerow([aug_filename, angle])
                    
                print(f'Processed image {i+1}: {img_path.name}')
    
    print(f'Rotation angles saved to {csv_path}')

if __name__ == '__main__':
    main()