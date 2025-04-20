import cv2
import numpy as np
import os
from pathlib import Path
import random

def remove_background(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Create mask
    mask = thresh > 0
    # Create white background
    result = np.zeros_like(image)
    result[mask] = [255, 255, 255]  # White color for character
    return result, mask

def load_handwritten_chars():
    char_path = Path('handwritten_char')
    chars = []
    for img_path in char_path.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(str(img_path))
            if img is not None:
                clean_char, mask = remove_background(img)
                chars.append((clean_char, mask))
    return chars

def add_random_chars(image, chars, num_chars):
    result = image.copy()
    height, width = image.shape[:2]
    
    for _ in range(num_chars):
        char_img, char_mask = random.choice(chars)
        char_h, char_w = char_img.shape[:2]
        
        # Random position
        x = random.randint(0, width - char_w)
        y = random.randint(0, height - char_h)
        
        # Region of interest
        roi = result[y:y+char_h, x:x+char_w]
        # Apply character only where mask is True
        result[y:y+char_h, x:x+char_w] = roi
        
    return result

def main():
    # Create raw_images directory
    raw_dir = Path('raw_images')
    raw_dir.mkdir(exist_ok=True)
    
    # Load handwritten characters
    chars = load_handwritten_chars()
    
    # Process each image in output_jpg
    output_path = Path('output_jpg')
    for i, img_path in enumerate(output_path.glob('*')):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Read original image
            original = cv2.imread(str(img_path))
            if original is None:
                continue
                
            # Save clean version
            clean_path = raw_dir / f'raw_{i}_clean.jpg'
            cv2.imwrite(str(clean_path), original)
            
            # Generate 10 augmented versions
            for j in range(10):
                num_chars = random.randint(50, 100)
                augmented = add_random_chars(original, chars, num_chars)
                aug_path = raw_dir / f'raw_{i}_{j+1}.jpg'
                cv2.imwrite(str(aug_path), augmented)
                
            print(f'Processed image {i+1}: {img_path.name}')

if __name__ == '__main__':
    main()