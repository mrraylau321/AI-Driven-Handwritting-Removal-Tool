import cv2
import numpy as np
import os
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed #Threading is applied to the image processing
from tqdm import tqdm  # for progress bar, easily to see the progress
import pickle  # Add this import at the top of the file

def append_images(img1, img2, is_horizontal=True):
    # Get the heights of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if is_horizontal:
        # Create black canvas with height of the larger image
        max_height = max(h1, h2)
        
        # Resize images to match the max height while maintaining aspect ratio
        if h1 != max_height:
            scale = max_height / h1
            new_width = int(w1 * scale)
            img1 = cv2.resize(img1, (new_width, max_height))
        
        if h2 != max_height:
            scale = max_height / h2
            new_width = int(w2 * scale)
            img2 = cv2.resize(img2, (new_width, max_height))
        
        # Horizontal concatenation
        combined_img = np.hstack((img1, img2))
    else:
        # Vertical concatenation
        max_width = max(w1, w2)
        
        # Resize images to match the max width while maintaining aspect ratio
        if w1 != max_width:
            scale = max_width / w1
            new_height = int(h1 * scale)
            img1 = cv2.resize(img1, (max_width, new_height))
        
        if w2 != max_width:
            scale = max_width / w2
            new_height = int(h2 * scale)    
        combined_img = np.vstack((img1, img2))  
    return combined_img

def remove_background(image):
    # Invert the image first (255 - pixel_value) for black background, white character
    inverted = 255 - image
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    mask = thresh > 0
    result = inverted.copy()
    bg_mask = ~mask
    result[bg_mask] = [255, 255, 255]
    return result, mask

def add_subtle_rgb_variation(color, variation_range=3):
    return [
        max(0, min(255, c + random.randint(-variation_range, variation_range)))
        for c in color
    ]

def random_color_variation(image, mask):
    result = image.copy()
    char_pixels = mask
    
    if char_pixels.any():
        # More natural ink-like colors (darker shades)
        if random.random() < 0.4:  # 40% chance of very dark color
            base_color = [
                random.randint(0, 60),
                random.randint(0, 60),
                random.randint(0, 60)
            ]
        else:  # 60% chance of having one dominant dark channel
            primary_channel = random.randint(0, 2)
            base_color = [
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            ]
            base_color[primary_channel] = random.randint(100, 255)
        
        #print(f"Base color: {base_color}")  # Debugging output
        
        # Apply slightly different colors to each pixel
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if char_pixels[y, x]:  # Ensure we are within bounds
                    result[y, x] = add_subtle_rgb_variation(base_color)
        
        # Debugging output to check the result after color variation
        #print(f"Result after color variation (sample): {result[0, 0]}")  # Check a sample pixel
        
        # Add random noise to some pixels
        noise_mask = char_pixels & (np.random.rand(*mask.shape) < 0.3)
        noise = np.random.randint(-15, 15, size=3)
        
        for i in range(3):
            channel = result[:,:,i]
            channel[noise_mask] = np.clip(channel[noise_mask] + noise[i], 0, 255)
            result[:,:,i] = channel
    
    return result

def is_space_occupied(result, x, y, w, h, color_threshold=15, occupation_threshold=0.1):
    roi = result[y:y+h, x:x+w]
    
    # Calculate standard deviation for each channel
    std_per_channel = np.std(roi, axis=(0,1))
    # Calculate mean color of the region
    mean_color = np.mean(roi, axis=(0,1))
    
    # Check if the region has similar colors (small standard deviation)
    is_uniform_color = np.all(std_per_channel < color_threshold)
    
    if not is_uniform_color:
        # If colors vary significantly, consider it occupied
        return True
    
    # For uniform colored regions, check if it's different from surrounding pixels
    # Get a slightly larger ROI to compare edges
    pad = 2
    y1, y2 = max(0, y-pad), min(result.shape[0], y+h+pad)
    x1, x2 = max(0, x-pad), min(result.shape[1], x+w+pad)
    larger_roi = result[y1:y2, x1:x2]
    
    # Calculate color difference with surrounding area
    surrounding_mean = np.mean(larger_roi, axis=(0,1))
    color_diff = np.abs(surrounding_mean - mean_color)
    
    # If the region color is significantly different from surroundings,
    # consider it occupied
    return np.any(color_diff > color_threshold)

def find_valid_position(result, char_w, char_h, line_height, is_horizontal):
    height, width = result.shape[:2]
    max_attempts = 50
    
    # Scale down if character is too large
    scale = 1.0
    min_scale = 0.1  # Minimum scale to prevent scaling down too much
    while char_w >= width or char_h >= height:
        width_scale = width * 0.9 / char_w
        height_scale = height * 0.9 / char_h
        scale = min(width_scale, height_scale)
        char_w = max(1, int(char_w * scale))
        char_h = max(1, int(char_h * scale))
    
    # Ensure we have valid ranges for random positions
    max_x = max(0, width - char_w)   # No need to subtract 1
    max_y = max(0, height - char_h)  # No need to subtract 1
    
    for _ in range(max_attempts):
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        if x + char_w <= width and y + char_h <= height:
            if not is_space_occupied(result, x, y, char_w, char_h):
                return x, y, scale
    
    # If we couldn't find a valid position after max attempts,
    # keep scaling down until minimum scale is reached
    while scale > min_scale:
        scale *= 0.9  # Scale down by 10%
        new_w = max(1, int(char_w * scale))
        new_h = max(1, int(char_h * scale))
        
        for _ in range(max_attempts):
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            
            if x + new_w <= width and y + new_h <= height:
                if not is_space_occupied(result, x, y, new_w, new_h):
                    return x, y, scale
    
    # If still no valid position, return default
    return 0, 0, scale

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[255, 255, 255]
    )
    return rotated

def random_rotate(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)   
    #Scan document with white background if rotated a bit
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    return rotated

def load_random_char():
    char_path = Path('AlphaDigits')
    char_files = list(char_path.glob('*.png'))
    if not char_files:
        raise FileNotFoundError("No PNG files found in AlphaDigits directory")
    
    # Randomly select a character file
    img_path = random.choice(char_files)
    img = cv2.imread(str(img_path))
    if img is not None:
        clean_char, mask = remove_background(img)
        return clean_char, mask
    return None, None

def load_all_chars():
    
    char_path = Path('/Users/raylau/LocalEnv/AlphaDigits')
    char_files = list(char_path.glob('*.png'))
    if not char_files:
        raise FileNotFoundError("No PNG files found in AlphaDigits directory")
    
    # Randomly select characters from the directory instead of all 700k characters
    selected_files = random.sample(char_files, 2000)
    #selected_files = random.sample(char_files, min(12, len(char_files)))
    
    chars = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for img_path in selected_files:
            futures.append(executor.submit(cv2.imread, str(img_path)))
        
        for future in as_completed(futures):
            img = future.result()
            if img is not None:
                clean_char, mask = remove_background(img)
                chars.append((clean_char, mask))

    return chars

def add_random_chars(image, num_chars, char_pool):
    result = image.copy()
    height, width = result.shape[:2]
    line_height = height // 20
    max_attempts_per_group = 5  # Max attempts to place a group
    
    chars_remaining = num_chars
    while chars_remaining > 0:
        # Randomly determine group size between 10 and 30, but not exceeding remaining chars
        max_group_possible = 20
        group_size = random.randint(5, max_group_possible)
        group_size = min(group_size, chars_remaining)
        chars_remaining -= group_size
        
        group_chars = []
        for _ in range(group_size):
            char_img, char_mask = random.choice(char_pool)
            if char_img is None:
                continue
            
            # Random scaling
            scale_factor = random.uniform(0.05, 0.3)
            new_w = max(1, int(char_img.shape[1] * scale_factor))
            new_h = max(1, int(char_img.shape[0] * scale_factor))
            
            char_img = cv2.resize(char_img, (new_w, new_h))
            char_mask = cv2.resize(char_mask.astype(np.uint8), (new_w, new_h)).astype(bool)
            
            group_chars.append((char_img, char_mask))
        
        if not group_chars:
            continue
        
        # Decide append direction for this group (95% horizontal)
        is_horizontal = random.random() < 0.95
        
        combined_char, combined_mask = group_chars[0]
        for i in range(1, len(group_chars)):
            next_char_img, next_mask = group_chars[i]
            
            # Append images with chosen direction
            if is_horizontal:
                # Horizontal appending logic (same as before)
                h1, w1 = combined_char.shape[:2]
                h2, w2 = next_char_img.shape[:2]
                max_height = max(h1, h2)
                
                # Resize current combined_char and mask to max height
                if h1 != max_height:
                    scale = max_height / h1
                    new_w = int(w1 * scale)
                    combined_char = cv2.resize(combined_char, (new_w, max_height))
                    combined_mask = cv2.resize(combined_mask.astype(np.uint8), (new_w, max_height)) > 0
                
                # Resize next_char_img and mask to max height
                if h2 != max_height:
                    scale = max_height / h2
                    new_w = int(w2 * scale)
                    next_char_img = cv2.resize(next_char_img, (new_w, max_height))
                    next_mask = cv2.resize(next_mask.astype(np.uint8), (new_w, max_height)) > 0
                
                combined_char = np.hstack((combined_char, next_char_img))
                combined_mask = np.hstack((combined_mask, next_mask))
            else:
                # Vertical appending logic (same as before)
                h1, w1 = combined_char.shape[:2]
                h2, w2 = next_char_img.shape[:2]
                max_width = max(w1, w2)
                
                # Resize current combined_char and mask to max width
                if w1 != max_width:
                    scale = max_width / w1
                    new_h = int(h1 * scale)
                    combined_char = cv2.resize(combined_char, (max_width, new_h))
                    combined_mask = cv2.resize(combined_mask.astype(np.uint8), (max_width, new_h)) > 0
                
                # Resize next_char_img and mask to max width
                if w2 != max_width:
                    scale = max_width / w2
                    new_h = int(h2 * scale)
                    next_char_img = cv2.resize(next_char_img, (max_width, new_h))
                    next_mask = cv2.resize(next_mask.astype(np.uint8), (max_width, new_h)) > 0
                
                combined_char = np.vstack((combined_char, next_char_img))
                combined_mask = np.vstack((combined_mask, next_mask))
        
        # Apply color variation
        combined_char = random_color_variation(combined_char, combined_mask)
        
        char_h, char_w = combined_char.shape[:2]
        attempts = 0
        placed = False
        
        # Try to find valid position with limited attempts
        while attempts < max_attempts_per_group and not placed:
            x, y, scale = find_valid_position(result, char_w, char_h, line_height, is_horizontal)
            
            # Scale if needed
            if scale < 1.0:
                new_w = max(1, int(char_w * scale))
                new_h = max(1, int(char_h * scale))
                scaled_char = cv2.resize(combined_char, (new_w, new_h))
                scaled_mask = cv2.resize(combined_mask.astype(np.uint8), (new_w, new_h)).astype(bool)
            else:
                scaled_char = combined_char
                scaled_mask = combined_mask
                new_w, new_h = char_w, char_h
            
            # Apply rotation
            rotated_char = random_rotate(scaled_char, random.uniform(-3, 3))
            
            # Initialize a counter for characters in the current group
            char_count = 0
            
            while attempts < max_attempts_per_group:
                if x + new_w <= width and y + new_h <= height:
                    # Check if the area is occupied by other characters
                    if not is_space_occupied(result, x, y, new_w, new_h):
                        # Update only non-white pixels in the result
                        mask = rotated_char != [255, 255, 255]
                        result[y:y+new_h, x:x+new_w][mask] = rotated_char[mask]
                        placed = True
                        char_count += 1  # Increment the character count

                attempts += 1  # Increment attempts
        
        # Break loop if placement fails after attempts
        if not placed:
            break
    
    return result

def process_single_image(args):
    try:
        clean_path, i, j, raw_dir, char_pool = args
        
        # Read the image inside the thread
        original = cv2.imread(str(clean_path))
        original = random_rotate(original, random.uniform(-6, 6)) #Rotate the image
        if original is None:
            print(f"Warning: Could not read image {clean_path}")
            return i, j, False
        
        # Check if char_pool is empty or None
        if not char_pool:
            print(f"Warning: Character pool is empty")
            return i, j, False
            
        # Generate noisy version with random number of characters
        image_area = original.shape[0] * original.shape[1]
        
        num_chars = random.randint(100, 250)
        #print(f"num_chars: {num_chars}")
        
        augmented = add_random_chars(original.copy(), num_chars, char_pool)
        
        # Save the augmented version with noise
        aug_path = raw_dir / f'raw_{i}_{j+1}.jpg'
        cv2.imwrite(str(aug_path), augmented)
        
        return i, j, True
    except Exception as e:
        print(f"Error in process_single_image: {str(e)}")
        return i, j, False

def main():
    raw_dir = Path('raw_images')
    raw_dir.mkdir(exist_ok=True)
    
    clean_dir = Path('raw_clean')
    
    # Load all characters in parallel at startup
    print("Loading character pool...")
    char_pool = load_all_chars()
    print(f"Loaded {len(char_pool)} characters")
    
    # Just collect file paths and create tasks
    tasks = []
    for clean_path in clean_dir.glob('raw_clean_*.jpg'):
        try:
            # Extract index from filename (raw_X_clean.jpg)
            i = int(clean_path.stem.split('_')[2].replace('.jpg', ''))
            # Sample image for each clean image
            for j in range(1):
                tasks.append((clean_path, i, j, raw_dir, char_pool))
                
        except Exception as e:
            print(f"Error processing {clean_path}: {e}")
            continue
    
    # Process images using thread pool
    print(f"Processing {len(tasks)} image variations...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                i, j, success = future.result()
                if success:
                    tqdm.write(f'Processed variation {j+1} for image {i}')
                else:
                    tqdm.write(f'Failed to process variation {j+1} for image {i}')
            except Exception as e:
                tqdm.write(f'Error processing task: {e}')

if __name__ == '__main__':
    main()