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
import wx
import wx.lib.scrolledpanel as scrolled
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import io
import sys
if os.path.splitext(sys.argv[0])[1] in (".py", ".pyw"):
    temp = os.path.abspath(__file__).split('/')
else:
    temp = sys.executable.split(".app/Contents/MacOS")[0].split('/')
filepath = ''
for pa in range(len(temp)-1):
    filepath = filepath + '/' + temp[pa]

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
    
    # Generate clean image
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    
    # Convert tensor back to PIL image
    denoised_tensor = denoised_tensor.cpu().squeeze(0)
    denoised_img = transforms.ToPILImage()(denoised_tensor)
    
    return denoised_img

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



def super_resolve(image,device, target_size=(3520, 2490), scale_factor=4):
    # Load OpenCV's ESPCN model (x4)
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(filepath+'/ESPCN_x4.pb')  # Download from OpenCV's repo
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


device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')

rotation_model = RotationPredictor().to(device)
rotation_model.load_state_dict(torch.load(filepath+'/DocumentAngleRegressionModel.pth', map_location=device))
rotation_model.eval()

denoising_model = UNetWithAttention(in_channels=3, out_channels=3).to(device)
denoising_model.load_state_dict(torch.load(filepath+'/Drop_text_model_temp.pth', map_location=device))
denoising_model.eval()

# GUI Application
class ImageDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        if filenames:
            self.window.set_image(filenames[0])
        return True

class ImageProcessorApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Document Image Processor", size=(1200, 800))
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Instruction Text
        instruction = wx.StaticText(self.panel, label="Drag and Drop Image Below:", style=wx.ALIGN_CENTER)
        main_sizer.Add(instruction, 0, wx.ALL | wx.EXPAND, 10)

        # Drop Area
        self.drop_area = wx.StaticBitmap(self.panel, size=(350, 350))
        self.drop_area.SetBackgroundColour(wx.Colour(240,240,240))
        drop_target = ImageDropTarget(self)
        self.drop_area.SetDropTarget(drop_target)
        main_sizer.Add(self.drop_area, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        # Generate Button
        self.generate_btn = wx.Button(self.panel, label="Generate")
        self.generate_btn.Bind(wx.EVT_BUTTON, self.on_generate)
        main_sizer.Add(self.generate_btn, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        # Result Display Area (ScrolledPanel)
        self.result_panel = scrolled.ScrolledPanel(self.panel, size=(-1,300), style=wx.SIMPLE_BORDER)
        self.result_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.result_panel.SetSizer(self.result_sizer)
        self.result_panel.SetupScrolling()
        main_sizer.Add(self.result_panel, 1, wx.ALL | wx.EXPAND, 10)

        self.panel.SetSizer(main_sizer)

        self.image_path = None
        self.Centre()
        self.Show()

    def set_image(self, file_path):
        self.image_path = file_path
        img = wx.Image(file_path, wx.BITMAP_TYPE_ANY)

        # Get original image size
        orig_width, orig_height = img.GetWidth(), img.GetHeight()

        # Get drop area size
        area_width, area_height = self.drop_area.GetSize()

        # Calculate scaling to maintain aspect ratio
        aspect_ratio = orig_width / orig_height
        new_width, new_height = area_width, area_height

        if aspect_ratio > area_width / area_height:
            # Width is the limiting factor
            new_height = area_width / aspect_ratio
        else:
            # Height is the limiting factor
            new_width = area_height * aspect_ratio

        scaled_img = img.Scale(int(new_width), int(new_height), wx.IMAGE_QUALITY_HIGH)

        # Create a bitmap with background color for centering
        bitmap = wx.Bitmap(area_width, area_height)
        dc = wx.MemoryDC(bitmap)
        dc.SetBrush(wx.Brush(wx.Colour(240, 240, 240)))
        dc.DrawRectangle(0, 0, area_width, area_height)

        # Center the scaled image on the bitmap
        x_offset = (area_width - int(new_width)) // 2
        y_offset = (area_height - int(new_height)) // 2
        dc.DrawBitmap(wx.Bitmap(scaled_img), x_offset, y_offset, True)
        dc.SelectObject(wx.NullBitmap)

        self.drop_area.SetBitmap(bitmap)
        self.Layout()

    def on_generate(self, event):
        if not self.image_path:
            wx.MessageBox("Please drop an image first!", "Error", wx.OK | wx.ICON_ERROR)
            return

        wx.BeginBusyCursor()
        wx.Yield()
        try:
            # Step 1: Original Image
            original_img = Image.open(self.image_path).convert('RGB')

            # Step 2: Rotation Correction
            rotated_img, angle = predict_and_fix_rotation(self.image_path, rotation_model, device)

            # Step 3: Denoising
            denoised_img = denoise_image(rotated_img, denoising_model, device)

            # Step 4: Super-resolution
            final_img = super_resolve(denoised_img, device)

            # Save final image
            output_path = "final_processed_image.jpg"
            final_img.save(output_path)

            # Display results
            self.display_results([("Original", original_img), 
                                  (f"Rotated ({angle:.2f}°)", rotated_img), 
                                  ("Denoised", denoised_img), 
                                  ("Super-Resolved", final_img)])

            wx.MessageBox(f"Processing completed!\nFinal image saved as {output_path}", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"An error occurred:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.EndBusyCursor()

    def display_results(self, images):
        self.result_sizer.Clear(delete_windows=True)

        for title, pil_img in images:
            img_bitmap = self.pil_to_wxbitmap(pil_img, size=(500,500))
            vbox = wx.BoxSizer(wx.VERTICAL)
            bmp = wx.StaticBitmap(self.result_panel, bitmap=img_bitmap)
            lbl = wx.StaticText(self.result_panel, label=title)
            vbox.Add(bmp, 0, wx.ALL|wx.CENTER, 5)
            vbox.Add(lbl, 0, wx.ALL|wx.CENTER, 5)
            self.result_sizer.Add(vbox, 0, wx.ALL, 10)

        self.result_panel.Layout()
        self.result_panel.SetupScrolling(scroll_x=True, scroll_y=False)

    def pil_to_wxbitmap(self, pil_img, size=(350,350)):
        pil_img = pil_img.resize(size, Image.LANCZOS)
        width, height = pil_img.size
        return wx.Bitmap.FromBuffer(width, height, pil_img.tobytes())

if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageProcessorApp()
    app.MainLoop()