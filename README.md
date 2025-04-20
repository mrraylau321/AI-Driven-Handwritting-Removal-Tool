# AI-Driven Handwriting Removal Tool

This repository contains an AI-driven pipeline for removing handwritten annotations from scanned documents, preserving the underlying printed text and structure. The project implements a multi-stage approach combining rotation correction, high-resolution Attention U-Net processing, and super-resolution restoration to achieve high-quality document cleaning.

## Features

- **Rotation Correction**: Automatically detects and corrects document misalignment
- **Handwriting Removal**: Separates and removes handwritten annotations from printed text
- **Super-Resolution**: Enhances document clarity and readability post-processing
- **GUI Interface**: Simple drag-and-drop interface for document processing

## Project Structure

- `GUI_visualization_tool.py`: Main application with wxPython interface for document processing
- `angle_regression.py`: CNN model for document rotation angle prediction
- `ArSSR_model_original.py`: Custom Arbitrary Scale Super-Resolution model
- `model_training_256256.py` & `model_training_880880.py`: Training scripts for Attention U-Net models
- `evaluate_metrics.py`: Evaluation metrics implementation (MAE, SSIM)
- `generate_images_for_metrics.py`: Script for generating test images for evaluation
- `visualization_combined_result_Public_pretrained.py`: Visualization tool for pipeline outputs

## Dataset Generation

The project uses synthetic datasets created by combining clean document templates with simulated handwritten annotations:

### Handwritten Elements
- **Alphabet Dataset**: Used to simulate diverse handwritten text annotations with natural variations in stroke width, character spacing, and writing styles
- **MNIST Dataset**: Incorporated for numerical annotations, providing realistic digit formations commonly found in form completions and numerical markups

### Data Augmentation
The handwritten elements from these datasets were applied to clean documents with:
- Varied ink colors (blue, black, dark green, purple)
- Random rotations to simulate natural writing angles
- Pixel-level variations in character placement
- Diverse stroke widths and intensities to mimic different writing implements
- Contextual placement patterns (95% horizontal, 5% vertical arrangements)

This approach created a comprehensive training dataset of approximately 10,000 document pairs that realistically simulates human annotation behaviors.

## Models

### Rotation Correction
A lightweight CNN predicts document rotation angles with average error < 1 degree.

### Handwriting Removal
Enhanced Attention U-Net architecture processes high-resolution (880x880) inputs to accurately differentiate between handwritten annotations and printed text.

### Super-Resolution
Three models were evaluated:
- **RealESRGAN**: For handling complex degradations (loading time: 1-2 seconds)
- **ESPCN**: Efficient model with sub-pixel convolution (loading time: <1 second)
- **ARSSR**: Custom model adapted for 2D document processing (loading time: >10 seconds)

ESPCN was selected for the final implementation due to its optimal balance of restoration quality (SSIM: 0.78) and computational efficiency.

## Installation & Usage

1. Clone the repository:
```
git clone https://github.com/mrraylau321/AI-Driven-Handwritting-Removal-Tool.git
cd AI-Driven-Handwritting-Removal-Tool
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Run the GUI application:
```
python GUI_visualization_tool.py
```

4. Drag and drop a document with handwritten annotations into the application window for processing.

## Evaluation Results

Performance was assessed using:
- **Mean Absolute Error (MAE)**: Quantifying pixel-level differences
- **Structural Similarity Index Measure (SSIM)**: Evaluating perceptual image quality
- **Visual Inspection**: Qualitative assessment of readability and text preservation

## Future Work

- OCR integration for text extraction and more precise evaluation
- Real-world dataset collection for improved model robustness
- Implementation of advanced architectures like diffusion models
- Patch-based processing for handling larger documents
- Controllable processing for selective content manipulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
