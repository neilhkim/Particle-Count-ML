"""
predict_test_dataset_labels.py

Script to predict particle counts on test images using a trained model.
Usage: python predict_test_dataset_labels.py <test_folder_path>
"""
from PIL import Image
import os
import sys
import torch
import tifffile
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, Grayscale

# Import the model definition from the training script
from train_models import ParticleCounterCNN, calculate_mean_std

def predict_folder(test_folder_path, model_path='particle_count_model.pth'):
    """
    Predict labels for all images in the specified folder
    
    Args:
        test_folder_path: Path to the folder containing test images
        
    Returns:
        Dictionary mapping filenames to predicted class labels
    """
    # Check if the folder exists
    if not os.path.exists(test_folder_path):
        print(f"Error: Folder '{test_folder_path}' not found.")
        return {}
    
    # Check if model file exists
    # model_path = 'particle_count_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return {}
    
    # Get dataset statistics for normalization (same as in training)
    # dataset_path = 'trainset03'  # Use the same statistics as training
    mean, std = calculate_mean_std(test_folder_path)
    print(f"Using dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Set up image transformation pipeline
    image_size = 100  # Same as in training
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Grayscale(num_output_channels=1),
        Normalize((mean,), (std,))
    ])
    
    # Set up device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize and load the model
    model = ParticleCounterCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create a dictionary to store results
    predictions = {}
    class_names = ['0', '1', '2', '3', '4']  # Same class names as in training
    
    # Process each image in the folder
    print(f"Processing images in {test_folder_path}...")
    for filename in os.listdir(test_folder_path):
        file_path = os.path.join(test_folder_path, filename)
        
        # Skip directories and non-image files
        if os.path.isdir(file_path):
            continue
        
        try:
            # Load and preprocess the image
            img = tifffile.imread(file_path)
            
            # Handle different data types for normalization
            if img.dtype == np.uint8:
                normalized_img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                normalized_img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.float32 or img.dtype == np.float64:
                normalized_img = img.astype(np.float32)
                if normalized_img.max() > 1.0:
                    normalized_img /= normalized_img.max()
            else:
                # For other types, normalize to 0-1 range
                min_val, max_val = img.min(), img.max()
                normalized_img = (img.astype(np.float32) - min_val) / (max_val - min_val)
            
            # Convert numpy array to PIL Image
            # If the image is multi-dimensional, convert only the first channel or average
            if len(normalized_img.shape) > 2:
                # Use the first channel or average all channels
                normalized_img = normalized_img[:, :, 0]  # Or np.mean(normalized_img, axis=2)
            
            # Convert to PIL Image (ensure values are in 0-1 range for float32)
            pil_img = Image.fromarray((normalized_img * 255).astype(np.uint8))
            
            # Apply transforms
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Store the prediction
            predictions[filename] = {
                'class': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': confidence
            }
            
            print(f"Predicted {filename}: Class {predicted_class} ('{class_names[predicted_class]}') with {confidence:.4f} confidence")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return predictions

def save_predictions_to_csv(predictions, output_file='predictions.csv'):
    """Save predictions to a CSV file"""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Predicted Class', 'Class Name', 'Confidence'])
        
        for filename, data in predictions.items():
            writer.writerow([
                filename, 
                data['class'], 
                data['class_name'],
                f"{data['confidence']:.4f}"
            ])
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Define a default test folder path for debugging
    # debug_test_folder_path = "scatter-8x"  # Change this to your test folder path
    debug_test_folder_path = "test02"  # Change this to your test folder path
    pth_path = "best_particle_count_model-test05-epoch37.pth"  # Change this to your model path
    
    # Check if running in debug mode
    is_debug_mode = sys.gettrace() is not None  # Check if debugger is active
    
    if len(sys.argv) < 2:
        test_folder_path = debug_test_folder_path
        print(f"Using default test folder: {test_folder_path}")
    else:
        test_folder_path = sys.argv[1]
    # Execute prediction
    print(f"Analyzing images in: {test_folder_path}")
    predictions = predict_folder(test_folder_path, pth_path)
    
    if predictions:
        save_predictions_to_csv(predictions)