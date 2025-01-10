import os
import shutil
import random

def split_dataset(source_path, output_path, train_ratio=0.9, seed=42):
    """
    Split dataset into train and test sets
    
    Parameters:
    - source_path: Path to original UIEB dataset
    - output_path: Path where train/test directories will be created
    - train_ratio: Proportion of images for training (default 80%)
    - seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'train', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train', 'ref'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'ref'), exist_ok=True)
    
    # Get all image filenames from raw directory
    raw_images = os.listdir(os.path.join(source_path, 'raw'))
    
    # Shuffle images
    random.shuffle(raw_images)
    
    # Calculate split index
    split_index = int(len(raw_images) * train_ratio)
    
    # Split images
    train_images = raw_images[:split_index]
    test_images = raw_images[split_index:]
    
    # Copy train images
    for img in train_images:
        # Copy raw image
        shutil.copy(
            os.path.join(source_path, 'raw', img),
            os.path.join(output_path, 'train', 'raw', f'im{train_images.index(img)+1}.png')
        )
        
        # Copy reference image
        shutil.copy(
            os.path.join(source_path, 'reference', img),
            os.path.join(output_path, 'train', 'ref', f'im{train_images.index(img)+1}.png')
        )
    
    # Copy test images
    for img in test_images:
        # Copy raw image
        shutil.copy(
            os.path.join(source_path, 'raw', img),
            os.path.join(output_path, 'test', 'raw', f'im{test_images.index(img)+1}.png')
        )
        
        # Copy reference image
        shutil.copy(
            os.path.join(source_path, 'reference', img),
            os.path.join(output_path, 'test', 'ref', f'im{test_images.index(img)+1}.png')
        )
    
    # Print dataset split information
    print(f"Total images: {len(raw_images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")

# Example usage
source_path = 'D:/vidhi/underwater/datasets/UIEB'
output_path = './data/UIEB'

split_dataset(source_path, output_path)

# import os
# import shutil

# def reorganize_validation_dataset(source_path, output_path):
#     """
#     Reorganize validation dataset into a structured format
    
#     Parameters:
#     - source_path: Path to original EVUP validation directory
#     - output_path: Path where reorganized test directory will be created
#     """
#     # Create output directories
#     os.makedirs(os.path.join(output_path, 'test', 'raw'), exist_ok=True)
    
#     # Get all image filenames from validation directory
#     validation_images = [f for f in os.listdir(source_path) if f.lower().endswith('.jpg')]
    
#     # Sort images to ensure consistent naming
#     validation_images.sort()
    
#     # Copy and rename images
#     for index, img in enumerate(validation_images, 1):
#         # Copy image with new naming convention
#         shutil.copy(
#             os.path.join(source_path, img),
#             os.path.join(output_path, 'test', 'raw', f'im{index}.png')
#         )
    
#     # Print dataset information
#     print(f"Total validation images: {len(validation_images)}")
#     print(f"Images copied to: {os.path.join(output_path, 'test', 'raw')}")

# # Example usage
# source_path = r'D:/vidhi/underwater/datasets/EVUP/validation'
# output_path = r'./data/EVUP'

# # Run the reorganization
# reorganize_validation_dataset(source_path, output_path)