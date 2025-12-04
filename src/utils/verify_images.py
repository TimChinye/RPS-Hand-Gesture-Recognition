import os
from PIL import Image
import argparse

def verify_images(dataset_dir):
    """
    Scans through a directory of images and reports any that are corrupt or unreadable.

    Args:
        dataset_dir (str): The path to the root of the original dataset 
                           (e.g., ../dataset)
    """
    print(f"--- Verifying images in: {os.path.abspath(dataset_dir)} ---")
    corrupt_files = []
    
    # Walk through all subdirectories and files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Check for common image extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)  # Open the image file
                    img.verify()  # Verify that it is a valid image
                except (IOError, SyntaxError) as e:
                    print(f"Corrupt file found: {img_path} | Reason: {e}")
                    corrupt_files.append(img_path)
    
    if not corrupt_files:
        print("\nVerification complete. All images seem to be valid!")
    else:
        print(f"\nVerification complete. Found {len(corrupt_files)} corrupt file(s).")
        print("You should delete these files before proceeding with training.")
        
    return corrupt_files

if __name__ == '__main__':
    # Set up to be run from the 'src' directory
    parser = argparse.ArgumentParser(description="Verify images in a dataset.")
    parser.add_argument(
        '--path', 
        type=str, 
        default='../dataset', 
        help="Path to the dataset directory from the current location."
    )
    args = parser.parse_args()
    
    verify_images(args.path)