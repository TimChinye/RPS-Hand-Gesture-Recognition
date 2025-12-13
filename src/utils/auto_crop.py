# src/utils/auto_crop.py
import cv2
import mediapipe as mp
from pathlib import Path
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
DEST_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_cropped"
PADDING = 20 # Pixels to add around the hand bounding box

def run_auto_crop():
    """
    Finds hands in the original dataset, crops them with padding,
    and saves them to a new directory.
    """
    print("--- Starting Automatic Hand Cropping Pipeline ---")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    if DEST_DIR.exists():
        print(f"Warning: Destination directory '{DEST_DIR}' already exists. Overwriting.")
        # Optional: uncomment to delete the directory before starting
        # shutil.rmtree(DEST_DIR) 
    
    image_count = 0
    cropped_count = 0

    # Iterate through all class folders in the source directory
    for class_path in SOURCE_DIR.iterdir():
        if not class_path.is_dir():
            continue
            
        dest_class_path = DEST_DIR / class_path.name
        dest_class_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing class: {class_path.name}")
        
        # Iterate through all images in the class folder
        for image_path in class_path.glob("*.png"):
            image_count += 1
            
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  - Could not read {image_path.name}, skipping.")
                continue

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands
            results = hands.process(image_rgb)
            
            # If a hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get bounding box coordinates
                    h, w, _ = image.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x < x_min: x_min = x
                        if x > x_max: x_max = x
                        if y < y_min: y_min = y
                        if y > y_max: y_max = y
                        
                    # Apply padding
                    x_min = max(0, x_min - PADDING)
                    y_min = max(0, y_min - PADDING)
                    x_max = min(w, x_max + PADDING)
                    y_max = min(h, y_max + PADDING)
                    
                    # Crop the image
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    
                    # Save the cropped image to the destination
                    if cropped_image.size > 0:
                        save_path = dest_class_path / image_path.name
                        cv2.imwrite(str(save_path), cropped_image)
                        cropped_count += 1
                    else:
                        print(f"  - Cropping failed for {image_path.name} (zero size), skipping.")
                    
                    # We only process the first hand found
                    break 
            else:
                print(f"  - No hand detected in {image_path.name}, skipping.")
                
    hands.close()
    print("\n--- Cropping Pipeline Complete ---")
    print(f"Total images processed: {image_count}")
    print(f"Successfully cropped and saved: {cropped_count}")
    print(f"New dataset available at: {DEST_DIR.resolve()}")


if __name__ == '__main__':
    run_auto_crop()