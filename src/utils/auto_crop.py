import cv2
import mediapipe as mp
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Configuration
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
DEST_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_cropped"
PADDING = 16 # Pixels to add around the hand bounding box

def run_auto_crop():
    # Finds hands in the original dataset, crops them with padding, and saves them to a new directory.
    
    logger.info("Starting Automatic Hand Cropping Pipeline")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    if DEST_DIR.exists():
        logger.warning(f"Destination directory '{DEST_DIR}' already exists. Contents may be overwritten.")
    
    image_count = 0
    cropped_count = 0

    # Iterate through all class folders in the source directory
    class_folders = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]
    
    for class_path in class_folders:
        dest_class_path = DEST_DIR / class_path.name
        dest_class_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing class: {class_path.name}")
        
        # Iterate through all images in the class folder
        for image_path in class_path.glob("*.png"):
            image_count += 1
            
            # Read the image
            image = cv2.imread(str(image_path))

            if image is None:
                logger.warning(f"Could not read {image_path.name}, skipping.")
                continue

            # Convert the BGR image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image to find hands
            results = hands.process(image_rgb)
            
            # If a hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get bounding box coordinates
                    h, w, _ = image.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        
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
                        logger.warning(f"Cropping failed for {image_path.name} (zero size), skipping.")
                    
                    # Only process the first hand found
                    break 
            else:
                logger.warning(f"No hand detected in {image_path.name}, skipping.")
                
    hands.close()

    logger.info("Cropping Pipeline Complete")
    logger.info(f"Total images processed: {image_count}")
    logger.info(f"Successfully cropped and saved: {cropped_count}")
    logger.info(f"New dataset available at: {DEST_DIR.resolve()}")

if __name__ == '__main__':
    run_auto_crop()