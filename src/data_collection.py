import cv2
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Configuration and Constants
CLASSES = ['rock', 'paper', 'scissors', 'none']
CAM_INDEX = 0

# Robust pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Map keyboard keys to class names
CLASS_KEY_MAP = {
    ord('r'): 'rock',
    ord('p'): 'paper',
    ord('s'): 'scissors',
    ord('n'): 'none'
}

def print_controls(current_class):
    # Prints the control menu to the console
    print("\n" + "="*20 + " CONTROLS " + "="*20)
    print(" [r, p, s, n] -> Switch mode (rock, paper, scissors, none)")
    print(" [c] -> Start capture countdown")
    print(" [+] -> Increase countdown delay")
    print(" [-] -> Decrease countdown delay")
    print(" [q] -> Quit the application")
    print("="*50 + "\n")
    print(f"Starting in '{current_class}' mode. Get ready!")

def run_collection():
    # Main function to run the interactive data collection process
    logger.info("Starting Rock-Paper-Scissors Data Collection")

    # Ensure dataset directories exist
    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset directory structured at: {DATASET_DIR.resolve()}")

    # Initialize counts
    try:
        count = {cls: len(os.listdir(DATASET_DIR / cls)) for cls in CLASSES}
        logger.info(f"Resuming from current image counts: {count}")

    except FileNotFoundError:
        logger.warning("Could not find existing dataset. Starting counts from zero.")
        count = {cls: 0 for cls in CLASSES}

    # Initialize Camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        logger.error("FATAL ERROR: Could not open webcam.")
        return

    # Application State
    countdown_duration = 3
    is_counting_down = False
    countdown_start_time = 0
    current_class = 'none'

    # Show UI controls
    print_controls(current_class)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Failed to capture frame from webcam. Exiting.")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Handle Countdown Logic
        if is_counting_down:
            elapsed_time = time.time() - countdown_start_time
            remaining_time = countdown_duration - elapsed_time

            if remaining_time > 0:
                # Draw countdown number
                countdown_text = str(int(remaining_time) + 1)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)

            else:
                # Capture Image
                img_name = f"{current_class}_{count[current_class]:04d}.png"
                save_path = DATASET_DIR / current_class / img_name
                cv2.imwrite(str(save_path), frame)
                
                logger.info(f"Captured: {save_path}")
                count[current_class] += 1
                is_counting_down = False

        # Draw HUD
        alpha = 0.6
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (450, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        hud_lines = [
            (f"Mode: {current_class.upper()}", (10, 30), (0, 255, 0)),
            (f"Count ({current_class}): {count[current_class]}", (10, 60), (255, 255, 255)),
            (f"Delay: {countdown_duration}s ([+]/[-])", (10, 90), (255, 255, 255))
        ]

        for text, pos, color in hud_lines:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Data Collection - Press 'q' to quit", frame)

        # Handle Inputs
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        # Check class switching via map
        if key in CLASS_KEY_MAP:
            current_class = CLASS_KEY_MAP[key]
        
        elif key == ord('c') and not is_counting_down:
            is_counting_down = True
            countdown_start_time = time.time()
        
        elif key in [ord('+'), ord('=')]:
            countdown_duration += 1
        
        elif key in [ord('-'), ord('_')]:
            if countdown_duration > 0:
                countdown_duration -= 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_collection()