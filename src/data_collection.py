# src/data_collection.py
import cv2
import os
import time
from pathlib import Path

# --- Robust Pathing & Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CLASSES = ['rock', 'paper', 'scissors', 'none']
CAM_INDEX = 0

def run_collection():
    """
    Main function to run the interactive data collection process.
    This function is imported and called by the main run.py script.
    """
    # ... (The rest of the `main()` function from your 01_data_collection.py goes here) ...
    # IMPORTANT: Change the one line that defines DATASET_DIR inside the function
    # It should now use the globally defined, robust path.
    # The existing line `DATASET_DIR = os.path.join('dataset')` should be DELETED.
    # The code will automatically use the DATASET_DIR defined at the top of this file.

    # --- Setup ---
    print("--- Rock-Paper-Scissors Data Collection ---")
    
    for cls in CLASSES:
        path = DATASET_DIR / cls
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset directory structured at: {DATASET_DIR.resolve()}")
    
    # ... (rest of the function continues as before)
    # The original main function from 01_data_collection.py fits here perfectly.
    # I'll paste the whole thing for clarity.

    try:
        count = {cls: len(os.listdir(DATASET_DIR / cls)) for cls in CLASSES}
        print("Resuming from current image counts:", count)
    except FileNotFoundError:
        print("Could not find existing dataset. Starting counts from zero.")
        count = {cls: 0 for cls in CLASSES}

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        return

    countdown_duration = 3
    is_counting_down = False
    countdown_start_time = 0
    current_class = 'none'

    print("\n" + "="*20 + " CONTROLS " + "="*20)
    print(" [r, p, s, n] -> Switch mode (rock, paper, scissors, none)")
    print(" [c] -> Start capture countdown")
    print(" [+] -> Increase countdown delay")
    print(" [-] -> Decrease countdown delay (min 1s)")
    print(" [q] -> Quit the application")
    print("="*50 + "\n")
    print(f"Starting in '{current_class}' mode. Get ready!")

    while True:
        # ... (rest of the while loop code is identical to your original file)
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if is_counting_down:
            elapsed_time = time.time() - countdown_start_time
            remaining_time = countdown_duration - elapsed_time

            if remaining_time > 0:
                countdown_text = str(int(remaining_time) + 1)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            else:
                img_name = f"{current_class}_{count[current_class]:04d}.png"
                save_path = DATASET_DIR / current_class / img_name
                cv2.imwrite(str(save_path), frame)
                
                print(f"Captured: {save_path}")
                count[current_class] += 1
                is_counting_down = False

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (450, 100), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        hud_text_mode = f"Mode: {current_class.upper()}"
        hud_text_count = f"Count ({current_class}): {count[current_class]}"
        hud_text_delay = f"Delay: {countdown_duration}s ([+]/[-])"
        cv2.putText(frame, hud_text_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, hud_text_count, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, hud_text_delay, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Data Collection - Press 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            current_class = 'rock'
        elif key == ord('p'):
            current_class = 'paper'
        elif key == ord('s'):
            current_class = 'scissors'
        elif key == ord('n'):
            current_class = 'none'
        elif key == ord('c') and not is_counting_down:
            is_counting_down = True
            countdown_start_time = time.time()
        elif key == ord('+') or key == ord('='):
            countdown_duration += 1
        elif key == ord('-') or key == ord('_'):
            if countdown_duration > 1:
                countdown_duration -= 1
    
    cap.release()
    cv2.destroyAllWindows()


# This allows the script to be run directly for debugging if needed
if __name__ == "__main__":
    run_collection()