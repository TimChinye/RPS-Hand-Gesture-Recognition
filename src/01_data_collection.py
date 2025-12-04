# src/01_data_collection.py

import cv2
import os
import time

def main():
    """
    An interactive, 1st-class script for collecting a Rock-Paper-Scissors dataset.

    Features:
    - On-the-fly class switching with single key presses.
    - An intuitive, mirrored camera feed.
    - A clear on-screen HUD showing the current mode, image count, and delay setting.
    - An adjustable self-timer countdown before each capture.
    - Resume-friendly counting that picks up where you left off.
    - Professional code structure and robust file naming.
    """
    # --- Constants and Configuration ---
    DATASET_DIR = os.path.join('dataset')
    CLASSES = ['rock', 'paper', 'scissors', 'none']
    
    # Capture settings
    CAM_INDEX = 0          # 0 for default webcam, change if you have multiple cameras
    
    # --- Dynamic Settings (can be changed during runtime) ---
    countdown_duration = 3 # Default self-timer duration in seconds

    # --- Setup ---
    print("--- Rock-Paper-Scissors Data Collection (1st Class Edition) ---")
    
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        os.makedirs(path, exist_ok=True)
    
    print(f"Dataset directory structured at: {os.path.abspath(DATASET_DIR)}")

    try:
        count = {cls: len(os.listdir(os.path.join(DATASET_DIR, cls))) for cls in CLASSES}
        print("Resuming from current image counts:", count)
    except FileNotFoundError:
        print("Could not find existing dataset. Starting counts from zero.")
        count = {cls: 0 for cls in CLASSES}

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam. Is it connected and not in use?")
        return

    # --- State Variables for Countdown ---
    is_counting_down = False
    countdown_start_time = 0

    # --- Main Application Loop ---
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
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --- Countdown Logic ---
        if is_counting_down:
            elapsed_time = time.time() - countdown_start_time
            remaining_time = countdown_duration - elapsed_time

            # --- Display Countdown Number ---
            if remaining_time > 0:
                countdown_text = str(int(remaining_time) + 1)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            else:
                # --- Capture Image when Countdown Finishes ---
                img_name = f"{current_class}_{count[current_class]:04d}.png"
                save_path = os.path.join(DATASET_DIR, current_class, img_name)
                cv2.imwrite(save_path, frame)
                
                print(f"Captured: {save_path}")
                count[current_class] += 1
                is_counting_down = False # Reset the countdown state

        # --- On-Screen Display (HUD) ---
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

        # --- Keypress Handling ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit key pressed. Shutting down.")
            break
        elif key == ord('r'):
            current_class = 'rock'
            print(f"Switched to mode: {current_class}")
        elif key == ord('p'):
            current_class = 'paper'
            print(f"Switched to mode: {current_class}")
        elif key == ord('s'):
            current_class = 'scissors'
            print(f"Switched to mode: {current_class}")
        elif key == ord('n'):
            current_class = 'none'
            print(f"Switched to mode: {current_class}")
        elif key == ord('c') and not is_counting_down:
            # Start the countdown only if not already counting
            is_counting_down = True
            countdown_start_time = time.time()
        elif key == ord('+') or key == ord('='):
            countdown_duration += 1
            print(f"Capture delay increased to {countdown_duration} seconds.")
        elif key == ord('-') or key == ord('_'):
            if countdown_duration > 0:
                countdown_duration -= 1
                print(f"Capture delay decreased to {countdown_duration} seconds.")

    # --- Cleanup ---
    print("\nData collection finished.")
    print("Final image counts:", count)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()