# src/utils/review.py
import cv2
import mediapipe as mp
from pathlib import Path
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
REVIEW_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_review"
GOOD_CROP_DIR = REVIEW_DIR / "good_crop"
KEEP_ORIGINAL_DIR = REVIEW_DIR / "keep_original"
DISCARD_DIR = REVIEW_DIR / "discard"
PADDING = 25

# --- Global variables for manual cropping state ---
ref_point = []
cropping = False
image_to_show = None

def manual_crop_callback(event, x, y, flags, param):
    """Callback function for mouse events for manual cropping."""
    global ref_point, cropping, image_to_show
    original_image = param[0]

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp_image = image_to_show.copy()
        cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Manual Review", temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        image_to_show = original_image.copy()
        cv2.rectangle(image_to_show, ref_point[0], ref_point[1], (0, 255, 0), 2)
        instructions = "[c]rop | [k]eep | [r]eset | [d]iscard & replace"
        cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def run_review():
    """Main function to run the interactive review and cropping pipeline."""
    global image_to_show, ref_point
    print("--- Starting Interactive Dataset Review Pipeline ---")

    for d in [GOOD_CROP_DIR, KEEP_ORIGINAL_DIR, DISCARD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for class_name in SOURCE_DIR.iterdir():
            if class_name.is_dir():
                (d / class_name.name).mkdir(exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    total_images = sum(1 for p in SOURCE_DIR.rglob("*.png"))
    processed_count = 0
    
    # This will store the path to the last successfully processed image for each class
    last_valid_path_per_class = {}

    for class_path in sorted(SOURCE_DIR.iterdir()):
        if not class_path.is_dir():
            continue
        
        print(f"\nProcessing class: {class_path.name}")
        
        for image_path in sorted(list(class_path.glob("*.png"))):
            processed_count += 1
            print(f"[{processed_count}/{total_images}] Processing {image_path.name}...")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print("  -> Could not read image. Moving to discard.")
                shutil.move(str(image_path), DISCARD_DIR / class_path.name / image_path.name)
                continue

            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                # --- AUTO-CROP PATH ---
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = image.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = max(0, int(min(x_coords)) - PADDING), min(w, int(max(x_coords)) + PADDING)
                y_min, y_max = max(0, int(min(y_coords)) - PADDING), min(h, int(max(y_coords)) + PADDING)
                
                cropped_image = image[y_min:y_max, x_min:x_max]
                save_path = GOOD_CROP_DIR / class_path.name / image_path.name
                if cropped_image.size > 0:
                    cv2.imwrite(str(save_path), cropped_image)
                    last_valid_path_per_class[class_path.name] = save_path
                    print("  -> Hand found. Auto-cropped successfully.")
                else:
                    print("  -> Auto-crop failed. Discarding and replacing.")
                    shutil.move(str(image_path), DISCARD_DIR / class_path.name / image_path.name)
                    if class_path.name in last_valid_path_per_class:
                        shutil.copy(last_valid_path_per_class[class_path.name], save_path)
            else:
                # --- MANUAL REVIEW PATH ---
                print("  -> No hand detected. MANUAL REVIEW REQUIRED.")
                image_to_show = image.copy()
                original_image = image.copy()
                window_name = "Manual Review"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_name, manual_crop_callback, [original_image])
                instructions = "[c]rop | [k]eep | [d]iscard+replace | [r]eset"
                cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                while True:
                    cv2.imshow(window_name, image_to_show)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('k'): # KEEP
                        save_path = KEEP_ORIGINAL_DIR / class_path.name / image_path.name
                        shutil.copy(str(image_path), save_path)
                        last_valid_path_per_class[class_path.name] = save_path
                        print("    -> User choice: KEEP original.")
                        break
                        
                    elif key == ord('d'): # DISCARD AND REPLACE
                        print("    -> User choice: DISCARD and REPLACE.")
                        shutil.move(str(image_path), DISCARD_DIR / class_path.name / image_path.name)
                        # Check if we have a 'last valid' image for this class to duplicate
                        if class_path.name in last_valid_path_per_class:
                            last_good_image = last_valid_path_per_class[class_path.name]
                            # Decide where the replacement should go based on the last good one's location
                            dest_folder = last_good_image.parent.parent
                            final_dest_folder = REVIEW_DIR / dest_folder.name / class_path.name
                            replacement_path = final_dest_folder / image_path.name # Use discarded image's name
                            shutil.copy(last_good_image, replacement_path)
                            print(f"    -> Replaced with a copy of {last_good_image.name}.")
                        else:
                            print("    -> No previous valid image in this class to replace with. Image discarded.")
                        break
                        
                    elif key == ord('r'): # RESET
                        image_to_show = original_image.copy()
                        cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        ref_point = []
                        print("    -> Reset crop box.")
                        
                    elif key == ord('c') and len(ref_point) == 2: # CROP
                        start_x, end_x = sorted([ref_point[0][0], ref_point[1][0]])
                        start_y, end_y = sorted([ref_point[0][1], ref_point[1][1]])
                        roi = original_image[start_y:end_y, start_x:end_x]
                        save_path = GOOD_CROP_DIR / class_path.name / image_path.name
                        if roi.size > 0:
                            cv2.imwrite(str(save_path), roi)
                            last_valid_path_per_class[class_path.name] = save_path
                            print("    -> User choice: SAVED manual crop.")
                        else:
                            # If manual crop fails, discard and replace
                            print("    -> Manual crop failed. Discarding and replacing.")
                            shutil.move(str(image_path), DISCARD_DIR / class_path.name / image_path.name)
                            if class_path.name in last_valid_path_per_class:
                                last_good_image = last_valid_path_per_class[class_path.name]
                                dest_folder = last_good_image.parent.parent
                                final_dest_folder = REVIEW_DIR / dest_folder.name / class_path.name
                                replacement_path = final_dest_folder / image_path.name
                                shutil.copy(last_good_image, replacement_path)
                                print(f"    -> Replaced with a copy of {last_good_image.name}.")
                            else:
                                print("    -> No previous valid image in this class to replace with. Image discarded.")
                        break
                
                cv2.destroyAllWindows()
                ref_point = []

    hands.close()
    print("\n--- Review Pipeline Complete ---")

if __name__ == '__main__':
    run_review()