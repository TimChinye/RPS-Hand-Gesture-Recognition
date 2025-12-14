import cv2
import mediapipe as mp
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Configuration
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
REVIEW_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_review"
GOOD_CROP_DIR = REVIEW_DIR / "good_crop"
KEEP_ORIGINAL_DIR = REVIEW_DIR / "keep_original"
DISCARD_DIR = REVIEW_DIR / "discard"
PADDING = 24

# Global variables for manual cropping state, required by the OpenCV callback
ref_point = []
cropping = False
image_to_show = None

def manual_crop_callback(event, x, y, flags, param):
    # Callback function for mouse events for manual cropping
    
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
        instructions = "[c]rop | [k]eep | [r]eset | [d]iscard & replace"
        
        cv2.rectangle(image_to_show, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def run_review():
    # Main function to run the interactive review and cropping pipeline
    global image_to_show, ref_point
    logger.info("Starting Interactive Dataset Review Pipeline")

    # Setup review directories
    review_dirs = [GOOD_CROP_DIR, KEEP_ORIGINAL_DIR, DISCARD_DIR]
    for d in review_dirs:
        d.mkdir(parents=True, exist_ok=True)

        for class_path in SOURCE_DIR.iterdir():
            if class_path.is_dir():
                (d / class_path.name).mkdir(exist_ok=True)

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    image_paths = sorted(list(SOURCE_DIR.rglob("*.png")))
    total_images = len(image_paths)
    
    # Stores the path to the last good image for replacing failed ones
    last_valid_path_per_class = {}

    for i, image_path in enumerate(image_paths):
        processed_count = i + 1
        class_name = image_path.parent.name
        logger.info(f"[{processed_count}/{total_images}] Processing {image_path.name} in class '{class_name}'...")
        
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read image. Moving to discard.")

            shutil.move(str(image_path), DISCARD_DIR / class_name / image_path.name)
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            # Auto-crop path
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = image.shape

            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min = max(0, int(min(x_coords)) - PADDING)
            x_max = min(w, int(max(x_coords)) + PADDING)
            y_min = max(0, int(min(y_coords)) - PADDING)
            y_max = min(h, int(max(y_coords)) + PADDING)
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            save_path = GOOD_CROP_DIR / class_name / image_path.name

            if cropped_image.size > 0:
                cv2.imwrite(str(save_path), cropped_image)

                last_valid_path_per_class[class_name] = save_path
                logger.info("  -> Hand found. Auto-cropped successfully.")

            else:
                logger.warning("  -> Auto-crop failed. Discarding and replacing.")
                shutil.move(str(image_path), DISCARD_DIR / class_name / image_path.name)

                if class_name in last_valid_path_per_class:
                    shutil.copy(last_valid_path_per_class[class_name], save_path)
        else:
            # Manual Review Path
            logger.info("  -> No hand detected. MANUAL REVIEW REQUIRED.")

            image_to_show = image.copy()
            original_image = image.copy()

            window_name = "Manual Review"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, manual_crop_callback, [original_image])

            instructions = "[c]rop | [k]eep | [d]iscard+replace | [r]eset"
            cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            action_taken = False
            while not action_taken:
                cv2.imshow(window_name, image_to_show)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('k'): # KEEP original
                    save_path = KEEP_ORIGINAL_DIR / class_name / image_path.name
                    shutil.copy(str(image_path), save_path)

                    last_valid_path_per_class[class_name] = save_path
                    logger.info("    -> User choice: KEEP original.")

                    action_taken = True
                    
                elif key == ord('d'): # DISCARD and REPLACE
                    logger.info("    -> User choice: DISCARD and REPLACE.")
                    shutil.move(str(image_path), DISCARD_DIR / class_name / image_path.name)

                    if class_name in last_valid_path_per_class:
                        last_good_path = last_valid_path_per_class[class_name]

                        # Determine destination based on where the last good one came from
                        dest_folder_type = last_good_path.parent.parent.name
                        replacement_dest = REVIEW_DIR / dest_folder_type / class_name / image_path.name

                        shutil.copy(last_good_path, replacement_dest)
                        logger.info(f"    -> Replaced with a copy of {last_good_path.name}.")

                    else:
                        logger.warning("    -> No previous valid image in this class to replace with. Image discarded.")

                    action_taken = True
                    
                elif key == ord('r'): # RESET crop box
                    image_to_show = original_image.copy()
                    cv2.putText(image_to_show, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    ref_point = []
                    logger.info("    -> Reset crop box.")
                    
                elif key == ord('c') and len(ref_point) == 2: # CROP manually
                    start_x, end_x = sorted([ref_point[0][0], ref_point[1][0]])
                    start_y, end_y = sorted([ref_point[0][1], ref_point[1][1]])
                    roi = original_image[start_y:end_y, start_x:end_x]
                    save_path = GOOD_CROP_DIR / class_name / image_path.name

                    if roi.size > 0:
                        cv2.imwrite(str(save_path), roi)

                        last_valid_path_per_class[class_name] = save_path
                        logger.info("    -> User choice: SAVED manual crop.")

                    else:
                        logger.warning("    -> Manual crop failed. Discarding and replacing.")
                        shutil.move(str(image_path), DISCARD_DIR / class_name / image_path.name)

                        if class_name in last_valid_path_per_class:
                            last_good_path = last_valid_path_per_class[class_name]
                            dest_folder_type = last_good_path.parent.parent.name
                            replacement_dest = REVIEW_DIR / dest_folder_type / class_name / image_path.name

                            shutil.copy(last_good_path, replacement_dest)
                            logger.info(f"    -> Replaced with a copy of {last_good_path.name}.")

                        else:
                            logger.warning("    -> No previous valid image in this class to replace with. Image discarded.")

                    action_taken = True
            
            cv2.destroyAllWindows()
            ref_point = []

    hands.close()
    logger.info("Review Pipeline Complete")

if __name__ == '__main__':
    run_review()