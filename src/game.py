import cv2
import numpy as np
import tensorflow as tf
import time
import logging
from pathlib import Path
from utils.lstm_predictor import MovePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILENAME = "transfer_model.keras"
ASSETS_DIRNAME = "assets"
MODEL_PATH = PROJECT_ROOT / "saved_models" / MODEL_FILENAME
ASSETS_PATH = Path(__file__).resolve().parent / ASSETS_DIRNAME

# Model & Image Settings
CLASS_NAMES = ['none', 'paper', 'rock', 'scissors']
IMAGE_SIZE = (150, 150)
CONFIDENCE_THRESHOLD = 0.5

# Calibration Settings
CALIBRATION_DURATION = 10
CALIBRATION_THRESHOLD = 0.75

# Gameplay Settings
CAM_INDEX = 0
GESTURE_HOLD_DURATION = 1
ROUND_COUNTDOWN_DURATION = 3
RESULTS_DISPLAY_DURATION = 3

# UI Settings
WINDOW_SCALE_FACTOR = 1.5
ASSET_DISPLAY_SIZE = (100, 100)

def preprocess_frame(frame):
    # Prepares a single frame for model prediction.
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    rescaled_frame = rgb_frame / 255.0

    return np.expand_dims(rescaled_frame, axis=0)

def get_ai_counter_move(player_predicted_move):
    # Determines the winning move against the player's predicted move.
    moves = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

    return moves.get(player_predicted_move, np.random.choice(list(moves.keys())))

def determine_winner(player_move, ai_move):
    # Determines the winner of the current round.

    if player_move == ai_move:
        return "It's a Tie!", "tie"
    
    if (player_move, ai_move) in [('rock', 'scissors'), ('scissors', 'paper'), ('paper', 'rock')]:
        return "You Win!", "player"
    
    return "AI Wins!", "ai"

def load_assets():
    # Loads and resizes all required UI images from the assets folder.
    logger.info("Loading UI Assets...")

    assets = {}
    required_files = ['rock.png', 'paper.png', 'scissors.png', 'thinking.png', 'happy.png', 'neutral.png', 'sad.png']

    for filename in required_files:
        path = ASSETS_PATH / filename

        if not path.exists():
            logger.error(f"Asset not found at {path}. Please check your src/assets folder.")
            return None
        
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assets[filename.split('.')[0]] = cv2.resize(img, ASSET_DISPLAY_SIZE)

    logger.info("Assets loaded successfully.")
    return assets

def overlay_image(background, overlay, x, y):
    # Overlays a transparent PNG onto a background frame.
    h, w, _ = overlay.shape

    alpha = overlay[:, :, 3] / 255.0
    y_end, x_end = y + h, x + w

    if y_end > background.shape[0] or x_end > background.shape[1]:
        return

    for c in range(0, 3):
        background[y:y_end, x:x_end, c] = (alpha * overlay[:, :, c]) + (background[y:y_end, x:x_end, c] * (1.0 - alpha))

def create_feature_extractor(model):
    # Creates a new model that extracts feature vectors from an intermediate layer.
    feature_layer = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))

    extractor = tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)
    logger.info("Feature extractor created from model.")

    return extractor

def cosine_similarity(v, V_cloud):
    # Calculates the max cosine similarity between a vector v and a cloud of vectors V.
    v_norm = v / np.linalg.norm(v)
    V_cloud_norm = V_cloud / np.linalg.norm(V_cloud, axis=1, keepdims=True)
    similarities = np.dot(V_cloud_norm, v_norm.T)

    return np.max(similarities)

def check_calibration_match(frame, feature_extractor, signatures):
    # Checks if the current frame's features match any calibrated signature cloud.
    if not signatures:
        return None
    
    processed_frame = preprocess_frame(frame)
    current_vector = feature_extractor.predict(processed_frame, verbose=0)
    
    sim_empty = cosine_similarity(current_vector, signatures["empty"])
    if sim_empty > CALIBRATION_THRESHOLD:
        return f"Empty Scene ({sim_empty:.2f})"
        
    sim_operator = cosine_similarity(current_vector, signatures["operator"])
    if sim_operator > CALIBRATION_THRESHOLD:
        return f"Operator ({sim_operator:.2f})"
        
    return None

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # Generates a Grad-CAM heatmap for a given image and model.
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = (last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def display_gradcam(frame, heatmap, alpha=0.6):
    # Applies a heatmap to a frame for visualization.
    heatmap = np.uint8(255 * heatmap)

    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (frame.shape[1], frame.shape[0]))

    superimposed_img = jet * alpha + frame * (1 - alpha)

    return np.uint8(superimposed_img)

def run_game():
    # Main function to launch the enhanced interactive game.
    logger.info("Starting Rock-Paper-Scissors Game")
    
    # Initialize components
    model = tf.keras.models.load_model(MODEL_PATH)
    feature_extractor = create_feature_extractor(model)
    assets = load_assets()
    cap = cv2.VideoCapture(CAM_INDEX)

    if not all([model, assets, cap.isOpened(), feature_extractor]):
        logger.error("Failed to initialize one or more components (model, assets, camera). Exiting.")
        return

    # Window setup
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    win_width = int(cam_width * WINDOW_SCALE_FACTOR)
    win_height = int(cam_height * WINDOW_SCALE_FACTOR)

    cv2.namedWindow('Rock Paper Scissors', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Rock Paper Scissors', win_width, win_height)

    # Game state variables
    game_state = 'calibrating_empty'
    analysis_mode = False
    signatures, feature_vectors_buffer = {}, []
    calibration_start_time = time.time()
    player_score, ai_score = 0, 0
    player_full_history = []
    predictor = MovePredictor()
    live_gesture, confidence = 'none', 0.0
    locked_player_move = None
    ai_move, winner_text, last_round_prediction_text = 'thinking', "Show your move!", ""
    gesture_hold_start_time, countdown_start_time, results_start_time = None, 0, 0
    
    # Print console UI
    print("\n--- Game Live ---")
    print("Controls: [1,2,3] Demo | [A] Analysis | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            analysis_mode = not analysis_mode
            game_state = 'analysis' if analysis_mode else 'playing'

            if analysis_mode:
                frozen_frame = frame.copy()

        display_frame = cv2.resize(frame, (win_width, win_height))

        # Main game state machine
        if game_state.startswith('calibrating'):
            phase = game_state.split('_')[1]
            prompt = "Please step out of view" if phase == "empty" else "Please sit normally"
            elapsed_time = time.time() - calibration_start_time
            remaining_time = CALIBRATION_DURATION - elapsed_time

            cv2.putText(display_frame, f"CALIBRATION: {phase.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.putText(display_frame, prompt, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(display_frame, str(int(remaining_time) + 1), (win_width//2 - 60, win_height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 15)
            
            feature_vectors_buffer.append(feature_extractor.predict(preprocess_frame(frame), verbose=0))

            if remaining_time <= 0:
                signatures[phase] = np.vstack(feature_vectors_buffer)
                logger.info(f"'{phase.upper()}' signature cloud created with {len(signatures[phase])} vectors.")

                if phase == "empty":
                    game_state, calibration_start_time, feature_vectors_buffer = 'calibrating_operator', time.time(), []

                else:
                    game_state = 'playing'

        elif game_state == 'playing':
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame, verbose=0)
            live_gesture = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            calibration_match = check_calibration_match(frame, feature_extractor, signatures)
            if calibration_match:
                live_gesture = 'none'

            if key in [ord('1'), ord('2'), ord('3')]:
                locked_player_move = {'1': 'rock', '2': 'paper', '3': 'scissors'}[chr(key)]
                game_state, countdown_start_time = 'countdown', time.time()

                continue

            if live_gesture in ['rock', 'paper', 'scissors'] and confidence > (CONFIDENCE_THRESHOLD * 100):
                if gesture_hold_start_time is None:
                    gesture_hold_start_time = time.time()

                elif time.time() - gesture_hold_start_time > GESTURE_HOLD_DURATION:
                    locked_player_move = live_gesture
                    game_state, countdown_start_time = 'countdown', time.time()
            else:
                gesture_hold_start_time = None

        elif game_state == 'countdown':
            if time.time() - countdown_start_time > ROUND_COUNTDOWN_DURATION:
                predicted_next_move = predictor.predict_next_move(player_full_history)
                ai_move = get_ai_counter_move(predicted_next_move)
                winner_text, round_winner = determine_winner(locked_player_move, ai_move)

                if round_winner == "player":
                    player_score += 1
                elif round_winner == "ai":
                    ai_score += 1

                player_full_history.append(locked_player_move)
                
                predictor.train(player_full_history)
                last_round_prediction_text = f"AI Previously Predicted You'd Pick: {predicted_next_move.upper()}"

                logger.info(f"Round: Player's move: {locked_player_move}, AI's move: {ai_move}, Winner: {round_winner}. Current Score (Player vs AI): {player_score}-{ai_score}")
                game_state, results_start_time = 'show_results', time.time()

        elif game_state == 'show_results':
            if time.time() - results_start_time > RESULTS_DISPLAY_DURATION:
                game_state, winner_text, ai_move = 'playing', "Show your move!", 'thinking'

        # UI Drawing
        if 'calibrating' not in game_state:
            hud_overlay = display_frame.copy()
            cv2.rectangle(hud_overlay, (0, 0), (win_width, 190), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(hud_overlay, 0.6, display_frame, 0.4, 0)
            
            if game_state != 'analysis':
                ai_status_image = assets['thinking']
                
                if game_state == 'show_results':
                    ai_status_image = assets[ai_move]
                    emotion_map = {"player": "sad", "ai": "happy", "tie": "neutral"}
                    overlay_image(display_frame, assets[emotion_map[round_winner]], int(win_width*0.6), int(win_height*0.5))
                
                overlay_image(display_frame, ai_status_image, int(win_width * 0.8), 20)
                
                # HUD Text
                cv2.putText(display_frame, f"Live Detection: {live_gesture.upper()} ({confidence:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"SCORE: You {player_score} - {ai_score} AI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, last_round_prediction_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if 'calibration_match' in locals() and calibration_match:
                    cv2.putText(display_frame, f"CALIBRATION: {calibration_match}", (20, win_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # State-specific overlays
                if game_state == 'countdown':
                    remaining = int(ROUND_COUNTDOWN_DURATION - (time.time() - countdown_start_time)) + 1
                    cv2.putText(display_frame, str(remaining), (win_width//2 - 50, win_height//2), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 10)
                    cv2.putText(display_frame, f"You chose {locked_player_move.upper()}!", (win_width//2 - 250, win_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                elif game_state == 'show_results':
                    result_summary = f"AI picked {ai_move.upper()} and {winner_text}"
                    cv2.putText(display_frame, result_summary, (win_width//2 - 400, win_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            else: # Analysis mode display
                last_conv_layer_name = next(layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D))
                processed_frozen = preprocess_frame(frozen_frame)
                actual_pred_index = np.argmax(model.predict(processed_frozen, verbose=0))
                heatmap_actual = generate_gradcam_heatmap(model, processed_frozen, last_conv_layer_name, actual_pred_index)
                display_actual = display_gradcam(cv2.resize(frozen_frame, (win_width//2, win_height-60)), heatmap_actual)

                scissors_index = CLASS_NAMES.index('scissors')
                heatmap_scissors = generate_gradcam_heatmap(model, processed_frozen, last_conv_layer_name, scissors_index)
                display_scissors = display_gradcam(cv2.resize(frozen_frame, (win_width//2, win_height-60)), heatmap_scissors)

                combined_display = np.vstack([np.zeros((60, win_width, 3), dtype=np.uint8), np.hstack((display_actual, display_scissors))])
                cv2.putText(combined_display, f"Why it chose: {CLASS_NAMES[actual_pred_index].upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_display, "Why it's NOT scissors", (win_width//2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                display_frame = combined_display

        cv2.imshow('Rock Paper Scissors', display_frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Game Closed")

if __name__ == '__main__':
    run_game()