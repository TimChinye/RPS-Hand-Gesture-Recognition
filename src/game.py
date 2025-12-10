# src/game.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from utils.lstm_predictor import MovePredictor # This will be our Smart AI

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "saved_models/transfer_model.keras" # Using our best model
CLASS_NAMES = ['none', 'paper', 'rock', 'scissors']
IMAGE_SIZE = (150, 150)
CAM_INDEX = 0

# --- HELPER FUNCTIONS ---

def preprocess_frame(frame):
    """Prepares a single frame for model prediction."""
    # Resize the frame to the target size
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    # Convert the BGR frame to RGB, as Keras models are trained on RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Rescale the pixel values to [0, 1]
    rescaled_frame = rgb_frame / 255.0
    # Expand dimensions to create a batch of 1
    return np.expand_dims(rescaled_frame, axis=0)

def get_ai_counter_move(player_predicted_move):
    """Determines the winning move against the player's predicted move."""
    if player_predicted_move == 'rock': return 'paper'
    if player_predicted_move == 'paper': return 'scissors'
    if player_predicted_move == 'scissors': return 'rock'
    # Default case if prediction is unclear
    return np.random.choice(['rock', 'paper', 'scissors'])

def determine_winner(player_move, ai_move):
    """Determines the winner of the current round."""
    if player_move == ai_move:
        return "It's a Tie!", None
    elif (player_move == 'rock' and ai_move == 'scissors') or \
         (player_move == 'scissors' and ai_move == 'paper') or \
         (player_move == 'paper' and ai_move == 'rock'):
        return "You Win!", "player"
    else:
        return "AI Wins!", "ai"

# --- MAIN GAME FUNCTION ---

def run_game():
    """Main function to launch the interactive game."""
    print("--- Starting Rock-Paper-Scissors Game ---")
    
    # 1. Load the gesture recognition model
    print(f"Loading gesture model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Gesture model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        return

    # 2. Initialize webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("FATAL: Could not open webcam.")
        return
    print("Webcam opened. Press 'q' to quit.")
    
    # 3. Initialize Game State & Smart AI
    player_full_history = []  # A full log of the player's valid moves for the AI
    predictor = MovePredictor(sequence_length=3)  # Init the LSTM Smart AI
    
    # Game variables
    player_score = 0
    ai_score = 0
    player_move = "none"
    ai_move = "Thinking..."
    winner_text = "Show your move!"
    predicted_player_move = "N/A"
    
    # Cooldown to make the game playable and prevent rapid-fire moves
    cooldown_frames = 30
    frame_counter = 0

    # --- Main Game Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Flip the frame horizontally for a natural, mirror-like view
        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # --- PREDICTION LOGIC ---
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        confidence = np.max(prediction) * 100
        
        # Only accept a gesture if confidence is high enough
        if confidence > 75:
            predicted_index = np.argmax(prediction)
            player_move = CLASS_NAMES[predicted_index]
        else:
            player_move = "none"

        # --- GAME LOGIC (runs on a cooldown) ---
        if frame_counter > cooldown_frames and player_move in ['rock', 'paper', 'scissors']:
            # 1. Predict what the player will do NEXT based on their history
            predicted_player_move = predictor.predict_next_move(player_full_history)
            
            # 2. The AI chooses its move to COUNTER the player's predicted move
            ai_move = get_ai_counter_move(predicted_player_move)
            
            # 3. Determine the winner of the CURRENT round
            winner_text, round_winner = determine_winner(player_move, ai_move)
            
            # 4. Update score
            if round_winner == "player":
                player_score += 1
            elif round_winner == "ai":
                ai_score += 1
            
            # 5. Add the CURRENT player move to their history log
            player_full_history.append(player_move)

            # 6. TRAIN the Smart AI on the newly updated history (online learning)
            predictor.train(player_full_history)

            # Reset the cooldown timer
            frame_counter = 0

        # --- UI / HUD DRAWING ---
        # Draw a semi-transparent background for the HUD
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (frame.shape[1], 160), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(hud_overlay, alpha, frame, 1 - alpha, 0)
        
        # Display text
        cv2.putText(frame, f"Your Move: {player_move.upper()} ({confidence:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"AI Move: {ai_move.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"SCORE: You {player_score} - {ai_score} AI", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, winner_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"AI Predicts You'll Play: {predicted_player_move.upper()}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow('Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Game Closed ---")

if __name__ == '__main__':
    # You will also need the lstm_predictor.py file in src/utils/
    run_game()