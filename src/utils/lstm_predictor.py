# src/utils/lstm_predictor.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

class MovePredictor:
    """
    An LSTM-based predictor for the next move in a sequence.
    This class implements online learning, training itself as the game progresses.
    """
    def __init__(self, sequence_length=3, num_features=3):
        """
        Initializes the predictor.
        Args:
            sequence_length (int): The number of past moves to consider for a prediction.
            num_features (int): The number of possible moves (rock, paper, scissors).
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.move_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.index_map = {v: k for k, v in self.move_map.items()}
        self.model = self._build_model()
        
        # Store sequences and their corresponding next moves for online learning
        self.sequences = []
        self.next_moves = []

    def _build_model(self):
        """
        Builds and compiles a simple LSTM model.
        - LSTM layer with 16 units: Capable of learning short-term temporal patterns.
        - Dense output layer with softmax: Outputs a probability for each of the 3 moves.
        """
        model = Sequential([
            LSTM(16, input_shape=(self.sequence_length, self.num_features)),
            Dense(self.num_features, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def _moves_to_one_hot(self, moves):
        """
        Converts a list of move strings (e.g., ['rock', 'paper']) into a
        one-hot encoded numpy array, which is the required format for the neural network.
        Example: 'rock' -> [1, 0, 0]
        """
        one_hot = np.zeros((len(moves), self.num_features))
        for i, move in enumerate(moves):
            if move in self.move_map:
                one_hot[i, self.move_map[move]] = 1
        return one_hot

    def train(self, player_history):
        """
        Performs online learning. Takes the player's full history, creates a new
        training sample from the most recent moves, and trains the LSTM for one epoch.
        """
        # We need at least sequence_length + 1 moves to form a single training sample
        # e.g., (move1, move2, move3) -> move4
        if len(player_history) < self.sequence_length + 1:
            return # Not enough data yet

        # Create a new training sample from the tail of the history
        sequence_moves = player_history[-(self.sequence_length + 1) : -1]
        next_move_str = player_history[-1]

        # Convert to one-hot encoding
        sequence_one_hot = self._moves_to_one_hot(sequence_moves)
        next_move_one_hot = self._moves_to_one_hot([next_move_str])

        self.sequences.append(sequence_one_hot)
        self.next_moves.append(next_move_one_hot)

        # Prepare all collected data for training
        # Reshape for LSTM: (number of samples, timesteps, features per timestep)
        X = np.array(self.sequences)
        y = np.array(self.next_moves).reshape(-1, self.num_features)
        
        # Train for a single epoch. This is "online learning" - the model
        # gets slightly better with every valid move the player makes.
        self.model.fit(X, y, epochs=1, verbose=0)
    
    def predict_next_move(self, player_history):
        """
        Predicts the player's next move based on their most recent moves.
        """
        # If we don't have enough history to form a full sequence, play randomly.
        if len(player_history) < self.sequence_length:
            return np.random.choice(['rock', 'paper', 'scissors']) 
        
        # Get the last sequence of moves
        last_sequence_moves = player_history[-self.sequence_length:]
        
        # Convert to one-hot and reshape for prediction: (1, timesteps, features)
        last_sequence_one_hot = self._moves_to_one_hot(last_sequence_moves)
        X_pred = last_sequence_one_hot.reshape(1, self.sequence_length, self.num_features)
        
        # Get the model's prediction
        prediction_probs = self.model.predict(X_pred, verbose=0)[0]
        predicted_index = np.argmax(prediction_probs)
        
        # Convert the predicted index back to a move string
        return self.index_map[predicted_index]