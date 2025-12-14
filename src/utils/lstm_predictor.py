import numpy as np
import logging
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CONFIG = {
    "sequence_length": 3,
    "num_features": 3,
    "batch_size": 16,
    "training_epochs": 10,
    "lstm_units": 16,
    "optimizer": "adam",
    "loss_function": "categorical_crossentropy"
}

# An LSTM-based predictor for the next move in a sequence
class MovePredictor:
    def __init__(self, sequence_length=CONFIG["sequence_length"], num_features=CONFIG["num_features"], batch_size=CONFIG["batch_size"]):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.batch_size = batch_size
        self.move_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.index_map = {v: k for k, v in self.move_map.items()}
        self.model = self._build_model()
        self.memory = []

    def _build_model(self):
        # Builds and compiles a simple LSTM model using values from CONFIG.
        model = Sequential([
            Input(shape=(self.sequence_length, self.num_features)),
            LSTM(CONFIG["lstm_units"]),
            Dense(self.num_features, activation='softmax')
        ])

        model.compile(optimizer=CONFIG["optimizer"], loss=CONFIG["loss_function"])
        logger.info("LSTM prediction model built.")
        return model

    def _moves_to_one_hot(self, moves):
        # Converts a list of move strings into a one-hot encoded numpy array.
        one_hot = np.zeros((len(moves), self.num_features))
        for i, move in enumerate(moves):
            if move in self.move_map:
                one_hot[i, self.move_map[move]] = 1
        return one_hot

    def train(self, player_history):
        # Learn quicker by focusing on the most recent move.
        if len(player_history) < self.sequence_length + 1:
            return

        # Prepare the latest experience
        sequence_moves = player_history[-self.sequence_length - 1 : -1]
        next_move_str = player_history[-1]
        sequence_one_hot = self._moves_to_one_hot(sequence_moves)
        next_move_one_hot = self._moves_to_one_hot([next_move_str])[0]
        
        # Reshape the single data point to be a valid batch of 1
        X_train = np.expand_dims(sequence_one_hot, axis=0)
        y_train = np.expand_dims(next_move_one_hot, axis=0)
        
        self.model.fit(X_train, y_train, epochs=CONFIG["training_epochs"], verbose=0)
    
    def predict_next_move(self, player_history):
        # Predicts the player's next move based on their most recent moves.
        if len(player_history) < self.sequence_length:
            return np.random.choice(list(self.move_map.keys()))
        
        last_sequence_moves = player_history[-self.sequence_length:]
        last_sequence_one_hot = self._moves_to_one_hot(last_sequence_moves)
        X_pred = last_sequence_one_hot.reshape(1, self.sequence_length, self.num_features)
        
        prediction_probs = self.model.predict(X_pred, verbose=0)[0]
        predicted_index = np.argmax(prediction_probs)
        
        return self.index_map[predicted_index]