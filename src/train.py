import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from utils.image_processing import create_data_generators
from models.architectures import create_scratch_model, create_transfer_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model Hyperparameters
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
COLOR_MODE = "rgb"
EPOCHS = 25

def plot_history(history, save_path):
    # Plots training & validation accuracy and loss.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()

def evaluate_model(model, test_generator, class_names, save_path_prefix):
    # Evaluates the model on the test set and saves the classification report and confusion matrix.
    model_name = Path(save_path_prefix).name
    logger.info(f"Evaluating Model: {model_name}")

    # Get predictions
    y_pred_probs = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_true_indices = test_generator.classes
    y_pred_indices = y_pred_indices[:len(y_true_indices)] # Ensure same length

    # Generate and save classification report
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    report_path = f"{save_path_prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_save_path = f"{save_path_prefix}_confusion_matrix.png"
    plt.savefig(cm_save_path)

    logger.info(f"Confusion matrix saved to {cm_save_path}")
    plt.close()

def run_training():
    # Orchestrates the model training and evaluation pipeline.
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Create data generators
    logger.info("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir=DATA_DIR / "train",
        validation_dir=DATA_DIR / "validation",
        test_dir=DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE
    )
    
    # Model configuration
    input_shape = IMAGE_SIZE + (3,) if COLOR_MODE == 'rgb' else IMAGE_SIZE + (1,)
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())

    # Train Model #1: Scratch CNN
    logger.info("Starting training for Model #1: Scratch CNN")
    scratch_model = create_scratch_model(input_shape=input_shape, num_classes=num_classes)
    scratch_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    scratch_model.summary(print_fn=logger.info)

    history_scratch = scratch_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    model_save_path = SAVED_MODELS_DIR / "scratch_model.keras"
    scratch_model.save(model_save_path)
    logger.info(f"Model #1 saved to {model_save_path}")

    plot_save_path = RESULTS_DIR / "scratch_model_training_curves.png"
    plot_history(history_scratch, plot_save_path)

    evaluate_model(
        model=scratch_model,
        test_generator=test_generator,
        class_names=class_names,
        save_path_prefix=str(RESULTS_DIR / "scratch_model")
    )
    logger.info("Model #1 Training Complete")

    # Train Model #2: Transfer Learning (MobileNetV2)
    logger.info("Starting training for Model #2: Transfer Learning (MobileNetV2)")
    transfer_model = create_transfer_model(input_shape=input_shape, num_classes=num_classes)
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    transfer_model.summary(print_fn=logger.info)

    history_transfer = transfer_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    model_save_path = SAVED_MODELS_DIR / "transfer_model.keras"
    transfer_model.save(model_save_path)
    logger.info(f"Model #2 saved to {model_save_path}")

    plot_save_path = RESULTS_DIR / "transfer_model_training_curves.png"
    plot_history(history_transfer, plot_save_path)

    evaluate_model(
        model=transfer_model,
        test_generator=test_generator,
        class_names=class_names,
        save_path_prefix=str(RESULTS_DIR / "transfer_model")
    )
    logger.info("Model #2 Training & Evaluation Complete")

if __name__ == "__main__":
    run_training()