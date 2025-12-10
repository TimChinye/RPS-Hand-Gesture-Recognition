import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf # type: ignore

from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from utils.image_processing import create_data_generators
from models.architectures import create_scratch_model, create_transfer_model

def plot_history(history, save_path):
    """Plots training & validation accuracy and loss."""
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
    print(f"Training history plot saved to {save_path}")
    plt.close()

def evaluate_model(model, test_generator, class_names, save_path_prefix):
    """
    Evaluates the model on the test set and saves the classification report
    and confusion matrix.
    """
    print(f"\n--- Evaluating Model: {save_path_prefix} ---")

    # 1. Get Predictions
    # Use predict() on the test generator to get model's output probabilities
    y_pred_probs = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
    # Use argmax to get the index of the highest probability class
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # 2. Get True Labels
    # The test generator's classes attribute holds the true label index for each image
    y_true_indices = test_generator.classes

    # Ensure we only evaluate on the number of samples in the generator
    y_pred_indices = y_pred_indices[:len(y_true_indices)]

    # 3. Generate and Save Classification Report
    print("Classification Report:")
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    print(report)
    with open(f"{save_path_prefix}_classification_report.txt", "w") as f:
        f.write(report)
    print(f"Classification report saved to {save_path_prefix}_classification_report.txt")

    # 4. Generate and Save Confusion Matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_save_path = f"{save_path_prefix}_confusion_matrix.png"
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to {cm_save_path}")
    plt.close()

# --- ROBUST PATHING & CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "data_dir": PROJECT_ROOT / "data",
    "image_size": (150, 150),
    "batch_size": 32,
    "color_mode": "rgb",
    "epochs": 25,
    "saved_models_dir": PROJECT_ROOT / "saved_models",
    "results_dir": PROJECT_ROOT / "results"
}

def run_training():
    """
    Orchestrates the model training pipeline.
    This function is imported and called by the main run.py script.
    """
    # Ensure output directories exist
    CONFIG["saved_models_dir"].mkdir(exist_ok=True)
    CONFIG["results_dir"].mkdir(exist_ok=True)
    
    # --- 1. Define Data Directories ---
    base_path = CONFIG["data_dir"]
    train_dir = base_path / "train"
    validation_dir = base_path / "validation"
    test_dir = base_path / "test"

    # --- 2. Create Data Generators ---
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir=train_dir,
        validation_dir=validation_dir,
        test_dir=test_dir,
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        color_mode=CONFIG["color_mode"]
    )
    
    # --- 3. Build and Train Model #1 (Scratch CNN) ---
    print("\n--- Training Model #1: Scratch CNN ---")

    # Define model input shape and number of classes
    input_shape = CONFIG["image_size"] + (3,) if CONFIG["color_mode"] == 'rgb' else CONFIG["image_size"] + (1,)
    num_classes = len(train_generator.class_indices)

    # Create the model
    scratch_model = create_scratch_model(input_shape=input_shape, num_classes=num_classes)

    # Compile the model
    scratch_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    scratch_model.summary() # Print a summary of the model architecture

    # Train the model
    history_scratch = scratch_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // CONFIG["batch_size"]
    )

    # Save the trained model and its history plot
    model_save_path = CONFIG["saved_models_dir"] / "scratch_model.keras"
    scratch_model.save(model_save_path)
    print(f"Model #1 saved to {model_save_path}")

    plot_save_path = CONFIG["results_dir"] / "scratch_model_training_curves.png"
    plot_history(history_scratch, plot_save_path)

    # Get class names for evaluation reports
    class_names = list(train_generator.class_indices.keys())

    # Evaluate the model
    evaluate_model(
        model=scratch_model,
        test_generator=test_generator,
        class_names=class_names,
        save_path_prefix=str(CONFIG["results_dir"] / "scratch_model")
    )

    print("--- Model #1 Training Complete ---")

    # --- 4. Build and Train Model #2 (Transfer Learning) ---
    print("\n--- Training Model #2: Transfer Learning (MobileNetV2) ---")

    # Create the model
    transfer_model = create_transfer_model(input_shape=input_shape, num_classes=num_classes)

    # Compile the model - use a lower learning rate for transfer learning as a best practice
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Use TensorFlow import
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    transfer_model.summary()

    # Train the model
    history_transfer = transfer_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // CONFIG["batch_size"]
    )

    # Save the model and plot its history
    model_save_path_t = CONFIG["saved_models_dir"] / "transfer_model.keras"
    transfer_model.save(model_save_path_t)
    print(f"Model #2 saved to {model_save_path_t}")

    plot_save_path_t = CONFIG["results_dir"] / "transfer_model_training_curves.png"
    plot_history(history_transfer, plot_save_path_t)

    # Evaluate the model
    evaluate_model(
        model=transfer_model,
        test_generator=test_generator,
        class_names=class_names,
        save_path_prefix=str(CONFIG["results_dir"] / "transfer_model")
    )

    print("--- Model #2 Training & Evaluation Complete ---")

if __name__ == "__main__":
    run_training()