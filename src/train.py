# src/train.py
from pathlib import Path

# Important: We are inside 'src', so we can import from 'utils' and 'models' directly
from utils.image_processing import create_data_generators
from models.architectures import create_scratch_model, create_transfer_model # We will create these next

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
    # TODO: Implement this logic
    # scratch_model = create_scratch_model(...)
    # scratch_model.compile(...)
    # history_scratch = scratch_model.fit(...)
    # scratch_model.save(CONFIG["saved_models_dir"] / "scratch_model.keras")
    # plot_history(history_scratch, CONFIG["results_dir"] / "scratch_model_history.png")
    # evaluate_model(scratch_model, test_generator, ...)

    # --- 4. Build and Train Model #2 (Transfer Learning) ---
    print("\n--- Training Model #2: Transfer Learning ---")
    # TODO: Implement this logic
    # transfer_model = create_transfer_model(...)
    # transfer_model.compile(...)
    # history_transfer = transfer_model.fit(...)
    # transfer_model.save(CONFIG["saved_models_dir"] / "transfer_model.keras")
    # plot_history(history_transfer, CONFIG["results_dir"] / "transfer_model_history.png")
    # evaluate_model(transfer_model, test_generator, ...)

if __name__ == "__main__":
    run_training()