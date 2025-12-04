# src/utils/prepare_dataset.py
import random
import shutil
from pathlib import Path
import warnings

# --- ROBUST PATHING & CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = {
    "source_dir": PROJECT_ROOT / "dataset",
    "dest_dir": PROJECT_ROOT / "data",
    # ... (rest of config is the same)
    "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
    "classes": ["rock", "paper", "scissors", "none"],
    "random_seed": 123
}

def run_split(): # <--- Renamed from split_dataset for clarity
    """
    Splits the raw dataset from /dataset into train/validation/test sets in /data.
    This function is imported and called by the main run.py script.
    """
    # --- (The rest of the function code is IDENTICAL to your original file) ---
    print("--- Starting Dataset Split ---")
    if not abs(sum(CONFIG["split_ratios"].values()) - 1.0) < 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    random.seed(CONFIG["random_seed"])
    print(f"Using random seed: {CONFIG['random_seed']} for reproducible splits.")

    if CONFIG["dest_dir"].exists():
        warnings.warn(f"Destination directory '{CONFIG['dest_dir']}' already exists. Files may be overwritten.")
    
    print("Creating destination directory structure...")
    for split in CONFIG["split_ratios"].keys():
        for class_name in CONFIG["classes"]:
            path = CONFIG["dest_dir"] / split / class_name
            path.mkdir(parents=True, exist_ok=True)
            
    total_files_processed = 0
    for class_name in CONFIG["classes"]:
        source_class_dir = CONFIG["source_dir"] / class_name
        if not source_class_dir.exists():
            print(f"Warning: Source directory not found for class '{class_name}'. Skipping.")
            continue

        image_files = [f for f in source_class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        random.shuffle(image_files)
        num_images = len(image_files)
        print(f"\nProcessing class: '{class_name}' ({num_images} images)")

        train_end = int(CONFIG["split_ratios"]["train"] * num_images)
        validation_end = train_end + int(CONFIG["split_ratios"]["validation"] * num_images)
        splits = {"train": image_files[:train_end], "validation": image_files[train_end:validation_end], "test": image_files[validation_end:]}
        
        for split_name, files in splits.items():
            dest_split_dir = CONFIG["dest_dir"] / split_name / class_name
            print(f"  -> Copying {len(files)} files to '{dest_split_dir}'")
            for file_path in files:
                shutil.copy(file_path, dest_split_dir / file_path.name)
            total_files_processed += len(files)
    
    print("\n--- Dataset Split Complete ---")
    print(f"Total files processed and copied: {total_files_processed}")
    print(f"Data is now available in: {CONFIG['dest_dir'].resolve()}")


if __name__ == "__main__":
    run_split()