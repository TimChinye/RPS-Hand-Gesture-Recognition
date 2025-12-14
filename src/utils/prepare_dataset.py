import random
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "dataset_final"
DEST_DIR = PROJECT_ROOT / "data"
SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
CLASSES = ["rock", "paper", "scissors", "none"]
RANDOM_SEED = 123

def run_split():
    # Splits the final dataset into train/validation/test sets.
    logger.info("Starting Dataset Split")
    
    if not abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-9:
        err_msg = "Split ratios must sum to 1.0"
        logger.error(err_msg)
        raise ValueError(err_msg)

    random.seed(RANDOM_SEED)
    logger.info(f"Using random seed: {RANDOM_SEED} for reproducible splits.")

    if DEST_DIR.exists():
        logger.warning(f"Destination directory '{DEST_DIR}' already exists. Files may be overwritten.")

    logger.info("Creating destination directory structure...")

    for split in SPLIT_RATIOS.keys():
        for class_name in CLASSES:
            path = DEST_DIR / split / class_name
            path.mkdir(parents=True, exist_ok=True)

    total_files_processed = 0
    for class_name in CLASSES:
        source_class_dir = SOURCE_DIR / class_name

        if not source_class_dir.exists():
            logger.warning(f"Source directory not found for class '{class_name}'. Skipping.")
            continue

        image_files = [
            f for f in source_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]

        random.shuffle(image_files)
        num_images = len(image_files)

        logger.info(f"Processing class: '{class_name}' ({num_images} images)")

        # Calculate split points
        train_end = int(SPLIT_RATIOS["train"] * num_images)
        validation_end = train_end + int(SPLIT_RATIOS["validation"] * num_images)
        
        splits = {
            "train": image_files[:train_end],
            "validation": image_files[train_end:validation_end],
            "test": image_files[validation_end:],
        }

        # Copy files to their destination
        for split_name, files in splits.items():
            dest_split_dir = DEST_DIR / split_name / class_name
            logger.info(f"  -> Copying {len(files)} files to '{dest_split_dir.relative_to(PROJECT_ROOT)}'")

            for file_path in files:
                shutil.copy(file_path, dest_split_dir / file_path.name)

            total_files_processed += len(files)

    logger.info("Dataset Split Complete")
    logger.info(f"Total files processed and copied: {total_files_processed}")
    logger.info(f"Data is now available in: {DEST_DIR.resolve()}")


if __name__ == "__main__":
    run_split()