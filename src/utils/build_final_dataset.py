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
REVIEW_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_review"
FINAL_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_final"

def run_build():
    # Combines reviewed images into a final dataset directory.
    logger.info("Building Final Dataset")
    
    if FINAL_DIR.exists():
        logger.warning(f"Final directory '{FINAL_DIR}' already exists. Deleting it.")
        shutil.rmtree(FINAL_DIR)
        
    FINAL_DIR.mkdir()

    sources_to_combine = [
        REVIEW_DIR / "good_crop",
        REVIEW_DIR / "keep_original"
    ]

    total_copied = 0
    for source_base in sources_to_combine:
        logger.info(f"Copying from {source_base.name}...")
        
        class_paths = [p for p in source_base.iterdir() if p.is_dir()]
        for class_path in class_paths:
            dest_class_path = FINAL_DIR / class_path.name
            dest_class_path.mkdir(exist_ok=True)
            
            count = 0
            for image_path in class_path.glob("*.png"):
                shutil.copy(str(image_path), dest_class_path)
                count += 1
            
            logger.info(f"  - Copied {count} images for class '{class_path.name}'")
            total_copied += count
            
    logger.info("Final Dataset Build Complete")
    logger.info(f"Total images in final dataset: {total_copied}")
    logger.info(f"Final dataset is ready at: {FINAL_DIR.resolve()}")
    
if __name__ == '__main__':
    run_build()