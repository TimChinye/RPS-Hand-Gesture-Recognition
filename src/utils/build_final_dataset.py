# src/utils/build_final_dataset.py
import shutil
from pathlib import Path

# --- CONFIGURATION ---
REVIEW_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_review"
FINAL_DIR = Path(__file__).resolve().parent.parent.parent / "dataset_final"

def run_build():
    """Combines reviewed images into a final dataset directory."""
    print("--- Building Final Dataset ---")
    
    if FINAL_DIR.exists():
        print(f"Warning: Final directory '{FINAL_DIR}' already exists. Deleting it.")
        shutil.rmtree(FINAL_DIR)
        
    FINAL_DIR.mkdir()

    sources_to_combine = [
        REVIEW_DIR / "good_crop",
        REVIEW_DIR / "keep_original"
    ]

    total_copied = 0
    for source_base in sources_to_combine:
        print(f"\nCopying from {source_base.name}...")
        for class_path in source_base.iterdir():
            if not class_path.is_dir(): continue
            
            dest_class_path = FINAL_DIR / class_path.name
            dest_class_path.mkdir(exist_ok=True)
            
            count = 0
            for image_path in class_path.glob("*.png"):
                shutil.copy(str(image_path), dest_class_path)
                count += 1
            
            print(f"  - Copied {count} images for class '{class_path.name}'")
            total_copied += count
            
    print("\n--- Final Dataset Build Complete ---")
    print(f"Total images in final dataset: {total_copied}")
    print(f"Final dataset is ready at: {FINAL_DIR.resolve()}")
    
if __name__ == '__main__':
    run_build()