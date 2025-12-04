# run.py
import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for the Rock-Paper-Scissors project pipeline."""
    parser = argparse.ArgumentParser(
        description="A complete pipeline for the RPS Hand Gesture Recognition project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "action", 
        nargs='?',
        choices=['collect', 'prepare', 'train', 'play'],
        help=(
            "The main action to perform:"
            "\n  collect   - Start the interactive webcam data collection."
            "\n  prepare   - Split the raw dataset into train/validation/test sets."
            "\n  train     - Train the CNN models on the prepared data."
            "\n  play      - Run the interactive game with a trained mode."
        )
    )
    
    # --- Check if any arguments were provided ---
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Add 'src' to the Python path ---
    # This is done *after* parsing args so we don't do it unnecessarily.
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT / 'src'))

    # --- Execute the chosen action with LAZY IMPORTS ---
    if args.action == 'collect':
        print("--- Loading Data Collection Module ---")
        from data_collection import run_collection # type: ignore
        run_collection()
        
    elif args.action == 'prepare':
        print("--- Loading Dataset Preparation Module ---")
        from utils.prepare_dataset import run_split # type: ignore
        run_split()
        
    elif args.action == 'train':
        print("--- Loading Training Module (this may take a moment)... ---")
        from train import run_training # type: ignore
        run_training()
        
    elif args.action == 'play':
        print("--- Loading Game Module (this may take a moment)... ---")
        from game import run_game # type: ignore
        run_game()

if __name__ == "__main__":
    main()