import argparse
import sys
import logging
import importlib
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Used to provide a cleaner approach than if/else spam and a large `help_msg` string.
COMMAND_MAP = { # arg: (module_path, function_name, desc_str)
    'collect': ('data_collection',           'run_collection', 'Start the interactive webcam data collection app.'),
    'crop':    ('utils.auto_crop',           'run_auto_crop',  'Auto-crop hands from the raw dataset.'),
    'review':  ('utils.review',              'run_review',     'Manually-review the collected & cropped images.'),
    'build':   ('utils.build_final_dataset', 'run_build',      'Build the final dataset structure.'),
    'prepare': ('utils.prepare_dataset',     'run_split',      'Split the raw dataset into train/validation/test sets.'),
    'train':   ('train',                     'run_training',   'Train the CNN models on the prepared data.'),
    'play':    ('game',                      'run_game',       'Run the interactive game with a trained model, using LTSM for prediction.' )
}

def execute_action(action_name):
    # Import the selected module and run the main function

    if action_name not in COMMAND_MAP:
        logger.error(f"Action '{action_name}' is not implemented in the command map.")
        return

    module_name, function_name, _ = COMMAND_MAP[action_name]

    logger.info(f"Initializing action: {action_name}")
    logger.info(f"Loading module: {module_name}...")

    try:
        module = importlib.import_module(module_name) # Equal to: from module_name import function_name
        getattr(module, function_name)() # Equal to: function_name()

    except ImportError as err:
        logger.error(f"Failed to import module '{module_name}'. Ensure paths are correct.")
        logger.error(f"Details: {err}")
        sys.exit(1)

    except AttributeError:
        logger.error(f"Module '{module_name}' exists, but function '{function_name}' was not found.")
        sys.exit(1)

    except Exception as err:
        logger.exception(f"An unexpected error occurred during execution: {err}")
        sys.exit(1)

def main():
    # Main entry point
    parser = argparse.ArgumentParser(
        description="Main entry point for the RPS Hand Gesture Recognition project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "action", 
        nargs='?',
        choices=list(COMMAND_MAP.keys()),
        help=(
            "\nThe main action to perform:\n" + 
            "\n".join([f"  {cmd:<7} - {details[2]}" for (cmd, details) in COMMAND_MAP.items()])
        )
    )

    # Check for args
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Add 'src' to the Python path so imports work correctly
    project_root = Path(__file__).resolve().parent
    src_path = project_root / 'src'
    sys.path.append(str(src_path))
    
    # Run the selected action
    args = parser.parse_args()
    execute_action(args.action)

if __name__ == "__main__":
    main()