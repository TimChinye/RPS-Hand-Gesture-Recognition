"""
Module for creating and configuring data generators for model training.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from pathlib import Path

def create_data_generators(train_dir, validation_dir, test_dir, image_size, batch_size, color_mode='rgb'):
    """
    Creates and configures training, validation, and test data generators.

    Args:
        train_dir (Path): Path to the training data directory.
        validation_dir (Path): Path to the validation data directory.
        test_dir (Path): Path to the test data directory.
        image_size (tuple): Target size for images, e.g., (150, 150).
        batch_size (int): Number of images per batch.
        color_mode (str): 'rgb' for 3 color channels, 'grayscale' for 1.

    Returns:
        tuple: A tuple containing (train_generator, validation_generator, test_generator).
    """
    print("--- Initializing Data Generators ---")

    # 1. Initialize the Training ImageDataGenerator with data augmentation
    # Data augmentation is a crucial technique to prevent overfitting by creating
    # modified versions of the training data on-the-fly.
    train_datagen = ImageDataGenerator(
        rescale = 1 / 255,
        rotation_range = 25,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    # 2. Initialize the Validation/Test ImageDataGenerator (no augmentation)
    # We only rescale the validation and test data. We must not augment it,
    # as we need a consistent, unbiased evaluation of the model's performance.
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # 3. Create the generators using the flow_from_directory method
    print("Creating Training Generator...")
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical', # For multi-class classification
        shuffle=True,             # Shuffle training data each epoch
        seed=321
    )

    print("Creating Validation Generator...")
    validation_generator = val_test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=False, # No need to shuffle validation data
        seed=321
    )
    
    print("Creating Test Generator...")
    test_generator = val_test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        # CRITICAL: Do NOT shuffle the test set. This ensures that predictions
        # align with file order, which is essential for later evaluation
        # (e.g., creating a confusion matrix).
        shuffle=False, 
        seed=321
    )
    
    print("--- Data Generators Created Successfully ---")
    return train_generator, validation_generator, test_generator