import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

logger = logging.getLogger(__name__)

def create_data_generators(train_dir, validation_dir, test_dir, image_size, batch_size, color_mode='rgb'):
    # Creates and configures training, validation, and test data generators.
    logger.info("Initializing Data Generators")

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

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create the generators using the flow_from_directory method
    logger.info("Creating Training Generator...")
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=True,
        seed=321
    )

    logger.info("Creating Validation Generator...")
    validation_generator = val_test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=False,
        seed=321
    )
    
    logger.info("Creating Test Generator...")
    test_generator = val_test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        shuffle=False, 
        seed=321
    )
    
    logger.info("Data Generators Created Successfully")
    return train_generator, validation_generator, test_generator