# src/models/architectures.py
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models # type: ignore

def create_scratch_model(input_shape, num_classes):
    """
    Defines and creates a Convolutional Neural Network from scratch.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (150, 150, 3)).
        num_classes (int): The number of output classes.

    Returns:
        A Keras Model instance.
    """
    model = models.Sequential([
        # ---- Feature Extraction Block 1 ----
        # Justification: Start with 32 filters to learn basic features like edges and textures.
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # ---- Feature Extraction Block 2 ----
        # Justification: Double the filters to 64 to learn more complex combinations of the initial features.
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # ---- Feature Extraction Block 3 ----
        # Justification: Increase to 128 filters to capture higher-level patterns specific to hand shapes.
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # ---- Classification Head ----
        # Justification: Flatten the 3D feature maps into a 1D vector to feed into the dense layers.
        layers.Flatten(),
        
        # Justification: A large dense layer to learn combinations of features from the entire image.
        layers.Dense(512, activation='relu'),
        
        # Justification: Dropout is a crucial regularization technique to prevent overfitting
        # by randomly setting a fraction of input units to 0 during training. 0.4 is a moderate rate.
        layers.Dropout(0.4),
        
        # Justification: The final output layer with 'softmax' activation for multi-class probability distribution.
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def create_transfer_model(input_shape, num_classes):
    """
    Defines and creates a model using transfer learning with MobileNetV2.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (150, 150, 3)).
        num_classes (int): The number of output classes.

    Returns:
        A Keras Model instance.
    """
    # 1. Load the pre-trained base model (MobileNetV2) without its top classification layer.
    # Justification: We use the powerful feature extraction layers trained on ImageNet, 
    # but we need to replace the original 1000-class classifier with our own.
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')

    # 2. Freeze the base model.
    # Justification: We don't want to update the learned weights of the base model during
    # the initial training phase. We only want to train our new classifier head.
    base_model.trainable = False

    # 3. Create our new model on top of the base model.
    model = models.Sequential([
        base_model,
        # Justification: GlobalAveragePooling2D reduces the spatial dimensions to a single
        # vector per feature map, which is an efficient way to prepare for the final Dense layer.
        layers.GlobalAveragePooling2D(),
        
        # Justification: A Dropout layer for regularization, even in transfer learning,
        # helps prevent our new classifier head from overfitting on the small dataset.
        layers.Dropout(0.3),
        
        # Justification: The final output layer for our specific 4-class problem.
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model