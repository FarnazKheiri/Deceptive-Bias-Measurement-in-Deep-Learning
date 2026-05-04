import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Lambda, Input
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, DenseNet121


# ==============================================================================
# 1. PREPROCESSING WRAPPERS
# ==============================================================================
def preprocess_input_densenet(x):
    """Applies DenseNet-specific preprocessing."""
    return tf.keras.applications.densenet.preprocess_input(x)


def preprocess_input_resnet(x):
    """Scales pixels to [-1, 1] as required by ResNetV2."""
    return tf.keras.applications.resnet_v2.preprocess_input(x)


# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================

def get_efficientnet_model():
    """EfficientNetB0: Freezes all but the last 6 layers."""
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers[:-6]:
        layer.trainable = False

    # EfficientNet usually includes internal rescaling, but custom heads are added similarly
    return build_sequential_model(base_model, preprocessing_type="rescaling")


def get_resnet_model():
    """ResNet50V2: Freezes all but the last 15 layers."""
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    return build_sequential_model(base_model, preprocessing_type="resnet")


def get_densenet_model():
    """DenseNet121: Freezes all but the last 10 layers."""
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    return build_sequential_model(base_model, preprocessing_type="densenet")


# ==============================================================================
# 3. ARCHITECTURE HELPER
# ==============================================================================
def build_sequential_model(base_model, preprocessing_type):
    """Constructs the final Sequential model with the custom classifier head."""
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    # Apply correct preprocessing logic via Lambda layers
    if preprocessing_type == "densenet":
        model.add(Lambda(preprocess_input_densenet))
    elif preprocessing_type == "resnet":
        model.add(Lambda(preprocess_input_resnet))

    model.add(base_model)

    # Custom Head (Identical across all experiments for fairness)
    model.add(MaxPooling2D((7, 7)))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation="relu"))
    model.add(Dense(2, activation="softmax"))

    return model
