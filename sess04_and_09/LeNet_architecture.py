import tensorflow as tf
from tensorflow.keras import layers, models


# Define LeNet architecture
def LeNet(input_shape, num_classes):
    model = models.Sequential()

    # Layer 1: Convolutional layer with 6 filters, each 5x5
    model.add(
        layers.Conv2D(6, kernel_size=(5, 5), activation="relu", input_shape=input_shape)
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 2: Convolutional layer with 16 filters, each 5x5
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Layer 3: Fully connected layer with 120 units
    model.add(layers.Dense(120, activation="relu"))

    # Layer 4: Fully connected layer with 84 units
    model.add(layers.Dense(84, activation="relu"))

    # Output layer with softmax activation for classification
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


# Define input shape and number of classes
input_shape = (32, 32, 3)
num_classes = 10

# Create the LeNet model
model = LeNet(input_shape, num_classes)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Display the model summary
model.summary()
