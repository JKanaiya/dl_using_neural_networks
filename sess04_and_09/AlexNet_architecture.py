from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


# Define the AlexNet model
def alexnet_model(input_shape, num_classes):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )

    # 4th Convolutional Layer
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )

    # 5th Convolutional Layer
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten the layers
    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation="softmax"))

    return model


# Define input shape and number of classes
input_shape = (224, 224, 3)  # Input shape for AlexNet
num_classes = 1000  # Number of output classes

# Create the AlexNet model
model = alexnet_model(input_shape, num_classes)

# Print the model summary
model.summary()
