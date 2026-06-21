import warnings

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Lambda
from keras.models import Model
from sklearn.model_selection import train_test_split

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split the data into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


# Define Siamese Network architecture
def siamese_network(input_shape):
    input_layer = Input(shape=input_shape)
    flatten = Flatten()(input_layer)
    dense_1 = Dense(128, activation="relu")(flatten)
    output_layer = Dense(32, activation="sigmoid")(dense_1)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# Create Siamese Network branches
input_shape = (28, 28)
siamese_branch = siamese_network(input_shape)

# Generate the encodings for the two inputs
left_input = Input(input_shape)
right_input = Input(input_shape)
encoded_left = siamese_branch(left_input)
encoded_right = siamese_branch(right_input)

# Calculate the Euclidean distance between the encodings
distance = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_left, encoded_right])

# Final classification layer
output_layer = Dense(1, activation="sigmoid")(distance)

# Create the Siamese Network model
siamese_model = Model(inputs=[left_input, right_input], outputs=output_layer)

# Compile and train the Siamese Network model
siamese_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
siamese_model.fit(
    [x_train[:, :, :, np.newaxis], x_train[:, :, :, np.newaxis]],
    y_train,
    epochs=5,
    batch_size=64,
    verbose=2,
)

# Evaluate the Siamese Network on the validation set
loss, accuracy = siamese_model.evaluate(
    [x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]], y_val, verbose=0
)
print(f"Validation Accuracy: {accuracy}")
