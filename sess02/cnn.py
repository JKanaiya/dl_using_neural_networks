import warnings

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the digits dataset
digits = load_digits()
X, y = digits.images, digits.target

# Reshape and preprocess the input data for CNN
X = X.reshape((X.shape[0], 8, 8, 1))  # assuming 8x8 images
X = X / 16.0  # normalize pixel values to [0, 1]
y = to_categorical(y)  # one-hot encode labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(8, 8, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))  # 10 classes for digits 0-9
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the CNN model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

# Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")
