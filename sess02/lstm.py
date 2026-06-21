import warnings

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Ignore warnings
warnings.filterwarnings("ignore")

# Generate a simple dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Reshape the input data for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a simple LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the LSTM model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

# Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")
