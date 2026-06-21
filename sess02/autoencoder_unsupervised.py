# Importing necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Normalize data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test = train_test_split(X_normalized, test_size=0.2, random_state=42)

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 2

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoder)

# Create and compile the autoencoder model
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder
autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
)

# Obtain encoded representation of input data
encoded_data = autoencoder.predict(X_normalized)
