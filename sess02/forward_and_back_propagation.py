# Forward Propagation

import numpy as np


def forward_propagation(inputs, weights, biases):
    # Calculate weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + biases

    # Apply sigmoid activation function
    output = 1 / (1 + np.exp(-weighted_sum))

    return output


# Example inputs
inputs = np.array([0.5, 0.3, 0.2])
weights = np.array([0.4, 0.7, 0.2])
biases = 0.1

# Perform forward propagation
output = forward_propagation(inputs, weights, biases)
print("Forward Propagation Output:", output)

# Backpropagation


def backpropagation(inputs, weights, biases, targets, learning_rate):
    # Forward pass
    weighted_sum = np.dot(inputs, weights) + biases
    output = 1 / (1 + np.exp(-weighted_sum))

    # Compute error
    error = targets - output

    # Compute gradient
    gradient = error * output * (1 - output)

    # Update weights and biases
    weights += learning_rate * np.dot(inputs.T, gradient)
    biases += learning_rate * np.sum(gradient)

    return error


# Example inputs and targets
inputs = np.array([[0.5, 0.3, 0.2]])
weights = np.array([[0.4], [0.7], [0.2]])
biases = 0.1
targets = np.array([0.8])

# Perform backpropagation
learning_rate = 0.1
error = backpropagation(inputs, weights, biases, targets, learning_rate)
print("Backpropagation Error:", error)
