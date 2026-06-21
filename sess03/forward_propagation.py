import matplotlib.pyplot as plt
import numpy as np


# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the forward propagation function
def forward_propagation(inputs, weights, biases):
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(weights, inputs) + biases
    # Apply activation function (sigmoid in this case)
    activation = sigmoid(weighted_sum)
    return activation, weighted_sum


inputs = np.array([0.5, 0.3, 0.2])

weights = np.array([[0.7, 0.2, 0.1], [0.3, 0.8, 0.5]])

biases = np.array([0.1, 0.3])

# Perform forward propagation
output, weighted_sum = forward_propagation(inputs, weights, biases)

# Visualization
plt.figure(figsize=(10, 5))

# Plot the input values
plt.subplot(1, 2, 1)
plt.bar(np.arange(len(inputs)), inputs, color="blue")
plt.title("Input Values")
plt.xlabel("Input Neuron")
plt.ylabel("Value")

# Plot the weighted sum values before activation
plt.subplot(1, 2, 2)
plt.bar(np.arange(len(weighted_sum)), weighted_sum, color="green")
plt.title("Weighted Sum (Before Activation)")
plt.xlabel("Neuron")
plt.ylabel("Weighted Sum")

plt.tight_layout()
plt.show()

print("Output after forward propagation:", output)
