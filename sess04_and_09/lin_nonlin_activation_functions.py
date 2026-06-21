import matplotlib.pyplot as plt
import numpy as np


# Linear activation function
def linear(x):
    return x


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Tanh activation function
def tanh(x):
    return np.tanh(x)


# Create input values
x = np.linspace(-5, 5, 1000)

# Calculate output values for each activation function
y_linear = linear(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# Plot the activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, y_linear, label="Linear", color="blue")
plt.plot(x, y_sigmoid, label="Sigmoid", color="green")
plt.plot(x, y_relu, label="ReLU", color="red")
plt.plot(x, y_tanh, label="Tanh", color="orange")

plt.title("Linear & Non-Linear Activation Functions", fontsize=20)
plt.xlabel("Input", fontsize=14)
plt.ylabel("Output", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
