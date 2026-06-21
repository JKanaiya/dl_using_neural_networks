import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


# Generate values for x
x = np.linspace(-10, 10, 100)

# Compute the ReLU function for each value of x
y = relu(x)
# Plot the ReLU function
plt.plot(x, y)
plt.title("Rectified Linear Unit (ReLU) Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.show()


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)


def swish(x):
    return x * (1 / (1 + np.exp(-x)))


def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


# Generate values for x
x = np.linspace(-10, 10, 100)

# Compute the activation functions for each value of x
y_leaky_relu = leaky_relu(x)
y_swish = swish(x)
y_gelu = gelu(x)

# Plot the activation functions
plt.plot(x, y_leaky_relu, label="Leaky ReLU")
plt.plot(x, y_swish, label="Swish")
plt.plot(x, y_gelu, label="GELU")

plt.title("ReLu Variants")
plt.xlabel("x")
plt.ylabel("Activation(x)")
plt.grid(True)
plt.legend()
plt.show()
