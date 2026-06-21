import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate values for x
x = np.linspace(-10, 10, 100)

# Compute the sigmoid function for each value of x
y = sigmoid(x)

# Plot the sigmoid function
plt.plot(x, y)
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid(True)
plt.show()
