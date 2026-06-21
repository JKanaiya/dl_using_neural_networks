import matplotlib.pyplot as plt
import numpy as np


def tanh(x):
    return np.tanh(x)


# Generate values for x
x = np.linspace(-10, 10, 100)

# Compute the tanh function for each value of x
y = tanh(x)

# Plot the tanh function
plt.plot(x, y)
plt.title("Hyperbolic Tangent (tanh) Activation Function")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.grid(True)
plt.show()
