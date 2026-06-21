import matplotlib.pyplot as plt
import numpy as np


def elu(x, alpha=1.0):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)


# Generate values for x
x = np.linspace(-5, 5, 100)

# Compute the ELU function for each value of x
y = elu(x)

# Plot the ELU function
plt.plot(x, y)
plt.title("Exponential Linear Unit (ELU) Activation Function")
plt.xlabel("x")
plt.ylabel("ELU(x)")
plt.grid(True)
plt.show()
