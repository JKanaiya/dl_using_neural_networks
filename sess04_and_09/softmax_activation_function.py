import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=0)


# Generate values for x (logits for each class)
logits = np.array([2.0, 1.0, 0.1])

# Compute the softmax probabilities
softmax_probs = softmax(logits)

# Plot the softmax probabilities
plt.bar(range(len(softmax_probs)), softmax_probs)
plt.title("Softmax Activation Function")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.xticks(range(len(softmax_probs)))
plt.grid(axis="y")
plt.show()
