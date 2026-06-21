import matplotlib.pyplot as plt
import numpy as np


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.uniform(
            -1, 1, (self.input_size, self.hidden_size)
        )
        self.weights_hidden_output = np.random.uniform(
            -1, 1, (self.hidden_size, self.output_size)
        )
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

    def forward(self, inputs):
        # Calculate input to hidden layer
        self.hidden_inputs = (
            np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        )
        # Apply activation function (sigmoid) to the hidden layer
        self.hidden_outputs = sigmoid(self.hidden_inputs)

        # Calculate input to output layer
        self.output_inputs = (
            np.dot(self.hidden_outputs, self.weights_hidden_output) + self.bias_output
        )
        # Apply activation function (sigmoid) to the output layer
        self.output_outputs = sigmoid(self.output_inputs)

        return self.output_outputs

    def backward(self, inputs, targets, learning_rate):
        # Calculate error at the output layer
        output_errors = targets - self.output_outputs
        # Calculate derivative of the output layer activation function (sigmoid)
        output_delta = output_errors * sigmoid_derivative(self.output_outputs)

        # Calculate error at the hidden layer
        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        # Calculate derivative of the hidden layer activation function (sigmoid)
        hidden_delta = hidden_errors * sigmoid_derivative(self.hidden_outputs)
        # Update weights and biases
        self.weights_hidden_output += (
            np.dot(self.hidden_outputs.T, output_delta) * learning_rate
        )
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate


# Generate synthetic data for training
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.array([int(x1 + x2 > 1) for x1, x2 in X])

# Initialize neural network
input_size = 2
hidden_size = 3
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training the neural network
epochs = 1000
learning_rate = 0.1
losses = []

for epoch in range(epochs):
    # Forward pass
    outputs = nn.forward(X)
    # Backward pass
    nn.backward(X, y.reshape(-1, 1), learning_rate)
    # Calculate loss (MSE)
    loss = np.mean(np.square(y.reshape(-1, 1) - outputs))
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Plotting the loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
