import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=100):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for epoch in range(self.max_epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error
                    errors += 1
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
            elif epoch == self.max_epochs - 1:
                print("Max epochs reached. Perceptron could not converge.")


# Example usage
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# Test the trained perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for test_input in test_inputs:
    print(f"{test_input} --> {perceptron.predict(test_input)}")
