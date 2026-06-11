
"""
=======================================================================================
Python script to demonstrate an artificial neural network 
=======================================================================================

This program implements a simple feed-forward neural network with one hidden layer, 
trained for manual forward propagation, forward propagation, and gradient descent,
to learn the XOR function.
"""

# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Sigmoid function and sigmoid derivative
# -----------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1.0 + x)

# -----------------------------------------------------------------------------
# 2. Neural Network class
# -----------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(
    self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float,
        seed: int = 42
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)

        # Weights are initialized randomly (small values) so that neurons start with different parameters and can learn distinct features during training
        self.weights_input_hidden: np.ndarray = rng.uniform(
            -1.0, 1.0, size=(self.input_size, self.hidden_size)
        )

        self.bias_hidden: np.ndarray = np.zeros(( 1, hidden_size ))

        self.weights_input_hidden: np.ndarray = rng.uniform(
            -1.0, 1.0, size=(self.hidden_size, self.output_size)
        )
        self.bias_output: np.ndarray = np.zeros((1, output_size))

        # Placeholders that will store intermediate values computed during forward propagation. These are required again during backward propagation, so they are stored as attributes
        self.hidden_weighted_sum: np.ndarray = np.zeros((1, hidden_size))
        self.hidden_activation: np.ndarray = np.zeros((1, hidden_size))
        self.output_weighted_sum: np.ndarray = np.zeros((1, output_size))
        self.output_activation: np.ndarray = np.zeros((1, output_size))

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        # hidden layer computations
        self.hidden_weighted_sum = (inputs @ self.weights_input_hidden + self.bias_hidden)
        self.hidden_activation = sigmoid(self.hidden_weighted_sum)

        # output layer computations
        self.output_weighted_sum = (self.hidden_activation @ self.weights_input_hidden + self.bias_output)

        self.output_weighted_sum = sigmoid(self.output_weighted_sum)

        return self.output_activation

    @staticmethod
    def calculate_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(targets - predictions) ** 2)





# -----------------------------------------------------------------------------
# 3. Main execution function
# -----------------------------------------------------------------------------
def main() -> None:

# -----------------------------------------------------------------------------
# 4. Run the script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
