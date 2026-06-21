# Import necessary libraries
import warnings

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Feedforward Neural Network (Multilayer Perceptron)
clf = MLPClassifier(
    hidden_layer_sizes=(5,), max_iter=1000, random_state=42, verbose=True
)

# Train the model using backpropagation
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
