# network.py
import numpy as np
import random
from typing import List, Tuple


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes: List[int]):
        """
        sizes: list of layer sizes, e.g. [784, 16, 16, 10]
        """
        self.sizes = sizes
        self.num_layers = len(sizes)

        # Random initialization of weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # -----------------------------
    # Core Methods
    # -----------------------------

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network given input a."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return nabla_b, nabla_w representing the gradient for the cost function.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]  # store activations layer by layer
        zs = []  # store z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        # Output layer error (MSE derivative)
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta: float):
        """Update weights and biases using gradient descent on a mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs: int, mini_batch_size: int, eta: float,
              test_data=None, track_cost=False):
        """Train the neural network with SGD."""
        training_data = list(training_data)
        n = len(training_data)
        test_data = list(test_data) if test_data else None
        history = {"cost": [], "train_acc": [], "test_acc": []}

        for epoch in range(1, epochs + 1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Track cost and accuracy
            if track_cost:
                cost = self.total_cost(training_data)
                train_acc = self.evaluate(training_data)
                test_acc = self.evaluate(test_data) if test_data else None
                history["cost"].append(cost)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                print(f"Epoch {epoch:02d}/{epochs} | Cost ~ {cost:.4f} "
                      f"| Train Acc: {train_acc:.2f}% "
                      f"| Test Acc: {test_acc:.2f}%" if test_data else "")
            else:
                print(f"Epoch {epoch} complete")

        return history

    def evaluate(self, data) -> float:
        """Return accuracy percentage on given data."""
        if not data:
            return 0.0
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        correct = sum(int(pred == truth) for (pred, truth) in results)
        return 100.0 * correct / len(data)

    def total_cost(self, data) -> float:
        """Compute average MSE cost on dataset."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += 0.5 * np.linalg.norm(a - y) ** 2
        return cost / len(data)

    # -----------------------------
    # Save & Load
    # -----------------------------

    def save(self, path: str):
        """Save model parameters to an .npz file."""
        np.savez(
            path,
            sizes=np.array(self.sizes, dtype=np.int64),
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
        )
        print(f"Model saved to {path}.npz")

    @classmethod
    def load(cls, path: str):
        """Load model parameters from an .npz file."""
        data = np.load(path, allow_pickle=True)
        sizes = data["sizes"]
        net = cls(sizes.tolist())
        net.weights = data["weights"]
        net.biases = data["biases"]
        print(f"Model loaded from {path}")
        return net
