import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
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

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None, track_cost=False):
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

            if track_cost:
                cost = self.total_cost(training_data)
                train_acc = self.evaluate(training_data)
                test_acc = self.evaluate(test_data) if test_data else None
                history["cost"].append(cost)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                print(f"Epoch {epoch:02d}/{epochs} | Cost: {cost:.4f} "
                      f"| Train Acc: {train_acc:.2f}% "
                      f"| Test Acc: {test_acc:.2f}%" if test_data else "")
            else:
                print(f"Epoch {epoch} complete")

        return history

    def evaluate(self, data):
        if not data:
            return 0.0
        results = [(np.argmax(self.feedforward(x)), 
                   np.argmax(y) if hasattr(y, 'shape') and len(y.shape) > 0 and y.shape[0] > 1 else int(y)) 
                   for (x, y) in data]
        correct = sum(int(pred == truth) for (pred, truth) in results)
        return 100.0 * correct / len(data)

    def total_cost(self, data):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if hasattr(y, 'shape') and len(y.shape) > 0 and y.shape[0] > 1:
                target = y
            else:
                target = np.zeros((10, 1))
                target[int(y)] = 1.0
            cost += 0.5 * np.linalg.norm(a - target) ** 2
        return cost / len(data)

    def save(self, path):
        if not path.endswith('.npz'):
            path += '.npz'
        np.savez(
            path,
            sizes=np.array(self.sizes, dtype=np.int64),
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        if not path.endswith('.npz'):
            path += '.npz'
        data = np.load(path, allow_pickle=True)
        sizes = data["sizes"]
        net = cls(sizes.tolist())
        net.weights = data["weights"]
        net.biases = data["biases"]
        print(f"Model loaded from {path}")
        return net