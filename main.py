import argparse
import os
import numpy as np
from network import NeuralNetwork
import struct
import tkinter as tk
from PIL import Image, ImageOps


# -----------------------------
# MNIST Loading
# -----------------------------
def load_mnist(path="C:/Users/Serhii/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1"):
    """Load MNIST dataset from raw IDX files (no gzip)."""
    def read_idx(filename):
        with open(filename, "rb") as f:
            data = f.read()
            magic, num_items = struct.unpack(">II", data[:8])
            if magic == 2051:
                rows, cols = struct.unpack(">II", data[8:16])
                images = np.frombuffer(data, dtype=np.uint8, offset=16)
                return images.reshape(num_items, rows * cols) / 255.0
            elif magic == 2049:
                return np.frombuffer(data, dtype=np.uint8, offset=8)
            else:
                raise ValueError(f"Invalid MNIST file: {filename}")

    x_train = read_idx(os.path.join(path, "train-images.idx3-ubyte"))
    y_train = read_idx(os.path.join(path, "train-labels.idx1-ubyte"))
    x_test = read_idx(os.path.join(path, "t10k-images.idx3-ubyte"))
    y_test = read_idx(os.path.join(path, "t10k-labels.idx1-ubyte"))

    x_train = [x.reshape(784, 1) for x in x_train]
    x_test = [x.reshape(784, 1) for x in x_test]

    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    training_data = list(zip(x_train, [vectorized_result(y) for y in y_train]))
    test_data = list(zip(x_test, y_test))

    return training_data, test_data


def vectorized_result(j: int) -> np.ndarray:
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# -----------------------------
# Tkinter Drawing GUI
# -----------------------------
class DigitDrawer:
    def __init__(self, model: NeuralNetwork):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Draw a digit (0–9)")

        self.canvas_size = 280
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.loss_label = tk.Label(self.window, text="Loss: N/A", font=("Arial", 16))
        self.loss_label.pack()
        self.pred_label = tk.Label(self.window, text="Prediction: N/A", font=("Arial", 24))
        self.pred_label.pack()
        self.conf_label = tk.Label(self.window, text="Confidence: N/A", font=("Arial", 16))
        self.conf_label.pack()

        self.clear_btn = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.last_update = None

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

        for i in range(-r, r):
            for j in range(-r, r):
                if 0 <= x + i < self.canvas_size and 0 <= y + j < self.canvas_size:
                    self.image.putpixel((x + i, y + j), 0)

        if self.last_update is None or (self.window.tk.call('after', 'info') == ''):
            self.window.after(100, self.predict_and_stats)

    def predict_and_stats(self):
        img = self.image.resize((28, 28)).convert("L")
        img = ImageOps.invert(img)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.reshape(-1, 1)

        output = self.model.feedforward(arr)
        prediction = np.argmax(output)
        confidence = float(output[prediction])

        target = np.zeros((10, 1))
        target[prediction] = 1.0
        loss = float(np.mean((output - target) ** 2))

        self.pred_label.config(text=f"Prediction: {prediction}")
        self.loss_label.config(text=f"Loss (MSE): {loss:.4f}")
        self.conf_label.config(text=f"Confidence: {confidence:.2%}")

        self.last_update = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.pred_label.config(text="Prediction: N/A")
        self.loss_label.config(text="Loss: N/A")
        self.conf_label.config(text="Confidence: N/A")

    def run(self):
        self.window.mainloop()



# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the network")
    parser.add_argument("--model", type=str, default="mnist_final.npz", help="Path to save/load model")
    args = parser.parse_args()

    if args.train:
        training_data, test_data = load_mnist()
        nn = NeuralNetwork([784, 32, 32, 10])
        nn.train(training_data, epochs=30, mini_batch_size=32, eta=3.0,
                 test_data=test_data, track_cost=True)
        nn.save(args.model)
    else:
        nn = NeuralNetwork.load(args.model)

    drawer = DigitDrawer(nn)
    drawer.run()


if __name__ == "__main__":
    main()
