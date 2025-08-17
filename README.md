# MNIST Feedforward Neural Network (NumPy-only)

A from-scratch implementation of a fully-connected neural network for handwritten digit recognition on MNIST, using **only NumPy** and **Tkinter** for visualization. No TensorFlow, PyTorch, or Keras dependencies.

## 🧠 Architecture

### Network Structure:

- **Input Layer**: 784 neurons (28×28 flattened grayscale pixels)
- **Hidden Layer 1**: 128 neurons (sigmoid activation)
- **Hidden Layer 2**: 64 neurons (sigmoid activation)
- **Output Layer**: 10 neurons (sigmoid activation, one-hot encoded)

### Training Configuration:

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD) with mini-batches
- **Weight Initialization**: Xavier/Glorot initialization (w ~ N(0, 1/√n_in))

## ⚙️ Hyperparameters

| Parameter     | Value | Rationale                                                                              |
| ------------- | ----- | -------------------------------------------------------------------------------------- |
| Batch Size    | 32    | Small batches for stable gradients with limited memory                                 |
| Epochs        | 30    | Sufficient for convergence on MNIST                                                    |
| Learning Rate | 0.25  | Works well with sigmoid activations; smaller slows convergence, larger may be unstable |

## 📁 Repository Structure

```
mnist-neural-network/
├── data/                        # MNIST dataset files
│   ├── train-images.idx3-ubyte  # Training images (60,000)
│   ├── train-labels.idx1-ubyte  # Training labels
│   ├── t10k-images.idx3-ubyte   # Test images (10,000)
│   └── t10k-labels.idx1-ubyte   # Test labels
├── main.py                      # Entry point, CLI, and GUI interface
├── network.py                   # Neural network implementation
├── mnist_final.npz              # Pre-trained model weights
└── README.md                    # Documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pillow
```

### Training a New Model

```bash
python main.py --train
```

### Testing Model Accuracy

```bash
python main.py --test --model mnist_final.npz
```

### Running the Interactive GUI

```bash
python main.py
```

### Using Custom Model Path

```bash
python main.py --model my_model.npz
```

## 🎨 Interactive Interface

The GUI provides a 28×28 pixel grid drawing canvas that mirrors the MNIST input format:

### Features:

- **Real-time Drawing**: Draw digits directly on the pixelated grid
- **Live Prediction**: Neural network predictions update as you draw
- **Visual Feedback**: Each grid cell darkens based on drawing intensity

### Performance Metrics:

- **Prediction**: Recognized digit (0-9)
- **Accuracy**: Model confidence percentage
- **Loss**: Mean squared error for current input
- **Clear Function**: Reset canvas for new digit

## 🔬 Technical Implementation

### Core Components

**Sigmoid Activation Function**

```
σ(z) = 1 / (1 + e^(-z))
```

**Backpropagation Algorithm**

- Forward pass: Compute activations layer by layer
- Backward pass: Compute gradients via chain rule
- Weight updates: Apply SGD with mini-batch averaging

**Loss Function (MSE)**

```
L = (1/2n) * Σ||ŷ - y||²
```

**Data Processing**

- **Normalization**: Pixel values scaled to [0,1] range
- **One-hot Encoding**: Labels converted to 10-dimensional vectors
- **Input Format**: 784×1 column vectors for each image

## 📊 Performance Expectations

**Typical Results:**

- Training Accuracy: ~98-99%
- Test Accuracy: ~95-97%
- Training Time: ~2-5 minutes (depending on hardware)
- Model Size: <50KB (.npz format)

## 🛠️ Customization

### Modifying Network Architecture

Edit the layer sizes in `main.py`:

```python
nn = NeuralNetwork([784, 32, 32, 10])  # Larger hidden layers
```

### Adjusting Hyperparameters

```python
nn.train(training_data,
         epochs=50,           # More training epochs
         mini_batch_size=20,  # Smaller batch size
         eta=0.05)           # Lower learning rate
```

## 🧪 Educational Value

This implementation demonstrates:

- **Fundamental ML Concepts**: Gradient descent, backpropagation, activation functions
- **Matrix Operations**: Efficient vectorized computations with NumPy
- **Neural Network Theory**: From scratch implementation without frameworks
- **Interactive Visualization**: Real-time model inference and feedback

## 🎯 Use Cases

- **Learning Tool**: Understand neural networks without framework abstractions
- **Prototyping**: Quick experimentation with network architectures
- **Demonstration**: Visual showcase of digit recognition capabilities
- **Research**: Baseline implementation for custom modifications

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Original dataset source
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) - Christopher Bishop

---

Built with ❤️ using pure NumPy and mathematical fundamentals
