# MNIST Feedforward Neural Network (NumPy-only)

A from-scratch implementation of a fully-connected neural network for handwritten digit recognition on MNIST, using **only NumPy** (plus Matplotlib for visuals). No TensorFlow, PyTorch, or Keras.

## Architecture

- Input: 784 (28x28 flattened grayscale)
- Hidden1: 16 (sigmoid)
- Hidden2: 16 (sigmoid)
- Output: 10 (sigmoid)
- Loss: Mean Squared Error (MSE)
- Optimizer: SGD with mini-batches

**Hyperparameters**

- Batch size: `10`
- Epochs: `30`
- Learning rate: `0.1`  
  With sigmoid activations and small layers, `0.1` converges reliably for MNIST. Smaller slows learning; larger often destabilizes early updates.

## Repository Structure
