# MLP Neural Network in C++

This project is a from-scratch implementation of a Multi-Layer Perceptron (MLP) neural network written in C++. It includes support for training with stochastic gradient descent, learning rate decay, and flexible layer definitions.

## Features

- Customizable number of layers and neurons
- ReLU activation function
- Forward and backward propagation
- Stochastic Gradient Descent (SGD) optimizer
- Learning rate decay
- Basic data loading utility

## Requirements

- C++11 or later
- Standard C++ libraries (no external dependencies)

## Data Format

- The program expects CSV files with numerical data.
- One of the columns should be the target label (you can specify which via `targetIdx`).
- Input features are normalized by dividing pixel values by `255.0`, as it was designed with MNIST data in mind.


