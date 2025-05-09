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

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JohnnyLee15/MLPNeuralNetProject.git
   cd MLPNeuralNetProject
   ```

2. **Compile the code**:

   If you have `make` installed, you can compile the program using the following command:

   ```bash
   make
   ```

3. **Run the program**:

   Once the code is compiled, you can run the MLP neural network with:

   ```bash
   ./mlp
   ```

4. **Clean up the build**:

To remove the object files and the compiled `mlp` executable, use the `make clean` command:

```bash
make clean
```

5. **Customization**:

   You can modify the number of layers, neurons, learning rate, or other parameters by adjusting the source code or adding your own configuration options as needed.



