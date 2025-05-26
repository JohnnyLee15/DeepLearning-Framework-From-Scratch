# MLP Neural Network in C++ 🤖

This project is a from-scratch implementation of a Multi-Layer Perceptron (MLP) neural network written in C++. It includes support for training with stochastic gradient descent, learning rate decay, batching, and flexible layer definitions.

## Features ✨

- Customizable number of layers and neurons
- ReLU and Softmax activation functions
- Forward and backward propagation
- Stochastic Gradient Descent (SGD) optimizer
- Learning rate decay
- Batching support for efficient training
- Basic data loading utility

## Requirements ⚙️

- C++11 or later
- OpenMP (for parallelization)
- Standard C++ libraries (no external dependencies)

## Supported Platforms 🖥️

- **macOS** (via Homebrew LLVM + libomp)
- **Linux** (Ubuntu, Debian, etc.)
- **Windows via WSL2** (Ubuntu or other Linux distros)

This project is intended for Unix-like environments. Native Windows builds are not supported.

## Installing OpenMP 🧩

If you're compiling with g++, OpenMP is typically included. To install it manually:

- On **macOS**:
  - Make sure you have [Homebrew](https://brew.sh/) installed.
  - Then run:
    ```bash
    brew install libomp
    ```

- On **Linux / WSL2 (Ubuntu, Debian, etc.)**:
   ```bash
   sudo apt install libomp-dev
   ```

- On **Windows**:
  - Native Windows builds (e.g., MSYS2, MinGW, or Visual Studio) are **not supported**.
  - Instead, please use [WSL2 (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/) with a Linux distribution like Ubuntu.

## Data Format 📂

- The program expects CSV files with numerical data.
- One of the columns should be the target label (you can specify which via `targetIdx`).

## How to Run 🚀

1. Clone the repository:

   ```bash
   git clone https://github.com/JohnnyLee15/MLPNeuralNetProject.git
   cd MLPNeuralNetProject
   ```

2. Compile the code:

   If you have `make` installed, you can compile the program using:

   ```bash
   make
   ```

   This uses OpenMP flags internally via the Makefile.

3. Run the program:

   Once compiled, you can run the MLP neural network:

   ```bash
   ./mlp
   ```

4. Clean the build:

   To remove the object files and compiled binary:

   ```bash
   make clean
   ```

## Customization 🛠️

   You can modify the number of layers, neurons, learning rate, epochs, and batch size in the source code (`Main.cpp`).

## License ⚖️ 
This project is licensed under the MIT License – see the [LICENSE](https://opensource.org/licenses/MIT) file for details.