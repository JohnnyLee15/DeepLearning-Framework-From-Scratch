# MLP Neural Network in C++ 🤖

This is a **from-scratch, modular implementation of a Multi-Layer Perceptron (MLP)** written in modern C++. It supports classification and regression tasks, fast matrix operations, and intuitive data handling — all without external ML libraries.

## Features ✨

- 🔧 **Fully customizable architecture**: any number of layers, neurons, and activation functions
- 🧠 **Supports classification and regression** with built-in task-specific utilities
- 🏃 **Efficient forward and backward propagation** with OpenMP parallelization
- 📆 **Mini-batch SGD training** with learning rate decay
- 📂 **Binary model saving/loading** (`.nn` format) with overwrite and rename support
- 🧪 **Clear console UI** for data loading, model saving, error handling, and training progress
- 📊 **Evaluation utilities** for MAPE, accuracy, and progress bars
- 📁 **CSV data loading** with automatic feature/target extraction and automatic one-hot encoding for categorical features
- 🌈 **Scaler support**: greyscale normalization and min-max scaling
- 🛠️ No external ML libraries – 100% custom C++

## Requirements ⚙️

- C++17 or later
- OpenMP (for multithreading)
- `make` (for building the project)
- Standard libraries only – no third-party dependencies

## Supported Platforms 🖥️

- **macOS** (via Homebrew + LLVM + libomp)
- **Linux** (Ubuntu/Debian)
- **Windows via WSL2** (Ubuntu or other Linux distros)

## Installing OpenMP 🧩

If you're compiling with g++, OpenMP is typically included. To install it manually:

- **macOS**:

  - Make sure you have [Homebrew](https://brew.sh/) installed.
  - Then run:
  ```bash
  brew install libomp
  ```

- **Linux / WSL2**:

  ```bash
  sudo apt install libomp-dev
  ```

- ❌ Native Windows builds are not supported — use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/).

## Usage 🚀

1. Clone the repository:

  ```bash
  git clone https://github.com/JohnnyLee15/MLPNeuralNetProject.git
  cd MLPNeuralNetProject
  ```

2. Compile the code:

  ```bash
  make
  ```

  The Makefile automatically includes the necessary OpenMP flags for parallelization support.

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

## Data Format 📂

- Input files must be in **CSV format**.
- The file must contain a **target column** (label or value).
- You can specify the **column name or index** for the target in the API:
  ```cpp
  data.readTrain("path.csv", "label");
  ```

## Model Saving / Loading 📂

- Models are saved in compact **binary format** (`.nn`) using:

  ```cpp
  nn.saveToBin("model.nn");
  ```

- Existing files will prompt for:

  - `[q]` Cancel
  - `[o]` Overwrite
  - `[r]` Rename

- Models can be loaded using:

  ```cpp
  NeuralNet nn = NeuralNet::loadFromBin("model.nn");
  ```

## Customization 🛠️

You can modify:

- Layer structure (in `Main.cpp`)
- Activation functions (`ReLU`, `Linear`, `Softmax`)
- Loss functions (`MSE`, `SoftmaxCrossEntropy`)
- Task type (`RegressionTask` or `ClassificationTask`)
- Training parameters:
  - `learningRate`
  - `decayRate`
  - `epochs`
  - `batchSize`

---

## License ⚖️

MIT License – see [LICENSE](https://opensource.org/licenses/MIT)

