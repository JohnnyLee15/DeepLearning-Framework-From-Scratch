# DeepLearning Framework in C++/Objective-C++ with Metal GPU Acceleration 🤖

This project is a **from-scratch, modular implementation of a DeepLearning framework** written in modern **C++** and **Objective-C++**, with custom **Metal GPU kernels** for acceleration on macOS. It now supports both **tabular** and **image data**, and includes a full **training pipeline**. No external ML libraries required.

## What Makes This Project Unique 🌟

This project started as a **from-scratch MLP experiment** and has grown into a **transparent, high-performance deep learning framework**.  
A few aspects that make it noteworthy:  

- **Truly from scratch** – implemented entirely in modern **C++** and **Objective-C++**, with no external ML libraries  
- **Custom GPU acceleration** – **Metal GPU kernels** on macOS, with **OpenMP fallback** on other platforms  
- **End-to-end pipelines** – save and load models, weights, scalers, and image transforms in a single compact `.nn` file  
- **Full training workflow** – learning rate decay, validation monitoring, early stopping, and multiple evaluation metrics  
- **Versatile data support** – works with both **tabular (CSV)** and **image datasets**, with preprocessing utilities included  

The goal is to bridge **learning-by-building** with **practical tools** for reproducible machine learning experiments.

## Features ✨

- **Core Layers**
  - **Dense**
  - **Conv2D**
  - **MaxPooling2D**
  - **Dropout**
  - **GlobalAveragePooling2D**
  - **Flatten**

- **Regularization & Optimization**
  - **L2 weight regularization** for Dense and Conv2D layers
  - **Mini-batch SGD** with learning rate decay
  - **Validation set support** during training
  - **Early stopping** with automatic saving of best weights (based on validation loss)

- **Data Handling**
  - **Tabular and image data support**
  - **Built-in data splitters** for training, validation, and test sets
  - **CSV parsing, image transforms, scalers** (min-max, greyscale normalization)

- **Pipeline Management**
  - Save & load entire pipelines, including:
  - **Feature scalers**
  - **Target scalers**
  - **Data type (tabular / image)**
  - **Model architecture & weights**
  - **Image transformer**

- **Training & Evaluation Utilities**
  - **Progress bars** with accuracy/loss reporting
  - **Validation monitoring** for early stopping
  - **Metrics:** accuracy, MAPE, RMSE, etc.
  - **Console UI** for training & error handling

- **Performance**
  - **Custom Metal GPU kernels** for macOS GPU acceleration
  - **Optimized C++ matrix operations** with OpenMP if not using macOS GPU acceleration
  - **100% from scratch** — no TensorFlow, PyTorch, or external ML libs

## Requirements ⚙️

- **C++17 or later**
- **OpenMP** for CPU multithreading
- **`make`** for building the project
- **macOS** with Xcode Command Line Tools and Metal support for GPU acceleration
- **Standard libraries only** – no third-party dependencies

## Supported Platforms 🖥️

- **macOS**  
- **Linux**  
- **Windows via WSL2**

## Installing OpenMP 🧩

If you're compiling with `g++`, OpenMP is typically included. To install it manually:

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

- **Native Windows builds are not supported** — use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/).

## Installing Xcode Command Line Tools 🛠️

On macOS, install the required Xcode Command Line Tools from the app store or with:

```bash
xcode-select --install
```

## Usage 🚀

1. Clone the repository:

  ```bash
  git clone https://github.com/JohnnyLee15/DeepLearning-Framework-From-Scratch.git
  cd DeepLearning-Framework-From-Scratch
  ```

2. Compile the code:

  ```bash
  make
  ```

  The Makefile automatically includes the necessary OpenMP flags for parallelization support.

3. Run the program:

  Once compiled, you can run the framework with:

  ```bash
  ./runModel
  ```

4. Clean the build:

  To remove the object files and compiled binary:

  ```bash
  make clean
  ```

## Data Format 📂

### Tabular Data
- Input files must be in **CSV format**.
- Datasets must be complete: **All rows should contain the same number of columns with no missing values**. Incomplete datasets will cause parsing errors.
- The file must contain a **target column** (label or value).
- You can specify the **column name or index** for the target in the API:
  ```cpp
  data.readTrain("path.csv", "label");
  ```

### Image Data
- Input files should be organized in a **directory structure** by class label, for example:
  ```bash
  Data/
    train/
      class1/
      class2/
    test/
      class1/
      class2/
  ```
- Images can be **preprocessed** (resized, normalized, greyscale/RGB) through the built-in image transformer.
- Targets are **automatically inferred** from the folder names.

## Model Saving / Loading 📂

Models are managed through a **Pipeline** object, which can encapsulate everything needed to reproduce training and inference. A pipeline may include:   

- **Model architecture + trained weights**  
- **Data object** (tabular or image)  
- **Feature scalar (optional)**  
- **Target scalar (optional)**  
- **Image transformer (optional, for image data)**  

No parameters are strictly required (you can save an empty pipeline), but in practice you will usually set at least a **model** and the **data object** used during training.  

Pipelines are stored in **compact binary format (`.nn`)**.  

### Saving
```cpp
Pipeline pipe;                               // Create pipeline
pipe.setData(data);                          // Data object (optional, but common)
pipe.setFeatureScalar(featureScalar);        // Optional: Feature scalar
pipe.setTargetScalar(targetScalar);          // Optional: Target scalar
pipe.setModel(nn);                           // Optional: Model
pipe.setImageTransformer2D(transformer);     // Optional: Image transformer
pipe.saveToBin("ExampleModel");              // Saves to ExampleModel.nn
```

### Loading
```cpp
Pipeline pipe = Pipeline::loadFromBin("ExampleModel");        

// Access components as needed:
ImageData2D data = *dynamic_cast<ImageData2D*>(pipe.getData());
ImageTransform2D transformer = *pipe.getImageTransformer();
Scalar *featureScalar = pipe.getFeatureScalar();
Scalar *targetScalar = pipe.getTargetScalar();      
NeuralNet *nn = pipe.getModel();
```

### Notes
- All pipeline components are **optional**.
  - **Image tasks (CNNs):** usually include an ImageTransformer, but not scalars.
  - **Tabular tasks:** often include feature/target scalars, but not an image transformer.
- Pipelines automatically **restore all saved components** for consistent inference.
- If a file already exists, you’ll be prompted:
  - `[q]` Cancel
  - `[o]` Overwrite
  - `[r]` Rename

### **Important when loading**

- **Tabular data:**  
  - Your training and testing datasets must use the **same columns in the same order** as the data used when the model was originally trained.  
  - If you change your features (e.g., add or remove columns), you **must retrain the model** because the architecture and scalars won’t auto-adjust.  
  - **Unseen feature categories** in your test data will be skipped safely (e.g., one-hot will be all zeros). To handle new categories, retrain the model with the expanded dataset.  
  - **Classification:** If your test data contains unseen **target labels**, this will throw an error since the saved label map has no encoding for them.  
  - When using scalars, call `scalar->transform()` to scale input data before training or testing, and `scalar->reverseTransform()` on model outputs if you need results back in the original scale.  

- **Image data:**  
  - Do not rely on auto-transforming, you must apply your **image transforms manually** before training and testing.  

- **General:**  
  - To save memory, call `.clear()` on unused raw data or tensors once you’ve prepared your transformed versions. This prevents holding multiple large copies of the same dataset in memory.  

## Customization 🛠️

You can customize the framework in several ways:

- **Layer architecture** – freely define the sequence of layers in your network.
- **Available layers** – `Dense`, `Conv2D`, `MaxPooling2D`, `Dropout`, `GlobalAveragePooling2D (GAP)`, and `Flatten`.
- **Activations** – `ReLU`, `Softmax`, `Linear`.
- **Loss functions** – `SoftmaxCrossEntropy`, `MSE`.
- **Regularization** – optional **L2 weight regularization** for `Dense` and `Conv2D`.
- **Training parameters** – learning rate, decay, epochs, batch size.
- **Validation & early stopping** – monitor validation performance and stop automatically when the model stops improving.
- **Data transforms** – apply scalars for tabular data, or image transforms (resize, normalize, greyscale/RGB) for image tasks.
- **Hardware acceleration** – enable GPU execution on macOS via Metal, or use CPU with OpenMP.
- **Saving & loading** – manage pipelines (model, data, scalars, transforms) with the `Pipeline` class.

## Example Usage: MNIST Classification 🎯

### Setup

```cpp
// Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
GpuEngine::init();

// Data Reading
const string trainPath = "DataFiles/MNIST/mnist_train.csv";
const string testPath = "DataFiles/MNIST/mnist_test.csv";
const string targetColumn = "label";

TabularData *data = new TabularData("classification");
data->readTrain(trainPath, targetColumn);
data->readTest(testPath, targetColumn);

// Splitting training data into train and validation sets
Split split = DataSplitter::stratifiedSplit(
    data->getTrainFeatures(), data->getTrainTargets(), 0.1f
);

// Scaling Data
Scalar *scalar = new Greyscale();
scalar->fit(split.xTrain);
const Tensor xTrain = scalar->transform(split.xTrain);
const Tensor xTest = scalar->transform(data->getTestFeatures());
const Tensor xVal = scalar->transform(split.xVal);

const vector<float> yTrain = split.yTrain;
const vector<float> yTest = data->getTestTargets();
const vector<float> yVal = split.yVal;

// Clearing unused data to save memory
data->clearTrain();
data->clearTest();
split.clear();

// Defining Model Architecture
Loss *loss = new SoftmaxCrossEntropy();
vector<Layer*> layers = {
    new Dense(512, new ReLU(), 1e-4f), // last parameter is l2 regularization
    new Dense(128, new ReLU(), 1e-4f), // last parameter is l2 regularization
    new Dropout(0.5), 
    new Dense(10, new Softmax())
};

// Creating Neural Network
NeuralNet *nn = new NeuralNet(layers, loss);

// Creating Early Stop Object
EarlyStop *stop = new EarlyStop(1, 1e-4, 5); // (patience, min delta, warm-up)

// Training Model
ProgressMetric *metric = new ProgressAccuracy();
nn->fit(
    xTrain, // Features
    yTrain, // Targets
    0.01,   // Learning rate
    0.01,   // Learning rate decay
    2,      // Number of epochs
    32,     // Batch Size
    *metric, // Progress metric
    xVal,  // Validation features
    yVal,   // Validation targets
    stop    // Early stop object
);

// Saving Model
Pipeline pipe;
pipe.setData(data);
pipe.setFeatureScalar(scalar);
pipe.setModel(nn);
pipe.saveToBin("models/ClassMnistTrain.nn");

// Testing Model
Tensor output = nn->predict(xTest);
vector<float> predictions = TrainingUtils::getPredictions(output);
float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
printf("\nTest Accuracy: %.2f%%.\n", accuracy);

// Delete pointers that don't belong to pipe
delete stop;
delete metric;
```

### Output

```
============================================================
              🧠 C++ NEURAL NETWORK TOOLKIT
         MLP & CNN For Classification & Regression
============================================================

📥 Loading training data from: "mnist_train.csv".
[✔] Loading Data.
[✔] Parsing Lines.
[✔] Extracting Features.
[✔] Extracting Targets.
------------------------------------------------------------

📥 Loading testing data from: "mnist_test.csv".
[✔] Loading Data.
[✔] Parsing Lines.
[✔] Extracting Features.
[✔] Extracting Targets.
------------------------------------------------------------

✂️  Splitting 60000 samples: 54004 | 5996
[✔] Splitting data with stratification.
------------------------------------------------------------

Epoch: 1/2
54004/54004 |██████████████████████████████████████████████████| Accuracy: 77.57%| Avg Loss: 0.75 | Elapsed: 1.44s
Avg Validation Loss: 0.33 | Validation Accuracy: 90.68%

Epoch: 2/2
54004/54004 |██████████████████████████████████████████████████| Accuracy: 89.09%| Avg Loss: 0.38 | Elapsed: 1.34s
Avg Validation Loss: 0.26 | Validation Accuracy: 92.73%
------------------------------------------------------------

[✔] Model saved successfully as "models/ClassMnistTrain.nn".
------------------------------------------------------------

Test Accuracy: 93.16%.
```

## Example Usage: California Housing Regression 🏡

### Setup

```cpp
// Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
GpuEngine::init();

// Data Reading
const string dataPath = "DataFiles/California_Housing/housing_clean.csv";
const string targetColumn = "median_house_value";

TabularData *data = new TabularData("regression");
data->readTrain(dataPath, targetColumn);

// Splitting training data into train, test, and validation sets
Split splitTest = DataSplitter::stratifiedSplit(
    data->getTrainFeatures(), data->getTrainTargets(), 0.2f
);

Split splitVal = DataSplitter::stratifiedSplit(
    splitTest.xTrain, splitTest.yTrain, 0.1f
);

// Scaling Features
Scalar *featureScalar = new Minmax();
featureScalar->fit(splitVal.xTrain);
const Tensor xTrain = featureScalar->transform(splitVal.xTrain);
const Tensor xTest = featureScalar->transform(splitTest.xVal);
const Tensor xVal = featureScalar->transform(splitVal.xVal);


// Scaling Targets
Scalar *targetScalar = new Minmax();
targetScalar->fit(splitVal.yTrain);
const vector<float> yTrain = targetScalar->transform(splitVal.yTrain);
const vector<float> yVal = targetScalar->transform(splitVal.yVal);
const vector<float> yTest = splitTest.yVal;

// Clearing unused data to save memory
data->clearTrain();
splitTest.clear();
splitVal.clear();

// Defining Model Architecture
Loss *loss = new MSE();
vector<Layer*> layers = {
    new Dense(512, new ReLU()),
    new Dense(256, new ReLU()),
    new Dense(128, new ReLU()),
    new Dense( 64, new ReLU()),
    new Dense(  1, new Linear())
};

// Creating Early Stop Object
EarlyStop *stop = new EarlyStop(1, 1e-4, 0); // (patience, min delta, warm-up)

// Creating Neural Network
NeuralNet *nn = new NeuralNet(layers, loss);

// Training Model
ProgressMetric *metric = new ProgressMAPE();
nn->fit(
    xTrain,  // Features
    yTrain,  // Targets
    0.01,   // Learning rate
    0.01,  // Learning rate decay
    2,     // Number of epochs
    32,      // Batch Size
    *metric,  // Progress metric
    xVal,  // Validation features
    yVal,   // Validation targets
    stop    // Early stop object
);

// Saving Model
Pipeline pipe;
pipe.setData(data);
pipe.setFeatureScalar(featureScalar);
pipe.setTargetScalar(targetScalar);
pipe.setModel(nn);
pipe.saveToBin("models/RegressHousingTrain");

// Testing Model
Tensor output = nn->predict(xTest);
Tensor predictions = targetScalar->reverseTransform(output);
float rmse = TrainingUtils::getRMSE(predictions, yTest);
printf("\nTest RMSE: %.2f.\n", rmse);
```

### Output

```
============================================================
              🧠 C++ NEURAL NETWORK TOOLKIT
         MLP & CNN For Classification & Regression
============================================================

📥 Loading training data from: "housing_clean.csv".
[✔] Loading Data.
[✔] Parsing Lines.
[✔] Extracting Features.
[✔] Extracting Targets.
------------------------------------------------------------

✂️  Splitting 20433 samples: 17840 | 2593
[✔] Splitting data with stratification.
------------------------------------------------------------

✂️  Splitting 17840 samples: 17446 | 394
[✔] Splitting data with stratification.
------------------------------------------------------------

Epoch: 1/2
17446/17446 |██████████████████████████████████████████████████| MAPE: 37.50%| Avg Loss: 0.16 | Elapsed: 0.54s
Avg Validation Loss: 0.20 | Validation MAPE: 32.22%

Epoch: 2/2
17446/17446 |██████████████████████████████████████████████████| MAPE: 32.39%| Avg Loss: 0.14 | Elapsed: 0.37s
Avg Validation Loss: 0.22 | Validation MAPE: 32.09%
------------------------------------------------------------

[✔] Model saved successfully as "models/RegressHousingTrain.nn".
------------------------------------------------------------

Test RMSE: 68862.59.
```

## Example Usage: Chest X-Ray Classification 🫁

### Setup

```cpp
// Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
GpuEngine::init();

// Image Resize Dims
const size_t SIZE = 128;

// Number of channels to read in
const size_t CHANNELS = 1;

// Data Paths
const string dataPath = "DataFiles/kaggle_chest_xray";

// Data Reading
ImageData2D *data = new ImageData2D(CHANNELS);
data->readTrain(dataPath);

// Transform data (resize to 128x128 and normalize)
ImageTransform2D *transformer = new ImageTransform2D(SIZE, SIZE, CHANNELS);
Tensor x = transformer->transform(data->getTrainFeatures());
vector<float> y = data->getTrainTargets();

// Splitting training data into train, test, and validation sets
Split splitTest = DataSplitter::stratifiedSplit(x, y, 0.2f);
Split splitVal = DataSplitter::stratifiedSplit(splitTest.xTrain, splitTest.yTrain, 0.1f);

const Tensor xTrain = splitVal.xTrain;
const Tensor xTest = splitTest.xVal;
const Tensor xVal = splitVal.xVal;

const vector<float> yTrain = splitVal.yTrain;
const vector<float> yTest = splitTest.yVal;
const vector<float> yVal = splitVal.yVal;

// Clearing unused data
data->clearTrain();
x.clear();
y.clear();
splitTest.clear();
splitVal.clear();

// Defining Model Architecture
Loss *loss = new SoftmaxCrossEntropy();
vector<Layer*> layers = {
    new Conv2D(32, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization 
    new Conv2D(32, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization
    new MaxPooling2D(2, 2, 2, "none"),

    new Conv2D(64, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization
    new Conv2D(64, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization
    new MaxPooling2D(2, 2, 2, "none"),

    new Conv2D(128, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization
    new Conv2D(128, 3, 3, 1, "same", new ReLU(), 1e-4f), // last parameter is l2 regularization
    new MaxPooling2D(2, 2, 2, "none"),

    new Flatten(),
    new Dense(128, new ReLU(), 1e-4f), // last parameter is l2 regularization
    new Dropout(0.4f),
    new Dense(2, new Softmax())
};

// Creating Neural Network
NeuralNet *nn = new NeuralNet(layers, loss);

// Creating Early Stop Object
EarlyStop *stop = new EarlyStop(8, 5e-4f, 5); // (patience, min delta, warm-up)

// Training Model
ProgressMetric *metric = new ProgressAccuracy();
nn->fit(
    xTrain, // Features
    yTrain, // Targets
    0.01f,  // Learning rate
    0.0f,    // Learning rate decay
    2,      // Number of epochs
    32,     // Batch Size
    *metric, // Progress metric
    xVal,  // Validation features
    yVal,   // Validation targets
    stop    // Early stop object
);

// Saving Model
Pipeline pipe;
pipe.setData(data);
pipe.setModel(nn);
pipe.setImageTransformer2D(transformer);
pipe.saveToBin("models/XrayCNNTrain");

// Testing Model
Tensor output = nn->predict(xTest);
vector<float> predictions = TrainingUtils::getPredictions(output);
float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
printf("\nTest Accuracy: %.2f%%.\n", accuracy);

// Delete pointers that don't belong to pipe
delete stop;
delete metric;
```

### Output

```
============================================================
              🧠 C++ NEURAL NETWORK TOOLKIT
         MLP & CNN For Classification & Regression
============================================================

📥 Loading training data from: "kaggle_chest_xray".
[✔] Scanning Image Directories.
[✔] Extracting Images.
[✔] Extracting Targets.
------------------------------------------------------------

🎨 Transforming 5856 images.
[✔] Resizing & Normalizing images.
------------------------------------------------------------

✂️  Splitting 5856 samples: 4686 | 1170
[✔] Splitting data with stratification.
------------------------------------------------------------

✂️  Splitting 4686 samples: 4219 | 467
[✔] Splitting data with stratification.
------------------------------------------------------------

Epoch: 1/2
4219/4219 |██████████████████████████████████████████████████| Accuracy: 73.74%| Avg Loss: 0.61 | Elapsed: 64.55s
Avg Validation Loss: 0.41 | Validation Accuracy: 82.87%

Epoch: 2/2
4219/4219 |██████████████████████████████████████████████████| Accuracy: 85.04%| Avg Loss: 0.36 | Elapsed: 65.48s
Avg Validation Loss: 0.21 | Validation Accuracy: 92.72%
------------------------------------------------------------

[!] File "models/XrayCNNTrain.nn" already exists.
   [q] Cancel save.
   [o] Overwrite existing file.
   [r] Rename and save as new file.

[>] Enter your choice: o
[!] Overwriting existing model.

[✔] Model saved successfully as "models/XrayCNNTrain.nn".
------------------------------------------------------------

Test Accuracy: 91.54%.
```

## Example Usage: Loading a Saved Pipeline 📦

### Setup

```cpp
// Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
GpuEngine::init();

// Data paths
const string dataPath = "DataFiles/kaggle_chest_xray";

// Loading Model
Pipeline pipe = Pipeline::loadFromBin("models/XrayCNNTrain");
ImageData2D *data = dynamic_cast<ImageData2D*>(pipe.getData());
ImageTransform2D *transformer = pipe.getImageTransformer();
NeuralNet *nn = pipe.getModel();

// Data Reading
data->readTrain(dataPath);

// Transform data (resize to 128x128 and normalize)
Tensor x = transformer->transform(data->getTrainFeatures());
vector<float> y = data->getTrainTargets();

// Splitting training data into train, test, and validation sets
Split splitTest = DataSplitter::stratifiedSplit(x, y, 0.2f);
Split splitVal = DataSplitter::stratifiedSplit(splitTest.xTrain, splitTest.yTrain, 0.1f);

const Tensor xTrain = splitVal.xTrain;
const Tensor xTest = splitTest.xVal;
const Tensor xVal = splitVal.xVal;

const vector<float> yTrain = splitVal.yTrain;
const vector<float> yTest = splitTest.yVal;
const vector<float> yVal = splitVal.yVal;

// Clearing unused data
data->clearTrain();
x.clear();
y.clear();
splitTest.clear();
splitVal.clear();

// Training Data
ProgressMetric *metric = new ProgressAccuracy();
nn->fit(
    xTrain, // Features
    yTrain, // Targets
    0.01f,  // Learning rate
    0.0f,    // Learning rate decay
    1,      // Number of epochs
    32,     // Batch Size
    *metric // Progress metric
);

// Save Model
pipe.saveToBin("models/XrayCNNLoad");

// Testing Model
Tensor output = nn->predict(xTest);
vector<float> predictions = TrainingUtils::getPredictions(output);
float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
printf("\nTest Accuracy: %.2f%%.\n", accuracy);
```

### Output

```
============================================================
              🧠 C++ NEURAL NETWORK TOOLKIT
         MLP & CNN For Classification & Regression
============================================================

[✔] Model successfully loaded from "models/XrayCNNTrain.nn".
------------------------------------------------------------

📥 Loading training data from: "kaggle_chest_xray".
[✔] Scanning Image Directories.
[✔] Extracting Images.
[✔] Extracting Targets.
------------------------------------------------------------

🎨 Transforming 5856 images.
[✔] Resizing & Normalizing images.
------------------------------------------------------------

✂️  Splitting 5856 samples: 4686 | 1170
[✔] Splitting data with stratification.
------------------------------------------------------------

✂️  Splitting 4686 samples: 4219 | 467
[✔] Splitting data with stratification.
------------------------------------------------------------

Epoch: 1/1
4219/4219 |██████████████████████████████████████████████████| Accuracy: 91.28%| Avg Loss: 0.23 | Elapsed: 64.49s
------------------------------------------------------------

[✔] Model saved successfully as "models/XrayCNNLoad.nn".
------------------------------------------------------------

Test Accuracy: 93.25%.
```
---

## License 

MIT License – see [LICENSE](https://opensource.org/licenses/MIT)

