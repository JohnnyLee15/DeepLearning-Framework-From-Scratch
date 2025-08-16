#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "core/model/NeuralNet.h"
#include "core/data/TabularData.h"
#include <iomanip>
#include "core/activations/ReLU.h"
#include "core/activations/Softmax.h"
#include "core/activations/Linear.h"
#include "core/losses/MSE.h"
#include "core/losses/Loss.h"
#include "core/losses/SoftmaxCrossEntropy.h"
#include "utils/ConsoleUtils.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include "utils/TrainingUtils.h"
#include "core/layers/Layer.h"
#include "core/layers/Dense.h"
#include "core/metrics/ProgressAccuracy.h"
#include "core/metrics/ProgressMAPE.h"
#include "core/model/Pipeline.h"
#include "utils/ImageTransform2D.h"
#include "core/layers/Conv2D.h"
#include "core/layers/MaxPooling2D.h"
#include "core/layers/Flatten.h"
#include "core/layers/GlobalAveragePooling2D.h"
#include "core/layers/Dropout.h"
#include "core/data/ImageData2D.h"
#include "core/gpu/GpuEngine.h"
#include "utils/EarlyStop.h"
#include "utils/DataSplitter.h"

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
    GpuEngine::init();

    // Image Resize Dims
    const size_t SIZE = 224;

    // Number of channels to read in
    const size_t CHANNELS = 1;

    // Data Paths
    const string dataPath = "DataFiles/chest_xray/data";

    // Data Reading
    ImageData2D *data = new ImageData2D();
    data->readTrain(dataPath, CHANNELS);

    // Transform data (resize to 224x224 and normalize)
    ImageTransform2D *transformer = new ImageTransform2D(SIZE, SIZE, CHANNELS);
    Tensor x = transformer->transform(data->getTrainFeatures());
    vector<float> y = data->getTrainTargets();

    // Splitting Data into training and testing set
    Split splitTest = DataSplitter::stratifiedSplit(x, y, 0.2f);
    const Tensor &xTest = splitTest.xVal;
    const vector<float> &yTest = splitTest.yVal;

    // Splitting training set into train and validation set
    Split splitVal = DataSplitter::stratifiedSplit(splitTest.xTrain, splitTest.yTrain, 0.1f);
    const Tensor &xTrain = splitVal.xTrain;
    const Tensor &xVal = splitVal.xVal;
    const vector<float> &yTrain= splitVal.yTrain;
    const vector<float> &yVal = splitVal.yVal;

    // Defining Model Architecture
    Loss *loss = new SoftmaxCrossEntropy();
    vector<Layer*> layers = {
        new Conv2D(32, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new Conv2D(32, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new MaxPooling2D(2, 2, 2, "none"),

        new Conv2D(64, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new Conv2D(64, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new MaxPooling2D(2, 2, 2, "none"),

        new Conv2D(128, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new Conv2D(128, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new MaxPooling2D(2, 2, 2, "none"),

        new Conv2D(256, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new Conv2D(256, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new MaxPooling2D(2, 2, 2, "none"),

        new Conv2D(512, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new Conv2D(512, 3, 3, 1, "same", new ReLU(), 5e-4f),
        new MaxPooling2D(2, 2, 2, "none"),

        new Flatten(),             
        new Dense(128, new ReLU(), 5e-4f),
        new Dropout(0.5f),
        new Dense(64, new ReLU(), 5e-4f),
        new Dropout(0.5f),
        new Dense(2, new Softmax())               
    };

    // Creating Neural Network
    NeuralNet *nn = new NeuralNet(layers, loss);

    // Creating Early Stop Object
    EarlyStop *stop = new EarlyStop(8, 5e-4f, 5);

    // Training Model
    ProgressMetric *metric = new ProgressAccuracy();
    nn->fit(
        xTrain, // Features
        yTrain, // Targets
        0.005f,  // Learning rate
        0.02f,    // Learning rate decay
        50,      // Number of epochs
        16,     // Batch Size
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
    printf("\nTest Accuracy %.2f.\n", accuracy);

    // Delete pointers that don't belong to pipe
    delete stop;
    delete metric;
}