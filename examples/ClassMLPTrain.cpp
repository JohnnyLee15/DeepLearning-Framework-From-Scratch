#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdio>
#include "core/model/NeuralNet.h"
#include "core/data/TabularData.h"
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
#include "core/layers/Dropout.h"
#include "core/layers/Flatten.h"
#include "core/metrics/ProgressAccuracy.h"
#include "core/metrics/ProgressMAPE.h"
#include "core/model/Pipeline.h"
#include "utils/ImageTransform2D.h"
#include "core/layers/Conv2D.h"
#include "core/layers/MaxPooling2D.h"
#include "core/layers/Flatten.h"
#include "core/data/ImageData2D.h"
#include "core/gpu/GpuEngine.h"
#include "utils/DataSplitter.h"
#include "utils/EarlyStop.h"
#include "core/data/BinLoader.h"

int main() {

    // Welcome Message
    ConsoleUtils::printTitle();

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
    Tensor xTrain = scalar->transform(split.xTrain);
    Tensor xTest = scalar->transform(data->getTestFeatures());
    Tensor xVal = scalar->transform(split.xVal);

    vector<float> yTrain = split.yTrain;
    vector<float> yTest = data->getTestTargets();
    vector<float> yVal = split.yVal;

    // Write splits to .bin for streaming; BinLoader consumes and frees RAM.
    BinLoader train("transformedData/train", xTrain, yTrain);
    BinLoader test("transformedData/test", xTest, yTest);
    BinLoader val("transformedData/val", xVal, yVal);

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
        train,
        0.01,   // Learning rate
        0.01,   // Learning rate decay
        50,      // Number of epochs
        32,     // Batch Size
        *metric,
        val,
        stop
    );

    // Saving Model
    Pipeline pipe;
    pipe.setData(data);
    pipe.setFeatureScalar(scalar);
    pipe.setModel(nn);
    pipe.saveToBin("models/ClassMnistTrain.nn");

    // Testing Model
    Tensor output = nn->predict(test);
    vector<float> predictions = TrainingUtils::getPredictions(output);
    yTest = test.loadTargets();
    float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
    printf("\nTest Accuracy: %.2f%%.\n", accuracy);

    // Delete pointers that don't belong to pipe
    delete stop;
    delete metric;
}