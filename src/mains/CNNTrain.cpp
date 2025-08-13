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

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
    GpuEngine::init();

    // Image Resize Dims
    const size_t SIZE = 256;

    // Number of channels to read in
    const size_t CHANNELS = 1;

    // Data Paths
    const string trainPath = "DataFiles/chest_xray/train";
    const string testPath = "DataFiles/chest_xray/test";

    // Data Reading
    ImageData2D *data = new ImageData2D();
    data->readTrain(trainPath, CHANNELS);
    data->readTest(testPath, CHANNELS);

    // Transform data (resize to 224x224 and normalize)
    ImageTransform2D *transformer = new ImageTransform2D(SIZE, SIZE, CHANNELS);
    Tensor xTrain = transformer->transform(data->getTrainFeatures());
    Tensor xTest = transformer->transform(data->getTestFeatures());
    vector<float> yTrain = data->getTrainTargets();
    vector<float> yTest = data->getTestTargets();

    // Defining Model Architecture
    Loss *loss = new SoftmaxCrossEntropy();
    vector<Layer*> layers = {
        new Conv2D(64, 3, 3, 2, "same", new ReLU()),
        new MaxPooling2D(2, 2, 2, "none"),

        new Conv2D(128, 3, 3, 1, "same", new ReLU()),
        new Conv2D(128, 3, 3, 1, "same", new ReLU()),
        new MaxPooling2D(2, 2, 2, "none"),          

        new Conv2D(256, 3, 3, 1, "same", new ReLU()),
        new Conv2D(256, 3, 3, 1, "same", new ReLU()),
        new MaxPooling2D(2, 2, 2, "none"),  

        new GlobalAveragePooling2D(),
        new Dense(128, new ReLU(), 0.0001f),
        new Dropout(0.5f),
        new Dense(2, new Softmax())
    };

    // Creating Neural Network
    NeuralNet *nn = new NeuralNet(layers, loss);

    // Training Model
    ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());
    nn->fit(
        xTrain, // Features
        yTrain, // Targets
        0.0015,  // Learning rate
        0.0025,    // Learning rate decay
        50,      // Number of epochs
        8,     // Batch Size
        *metric // Progress metric
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
}