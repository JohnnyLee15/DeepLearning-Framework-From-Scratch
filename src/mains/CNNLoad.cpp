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
#include "core/layers/Dropout.h"
#include "core/data/ImageData2D.h"
#include "core/gpu/GpuEngine.h"
#include "utils/DataSplitter.h"

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

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

    return 0;
}
