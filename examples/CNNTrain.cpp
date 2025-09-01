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
#include "core/data/BinLoader.h"

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
    const string dataPath = "DataFiles/chest_xray";

    // Data Reading
    ImageData2D *data = new ImageData2D(CHANNELS);
    data->readTrain(dataPath);

    // Transform data (resize to 224x224 and normalize)
    ImageTransform2D *transformer = new ImageTransform2D(SIZE, SIZE, CHANNELS);
    Tensor x = transformer->transform(data->getTrainFeatures());
    vector<float> y = data->getTrainTargets();

    // Splitting training data into train, test, and validation sets
    Split splitTest = DataSplitter::stratifiedSplit(x, y, 0.2f);
    Split splitVal = DataSplitter::stratifiedSplit(splitTest.xTrain, splitTest.yTrain, 0.1f);

    Tensor xTrain = splitVal.xTrain;
    Tensor xTest = splitTest.xVal;
    Tensor xVal = splitVal.xVal;

    vector<float> yTrain = splitVal.yTrain;
    vector<float> yTest = splitTest.yVal;
    vector<float> yVal = splitVal.yVal;

    // Write splits to .bin for streaming; BinLoader consumes and frees RAM.
    BinLoader train("transformedData/train", xTrain, yTrain);
    BinLoader test("transformedData/test", xTest, yTest);
    BinLoader val("transformedData/val", xVal, yVal);

    // Clearing unused data
    data->clearTrain();
    x.clear();
    y.clear();
    splitTest.clear();
    splitVal.clear();

    // Defining Model Architecture
    Loss *loss = new SoftmaxCrossEntropy();
    vector<Layer*> layers = {
        // Stem: larger RF early
        new Conv2D(32, 7, 7, 2, "same", new ReLU(), 2e-4f),
        new MaxPooling2D(3, 2, 2, "none"),

        // Stage 1 (64 ch)
        new Conv2D(64, 3, 3, 1, "same", new ReLU(), 2e-4f),
        new Conv2D(64, 3, 3, 1, "same", new ReLU(), 2e-4f),

        // Stage 2 (128 ch) — downsample via stride-2 conv
        new Conv2D(128, 3, 3, 2, "same", new ReLU(), 2e-4f),
        new Conv2D(128, 3, 3, 1, "same", new ReLU(), 2e-4f),

        // Stage 3 (256 ch) — downsample
        new Conv2D(256, 3, 3, 2, "same", new ReLU(), 2e-4f),
        new Conv2D(256, 3, 3, 1, "same", new ReLU(), 2e-4f),

        // Optional light Stage 4 (kept at 256 to avoid overfitting)
        new Conv2D(256, 3, 3, 2, "same", new ReLU(), 2e-4f),
        new Conv2D(256, 3, 3, 1, "same", new ReLU(), 2e-4f),

        new GlobalAveragePooling2D(),
        new Dropout(0.35f),
        new Dense(6, new Softmax(), 2e-4f)
    };

    // Creating Neural Network
    NeuralNet *nn = new NeuralNet(layers, loss);

    // Creating Early Stop Object
    EarlyStop *stop = new EarlyStop(8, 5e-4f, 5); // (patience, min delta, warm-up)

    // Training Model
    ProgressMetric *metric = new ProgressAccuracy();
    nn->fit(
        train,  // Train bin loader
        0.005f,  // Learning rate
        0.0f,    // Learning rate decay
        50,      // Number of epochs
        16,     // Batch Size
        *metric, // Progress metric
        val,    // Validation bin loader
        stop    // Early stop object
    );

    // Saving Model
    Pipeline pipe;
    pipe.setData(data);
    pipe.setModel(nn);
    pipe.setImageTransformer2D(transformer);
    pipe.saveToBin("models/XrayCNNTrain");

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