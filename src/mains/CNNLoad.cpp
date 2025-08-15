// #include <fstream>
// #include <iostream>
// #include <vector>
// #include <sstream>
// #include "core/model/NeuralNet.h"
// #include "core/data/TabularData.h"
// #include <iomanip>
// #include "core/activations/ReLU.h"
// #include "core/activations/Softmax.h"
// #include "core/activations/Linear.h"
// #include "core/losses/MSE.h"
// #include "core/losses/Loss.h"
// #include "core/losses/SoftmaxCrossEntropy.h"
// #include "utils/ConsoleUtils.h"
// #include "utils/Greyscale.h"
// #include "utils/Minmax.h"
// #include "utils/TrainingUtils.h"
// #include "core/layers/Layer.h"
// #include "core/layers/Dense.h"
// #include "core/metrics/ProgressAccuracy.h"
// #include "core/metrics/ProgressMAPE.h"
// #include "core/model/Pipeline.h"
// #include "utils/ImageTransform2D.h"
// #include "core/layers/Conv2D.h"
// #include "core/layers/MaxPooling2D.h"
// #include "core/layers/Flatten.h"
// #include "core/layers/Dropout.h"
// #include "core/data/ImageData2D.h"
// #include "core/gpu/GpuEngine.h"

// int main() {
//     // Welcome Message
//     ConsoleUtils::printTitle();

//     // Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
//     GpuEngine::init();

//     // Data paths
//     const string trainPath = "DataFiles/mnist_png/train";
//     const string testPath = "DataFiles/mnist_png/test";

//     // Number of channels to read in
//     const size_t CHANNELS = 1;

//     // Loading Model
//     Pipeline pipe = Pipeline::loadFromBin("models/XrayCNNTrain");
//     ImageData2D data = *dynamic_cast<ImageData2D*>(pipe.getData());
//     ImageTransform2D transformer = *pipe.getImageTransformer();
//     NeuralNet *nn = pipe.getModel();

//     // Data Reading
//     data.readTrain(trainPath, CHANNELS);
//     data.readTest(testPath, CHANNELS);

//     // Transform data (resize to 64x64 and normalize)
//     Tensor xTrain = transformer.transform(data.getTrainFeatures());
//     Tensor xTest = transformer.transform(data.getTestFeatures());
//     vector<float> yTrain = data.getTrainTargets();
//     vector<float> yTest = data.getTestTargets();

//     // Training Data
//     ProgressMetric *metric = new ProgressAccuracy();
//     nn->fit(
//         xTrain, // Features
//         yTrain, // Targets
//         0.01,  // Learning rate
//         0.2,    // Learning rate decay
//         2,      // Number of epochs
//         32,     // Batch Size
//         *metric // Progress metric
//     );

//     // Save Model
//     pipe.saveToBin("models/XrayCNNLoad");

//     // Testing Model
//     Tensor output = nn->predict(xTest);
//     vector<float> predictions = TrainingUtils::getPredictions(output);
//     float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
//     printf("\nTest Accuracy %.2f.\n", accuracy);

//     return 0;
// }
