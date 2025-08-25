// #include <fstream>
// #include <iostream>
// #include <vector>
// #include <sstream>
// #include <cstdio>
// #include "core/model/NeuralNet.h"
// #include "core/data/TabularData.h"
// #include "core/activations/ReLU.h"
// #include "core/activations/Softmax.h"
// #include "core/activations/Linear.h"
// #include "core/losses/MSE.h"
// #include "core/losses/Loss.h"
// #include "core/losses/SoftmaxCrossEntropy.h"
// #include "utils/ConsoleUtils.h"
// #include "utils/Greyscale.h"
// #include "utils/Minmax.h"
// #include "core/layers/Dropout.h"
// #include "core/layers/Flatten.h"
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
// #include "core/data/ImageData2D.h"
// #include "core/gpu/GpuEngine.h"

// int main() {

//     // Welcome Message
//     ConsoleUtils::printTitle();

//     // Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
//     GpuEngine::init();

//     // Data Paths
//     const string trainPath = "DataFiles/MNIST/mnist_train.csv";
//     const string testPath = "DataFiles/MNIST/mnist_test.csv";

//     // Must be in the same position as when the model was trained
//     const string targetColumn = "label";

//     // Loading Model
//     Pipeline pipe = Pipeline::loadFromBin("models/ClassMnistTrain.nn");
//     NeuralNet *nn = pipe.getModel();
//     TabularData *data = dynamic_cast<TabularData*>(pipe.getData());
//     Scalar *scalar = pipe.getFeatureScalar();
    
//     // Data Reading
//     data->readTrain(trainPath, targetColumn);
//     data->readTest(testPath, targetColumn);

//     // Scaling Data
//     Tensor xTrain = scalar->transform(data->getTrainFeatures());
//     Tensor xTest = scalar->transform(data->getTestFeatures());

//     vector<float> yTrain = data->getTrainTargets();
//     vector<float> yTest = data->getTestTargets();

//     // Training Model
//     ProgressMetric *metric = new ProgressAccuracy();
//     nn->fit(
//         xTrain,  // Features
//         yTrain,  // Targets
//         0.01,  // Learning rate
//         0.01,  // Learning rate decay
//         2,       // Number of epochs
//         32,      // Batch Size
//         *metric  // Progress metric
//     );

//     // Saving Model
//     pipe.saveToBin("models/ClassMnistLoad.nn");

//     // Testing Model
//     Tensor output = nn->predict(xTest);
//     vector<float> predictions = TrainingUtils::getPredictions(output);
//     float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
//     printf("\nTest Accuracy: %.2f%%.\n", accuracy);
// }