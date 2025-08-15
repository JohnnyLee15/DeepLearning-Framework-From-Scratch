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
// #include "utils/TrainingUtils.h"
// #include "core/layers/Layer.h"
// #include "core/layers/Dense.h"
// #include "core/layers/Dropout.h"
// #include "core/layers/Flatten.h"
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

//     // Data Reading
//     const string trainPath = "DataFiles/MNIST/mnist_train.csv";
//     const string testPath = "DataFiles/MNIST/mnist_test.csv";
//     const string targetColumn = "label";

//     TabularData *data = new TabularData("classification");
//     data->readTrain(trainPath, targetColumn);
//     data->readTest(testPath, targetColumn);

//     // Feature Scalar
//     Scalar *scalar = new Greyscale();
//     scalar->fit(data->getTrainFeatures());
//     Tensor xTrain = scalar->transform(data->getTrainFeatures());
//     Tensor xTest = scalar->transform(data->getTestFeatures());

//     vector<float> yTrain = data->getTrainTargets();
//     vector<float> yTest = data->getTestTargets();

//     // Defining Model Architecture
//     Loss *loss = new SoftmaxCrossEntropy();
//     vector<Layer*> layers = {
//         new Dense(256, new ReLU()),
//         new Dropout(0.5), 
//         new Dense(128, new ReLU()),
//         new Dense(10, new Softmax())
//     };

//     // Creating Neural Network
//     NeuralNet *nn = new NeuralNet(layers, loss);

//     // Training Model
//     ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());
//     nn->fit(
//         xTrain, // Features
//         yTrain, // Targets
//         0.01,   // Learning rate
//         0.01,   // Learning rate decay
//         2,      // Number of epochs
//         32,     // Batch Size
//         *metric // Progress metric
//     );

//     // Saving Model
//     Pipeline pipe;
//     pipe.setData(data);
//     pipe.setFeatureScalar(scalar);
//     pipe.setModel(nn);
//     pipe.saveToBin("models/ClassMnistTrain.nn");

//     // Testing Model
//     Tensor output = nn->predict(xTest);
//     vector<float> predictions = TrainingUtils::getPredictions(output);
//     float accuracy = 100.0f * TrainingUtils::getAccuracy(yTest, predictions);
//     printf("\nTest Accuracy: %.2f.\n", accuracy);
// }