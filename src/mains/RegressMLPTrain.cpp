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
// #include "core/metrics/ProgressAccuracy.h"
// #include "core/metrics/ProgressMAPE.h"
// #include "core/layers/Dropout.h"
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
//     const string trainPath = "DataFiles/California_Housing/housing_clean.csv";
//     const string testPath = "DataFiles/California_Housing/housing_clean.csv";
//     const string targetColumn = "median_house_value";

//     TabularData *data = new TabularData("regression");
//     data->readTrain(trainPath, targetColumn);
//     data->readTest(testPath, targetColumn);

//     // Scaling Training and Testing Features
//     Scalar *featureScalar = new Minmax();
//     featureScalar->fit(data->getTrainFeatures());
    
//     Tensor xTrain = featureScalar->transform(data->getTrainFeatures());
//     Tensor xTest = featureScalar->transform(data->getTestFeatures());

//     // Scaling Training Targets
//     Scalar *targetScalar = new Minmax();
//     targetScalar->fit(data->getTrainTargets());

//     vector<float> yTrain = targetScalar->transform(data->getTrainTargets());
//     vector<float> yTest = data->getTestTargets();

//     // Defining Model Architecture
//     Loss *loss = new MSE();
//     vector<Layer*> layers = {
//     new Dense(512, new ReLU()),
//     new Dense(256, new ReLU()),
//     new Dense(128, new ReLU()),
//     new Dense( 64, new ReLU()),
//     new Dense(  1, new Linear())
//     };

//     // Creating Neural Network
//     NeuralNet *nn = new NeuralNet(layers, loss);

//     // Training Model
//     ProgressMetric *metric = new ProgressMAPE(data->getNumTrainSamples());
//     nn->fit(
//         xTrain,  // Features
//         yTrain,  // Targets
//         0.001,   // Learning rate
//         0.0001,  // Learning rate decay
//         300,     // Number of epochs
//         32,      // Batch Size
//         *metric  // Progress metric
//     );

//     // Saving Model
//     Pipeline pipe;
//     pipe.setData(data);
//     pipe.setFeatureScalar(featureScalar);
//     pipe.setTargetScalar(targetScalar);
//     pipe.setModel(nn);
//     pipe.saveToBin("models/RegressHousingTrain");

//     // Testing Model
//     Tensor output = nn->predict(xTest);
//     Tensor predictions = targetScalar->reverseTransform(output);
//     float rmse = TrainingUtils::getRMSE(predictions, yTest);
//     printf("\nTest RMSE: %.2f.\n", rmse);
// }