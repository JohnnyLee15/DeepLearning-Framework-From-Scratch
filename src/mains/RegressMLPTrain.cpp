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
// #include "utils/DataSplitter.h"
// #include "utils/EarlyStop.h"

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

//     // Splitting training data into train and validation sets
//     Split split = DataSplitter::stratifiedSplit(
//         data->getTrainFeatures(), data->getTrainTargets(), 0.1f
//     );

//     // Scaling Features
//     Scalar *featureScalar = new Minmax();
//     featureScalar->fit(split.xTrain);
//     const Tensor xTrain = featureScalar->transform(split.xTrain);
//     const Tensor xTest = featureScalar->transform(data->getTestFeatures());
//     const Tensor xVal = featureScalar->transform(split.xVal);


//     // Scaling Targets
//     Scalar *targetScalar = new Minmax();
//     targetScalar->fit(split.yTrain);
//     const vector<float> yTrain = targetScalar->transform(split.yTrain);
//     const vector<float> &yVal = targetScalar->transform(split.yVal);
//     const vector<float> &yTest = data->getTestTargets();

//     // Defining Model Architecture
//     Loss *loss = new MSE();
//     vector<Layer*> layers = {
//         new Dense(512, new ReLU()),
//         new Dense(256, new ReLU()),
//         new Dense(128, new ReLU()),
//         new Dense( 64, new ReLU()),
//         new Dense(  1, new Linear())
//     };

//     // Creating Early Stop Object
//     EarlyStop *stop = new EarlyStop(1, 1e-4, 5);

//     // Creating Neural Network
//     NeuralNet *nn = new NeuralNet(layers, loss);

//     // Training Model
//     ProgressMetric *metric = new ProgressMAPE();
//     nn->fit(
//         xTrain,  // Features
//         yTrain,  // Targets
//         0.001,   // Learning rate
//         0.0001,  // Learning rate decay
//         300,     // Number of epochs
//         32,      // Batch Size
//         *metric,  // Progress metric
//         xVal,  // Validation features
//         yVal,   // Validation targets
//         stop    // Early stop object
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