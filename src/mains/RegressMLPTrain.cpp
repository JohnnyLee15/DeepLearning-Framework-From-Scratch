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
// #include "core/data/BinLoader.h"

// int main() {

//     // Welcome Message
//     ConsoleUtils::printTitle();

//     // Initialize Gpu if on Mac (Safe to call on non-Mac, it just won't do anything)
//     GpuEngine::init();

//     // Data Reading
//     const string dataPath = "DataFiles/California_Housing/housing_clean.csv";
//     const string targetColumn = "median_house_value";

//     TabularData *data = new TabularData("regression");
//     data->readTrain(dataPath, targetColumn);

//     // Splitting training data into train, test, and validation sets
//     Split splitTest = DataSplitter::stratifiedSplit(
//         data->getTrainFeatures(), data->getTrainTargets(), 0.2f
//     );

//     Split splitVal = DataSplitter::stratifiedSplit(
//         splitTest.xTrain, splitTest.yTrain, 0.1f
//     );

//     // Scaling Features
//     Scalar *featureScalar = new Minmax();
//     featureScalar->fit(splitVal.xTrain);
//     Tensor xTrain = featureScalar->transform(splitVal.xTrain);
//     Tensor xTest = featureScalar->transform(splitTest.xVal);
//     Tensor xVal = featureScalar->transform(splitVal.xVal);


//     // Scaling Targets
//     Scalar *targetScalar = new Minmax();
//     targetScalar->fit(splitVal.yTrain);
//     vector<float> yTrain = targetScalar->transform(splitVal.yTrain);
//     vector<float> yVal = targetScalar->transform(splitVal.yVal);
//     vector<float> yTest = splitTest.yVal;

//     // Write splits to .bin for streaming; BinLoader consumes and frees RAM.
//     BinLoader train("transformedData/train", xTrain, yTrain);
//     BinLoader test("transformedData/test", xTest, yTest);
//     BinLoader val("transformedData/val", xVal, yVal);

//     // Clearing unused data to save memory
//     data->clearTrain();
//     splitTest.clear();
//     splitVal.clear();

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
//     EarlyStop *stop = new EarlyStop(8, 1e-4, 10); // (patience, min delta, warm-up)

//     // Creating Neural Network
//     NeuralNet *nn = new NeuralNet(layers, loss);

//     // Training Model
//     ProgressMetric *metric = new ProgressMAPE();
//     nn->fit(
//         train,
//         0.01,   // Learning rate
//         0.01,  // Learning rate decay
//         200,     // Number of epochs
//         512,      // Batch Size
//         *metric,  // Progress metric
//         val,
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
//     Tensor output = nn->predict(test);
//     Tensor predictions = targetScalar->reverseTransform(output);
//     yTest = test.loadTargets();
//     float rmse = TrainingUtils::getRMSE(predictions, yTest);
//     printf("\nTest RMSE: %.2f.\n", rmse);
// }