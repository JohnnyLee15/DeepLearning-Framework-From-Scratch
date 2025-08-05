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
// #include "core/data/ImageData2D.h"
// #include "core/gpu/GpuEngine.h"

// int main() {

//     // Welcome Message
//     ConsoleUtils::printTitle();

//     #ifdef __APPLE__
//         GpuEngine::init();
//     #endif

//     // // Data Reading
//     // TabularData *data = new TabularData("classification");
//     // data->readTrain("DataFiles/MNIST/mnist_train.csv", "label");
//     // data->readTest("DataFiles/MNIST/mnist_test.csv", "label");

//     // Scalar *scalar = new Greyscale();
//     // scalar->fit(data->getTrainFeatures());
    
//     // Tensor xTrain = scalar->transform(data->getTrainFeatures());
//     // const vector<float> &yTrain = data->getTrainTargets();

//     // Loss *loss = new SoftmaxCrossEntropy();
//     // vector<Layer*> layers = {
//     //     new Dense(256, new ReLU()),
//     //     new Dense(128, new ReLU()),
//     //     new Dense(10, new Softmax())
//     // };

//     // NeuralNet *nn = new NeuralNet(layers, loss);
//     // ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());

//     // nn->fit(
//     //     xTrain,
//     //     yTrain,
//     //     0.01,
//     //     0.01,
//     //     2,
//     //     32,
//     //     *metric
//     // );

//     // Pipeline pipe;
//     // pipe.setData(data);
//     // pipe.setFeatureScalar(scalar);
//     // pipe.setModel(nn);

//     // pipe.saveToBin("model");

//     // Tensor xTest = scalar->transform(data->getTestFeatures());
//     // const vector<float> &yTest = data->getTestTargets();


//     Pipeline pipe = Pipeline::loadFromBin("model");
//     NeuralNet *nn = pipe.getModel();
//     TabularData data = *dynamic_cast<TabularData*>(pipe.getData());
//     Scalar *scalar = pipe.getFeatureScalar();
    
//     data.readTest("DataFiles/MNIST/mnist_test.csv", "label");

//     Tensor xTest = scalar->transform(data.getTestFeatures());
//     vector<float> yTest = data.getTestTargets();

//     Tensor output = nn->predict(xTest);
//     vector<float> predictions = TrainingUtils::getPredictions(output);
//     float accuracy = TrainingUtils::getAccuracy(predictions, yTest);
//     cout << "Test Accuracy: " << accuracy << endl;
// }