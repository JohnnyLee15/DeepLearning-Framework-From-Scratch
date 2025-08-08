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

//     GpuEngine::init();

//     const size_t SIZE = 256;
//     const size_t CHANNELS = 1;
//     const string trainPath = "DataFiles/chest_xray/train";
//     const string testPath = "DataFiles/chest_xray/test";

//     // Data Reading
//     ImageData2D *data = new ImageData2D();
//     data->readTrain(trainPath);
//     data->readTest(testPath);

//     ImageTransform2D *transformer = new ImageTransform2D(SIZE, SIZE, CHANNELS);
//     Tensor xTrain = transformer->transform(data->getTrainFeatures());
//     Tensor xTest = transformer->transform(data->getTestFeatures());
//     vector<float> yTrain = data->getTrainTargets();
//     vector<float> yTest = data->getTestTargets();

//     Loss *loss = new SoftmaxCrossEntropy();
//     vector<Layer*> layers = {
//         new Conv2D(16, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(16, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),

//         new Conv2D(32, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(32, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),

//         new Conv2D(64, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(64, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),

//         new Conv2D(128, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(128, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),

//         new Conv2D(256, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(256, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),

//         new Flatten(),
//         new Dense(256, new ReLU()),
//         new Dense(128, new ReLU()),
//         new Dense(2, new Softmax())
//     };

//     NeuralNet *nn = new NeuralNet(layers, loss);
//     ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());

//     nn->fit(
//         xTrain,
//         yTrain,
//         0.0001,
//         0.00001,
//         2,
//         8,
//         *metric
//     );

//     Pipeline pipe;
//     pipe.setData(data);
//     pipe.setModel(nn);
//     pipe.setImageTransformer2D(transformer);
//     pipe.saveToBin("models/SmallXrayClassifierTrain");

//     Tensor output = nn->predict(xTest);
//     vector<float> predictions = TrainingUtils::getPredictions(output);
//     float accuracy = TrainingUtils::getAccuracy(yTest, predictions);
//     printf("\nTest Accuracy %.2f.\n", accuracy);
// }