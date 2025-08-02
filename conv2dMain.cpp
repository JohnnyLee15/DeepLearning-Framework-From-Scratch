// #define STB_IMAGE_IMPLEMENTATION

// #include <stb/stb_image.h>
// #include <fstream>
// #include <filesystem>
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

// using namespace std;
// namespace fs = filesystem;


// int main() {
//     // Welcome Message
//     ConsoleUtils::printTitle();

//     // Data Reading
//     ImageTransform2D *transformer = new ImageTransform2D(64,64,1);

//     string trainPath = "DataFiles/chest_xray/train";
//     vector<float> trainFeaturesRaw;
//     vector<string> trainTargetsRaw;
//     size_t numTrainSamples = 0;
//     size_t pneumoniaKeepCount = 0;
//     size_t pneumoniaKeepLimit = 2375; // Example: 3875 - 1500 = 2375

//     for (const auto &labelDir : fs::directory_iterator(trainPath)) {
//         string label = labelDir.path().filename().string();
//         for (const auto &image : fs::directory_iterator(labelDir.path())) {
//             // 🟢 If it’s PNEUMONIA, maybe skip
//             if (label == "PNEUMONIA") {
//                 if (pneumoniaKeepCount >= pneumoniaKeepLimit) {
//                     continue;  // Skip this extra pneumonia image
//                 }
//                 pneumoniaKeepCount++;
//             }

//             string imgPath = image.path().string();

//             int w, h, c;
//             unsigned char *input = stbi_load(
//                 imgPath.c_str(),
//                 &w, &h, &c,
//                 1
//             );

//             if (!input) {
//                 ConsoleUtils::fatalError("Could not load image: " + imgPath);
//             }

//             vector<float> processed = transformer->transform(input, h, w, c);
//             trainFeaturesRaw.insert(trainFeaturesRaw.end(), processed.begin(), processed.end());
//             trainTargetsRaw.push_back(label);
//             numTrainSamples++;
//             stbi_image_free(input);
//         }
//     }

//     cout << "DONE TRAIN" << endl;

//     string testPath = "DataFiles/chest_xray/test";
//     vector<float> testFeaturesRaw;
//     vector<string> testTargetsRaw;
//     size_t numTestSamples = 0;
//     for (const auto &labelDir : fs::directory_iterator(testPath)) {
//         string label = labelDir.path().filename().string();
//         for (const auto &image : fs::directory_iterator(labelDir.path())) {
//             string imgPath = image.path().string();

//             int w, h, c;
//             unsigned char *input = stbi_load(
//                 imgPath.c_str(),
//                 &w, &h, &c,
//                 1
//             );

//             if (!input) {
//                 ConsoleUtils::fatalError("Could not load image: " + imgPath);
//             }

//             vector<float>  processed = transformer->transform(input, h, w, c);
//             testFeaturesRaw.insert(testFeaturesRaw.end(), processed.begin(), processed.end());
//             testTargetsRaw.push_back(label);
//             numTestSamples++;
//             stbi_image_free(input);
//         }
//     }

//     cout << "DONE TEST" << endl;

//     Tensor xTrain = Tensor(trainFeaturesRaw, {numTrainSamples, 64, 64, 1});
//     Tensor xTest = Tensor(testFeaturesRaw, {numTestSamples, 64, 64, 1});
//     ImageData2D data;

//     data.setTrainFeatures(xTrain);
//     data.setTestFeatures(xTest);
//     data.setTrainTargets(trainTargetsRaw);
//     data.setTestTargets(testTargetsRaw);

//     const vector<float> &yTrain = data.getTrainTargets();
//     const vector<float> &yTest = data.getTestTargets();

//     Loss *loss = new SoftmaxCrossEntropy();
//     vector<Layer*> layers = {
//         new Conv2D(16, 3, 3, 1, "same", new ReLU()),
//         new Conv2D(16, 3, 3, 1, "same", new ReLU()),
//         new MaxPooling2D(2, 2, 2, "none"),
//         new Flatten(),
//         new Dense(32, new ReLU()),
//         new Dense(2, new Softmax())
//     };

//     NeuralNet nn(layers, loss);
//     ProgressMetric *metric = new ProgressAccuracy(data.getNumTrainSamples());

//     nn.fit(
//         xTrain,
//         yTrain,
//         0.001,
//         0.00001,
//         15,
//         32,
//         *metric
//     );


//     // Tensor output = nn->predict(xTest);
//     // vector<float> predictions = TrainingUtils::getPredictions(output);
//     // float accuracy = TrainingUtils::getAccuracy(predictions, yTest);
//     // cout << "Test Accuracy: " << accuracy << endl;
// }