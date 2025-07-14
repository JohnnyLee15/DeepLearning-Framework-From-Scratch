#define STB_IMAGE_IMPLEMENTATION

#include "stb/stb_image.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <sstream>
#include "core/NeuralNet.h"
#include "core/TabularData.h"
#include <iomanip>
#include "activations/ReLU.h"
#include "activations/Softmax.h"
#include "activations/Linear.h"
#include "losses/MSE.h"
#include "losses/Loss.h"
#include "losses/SoftmaxCrossEntropy.h"
#include "utils/ConsoleUtils.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include "utils/TrainingUtils.h"
#include "core/Layer.h"
#include "core/DenseLayer.h"
#include "core/Matrix.h"
#include "core/ProgressAccuracy.h"
#include "core/ProgressMAPE.h"
#include "core/Pipeline.h"
#include "utils/ImageTransform2D.h"
#include "core/Conv2D.h"
#include "core/MaxPooling2D.h"
#include "core/Flatten.h"
#include "core/ImageData2D.h"

using namespace std;
namespace fs = filesystem;

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Data Reading
    ImageTransform2D *transformer = new ImageTransform2D(64,64,1);

    string trainPath = "DataFiles/chest_xray/train";
    vector<float> trainFeaturesRaw;
    vector<string> trainTargetsRaw;
    size_t numTrainSamples = 0;

    size_t pneumoniaKeepCount = 0;
    size_t pneumoniaKeepLimit = 2375;

    for (const auto &labelDir : fs::directory_iterator(trainPath)) {
        string label = labelDir.path().filename().string();
        for (const auto &image : fs::directory_iterator(labelDir.path())) {
  
            if (label == "PNEUMONIA") {
                if (pneumoniaKeepCount >= pneumoniaKeepLimit) {
                    continue;  
                }
                pneumoniaKeepCount++;
            }

            string imgPath = image.path().string();

            int w, h, c;
            unsigned char *input = stbi_load(
                imgPath.c_str(),
                &w, &h, &c,
                1
            );

            if (!input) {
                ConsoleUtils::fatalError("Could not load image: " + imgPath);
            }

            vector<float> processed = transformer->transform(input, h, w, c);
            trainFeaturesRaw.insert(trainFeaturesRaw.end(), processed.begin(), processed.end());
            trainTargetsRaw.push_back(label);
            numTrainSamples++;
            stbi_image_free(input);
        }
    }

    cout << "DONE TRAIN" << endl;

    string testPath = "DataFiles/chest_xray/test";
    vector<float> testFeaturesRaw;
    vector<string> testTargetsRaw;
    size_t numTestSamples = 0;
    for (const auto &labelDir : fs::directory_iterator(testPath)) {
        string label = labelDir.path().filename().string();
        for (const auto &image : fs::directory_iterator(labelDir.path())) {
            string imgPath = image.path().string();

            int w, h, c;
            unsigned char *input = stbi_load(
                imgPath.c_str(),
                &w, &h, &c,
                1
            );

            if (!input) {
                ConsoleUtils::fatalError("Could not load image: " + imgPath);
            }

            vector<float>  processed = transformer->transform(input, h, w, c);
            testFeaturesRaw.insert(testFeaturesRaw.end(), processed.begin(), processed.end());
            testTargetsRaw.push_back(label);
            numTestSamples++;
            stbi_image_free(input);
        }
    }

    cout << "DONE TEST" << endl;

    Tensor xTrain = Tensor(trainFeaturesRaw, {numTrainSamples, 64, 64, 1});
    Tensor xTest = Tensor(testFeaturesRaw, {numTestSamples, 64, 64, 1});
    ImageData2D data;

    data.setTrainFeatures(xTrain);
    data.setTestFeatures(xTest);
    data.setTrainTargets(trainTargetsRaw);
    data.setTestTargets(testTargetsRaw);

    const vector<float> &yTrain = data.getTrainTargets();
    const vector<float> &yTest = data.getTestTargets();

    Loss *loss = new SoftmaxCrossEntropy();
    vector<Layer*> layers = {
        new Conv2D(16, 3, 3, 1, "same", new ReLU()),
        new Conv2D(16, 3, 3, 1, "same", new ReLU()),
        new MaxPooling2D(2, 2, 2, "none"),
        new Flatten(),
        new DenseLayer(32, new ReLU()),
        new DenseLayer(2, new Softmax())
    };

    NeuralNet nn(layers, loss);
    ProgressMetric *metric = new ProgressAccuracy(data.getNumTrainSamples());

    nn.fit(
        xTrain,
        yTrain,
        0.001,
        0.00001,
        5,
        32,
        *metric
    );

    Tensor output = nn.predict(xTest);
    vector<float> preds = TrainingUtils::getPredictions(output);
    float accuracy = TrainingUtils::getAccuracy(yTest, preds);
    cout << "Accuracy: " << accuracy << endl;
}