#define STB_IMAGE_IMPLEMENTATION

#include <stb/stb_image.h>
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
    TabularData *data = new TabularData("classification");
    data->readTrain("DataFiles/MNIST/mnist_train.csv", "label");

    Scalar *scalar = new Greyscale();
    scalar->fit(data->getTrainFeatures());

    Tensor xTrain = scalar->transform(data->getTrainFeatures());
    const vector<float> &yTrain = data->getTrainTargets();

    Loss *loss = new SoftmaxCrossEntropy();
    ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());

    vector<Layer*> layers = {
        new DenseLayer(64, new ReLU()),
        new DenseLayer(32, new ReLU()),
        new DenseLayer(10, new ReLU())
    };

    NeuralNet *nn = new NeuralNet(layers, loss);

    nn->fit(
        xTrain,
        yTrain,
        0.01,
        0.01,
        5,
        32,
        *metric
    );
}