#include <fstream>
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

using namespace std;

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Data Reading
    Data *data = new TabularData("classification");
    data->readTrain("DataFiles/MNIST/mnist_train.csv", "label");
    data->readTest("DataFiles/MNIST/mnist_test.csv", "label"); 

    // Scaling Data
    Scalar *featureScalar = new Greyscale();
    featureScalar->fit(data->getTrainFeatures());

    Tensor xTrain = featureScalar->transform(data->getTrainFeatures());
    const vector<double> &yTrain = data->getTrainTargets();

    Tensor xTest =featureScalar->transform(data->getTestFeatures());
    const vector<double> &yTest = data->getTestTargets();

    // Architecture
    size_t numFeatures = data->getTrainFeatures().getShape()[1];
    Loss *loss = new SoftmaxCrossEntropy();
        vector<Layer*> layers = {
        new DenseLayer(64, numFeatures, new ReLU()), 
        new DenseLayer(32, 64, new ReLU()),
        new DenseLayer(10, 32, new Softmax())        
    };

    // Model Creation
    NeuralNet *nn = new NeuralNet(layers, loss);

    // Train
    ProgressMetric *metric = new ProgressAccuracy(data->getNumTrainSamples());

    nn->fit(
        xTrain,
        yTrain,
        0.01,  
        0.01,  
        500,  
        32,
        *metric
    );

    // Save
    Pipeline pipe;
    pipe.setModel(nn);
    pipe.setData(data);
    pipe.setFeatureScalar(featureScalar);
    pipe.saveToBin("model");

    // Test
    Tensor output = nn->predict(xTest);
    vector<double> predictions = TrainingUtils::getPredictions(output);
    double accuracy = TrainingUtils::getAccuracy(yTest, predictions);
    cout << "Test Accuracy: " << accuracy << endl;
}