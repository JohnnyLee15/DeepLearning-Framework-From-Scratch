#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "core/NeuralNet.h"
#include "core/Data.h"
#include <iomanip>
#include "activations/ReLU.h"
#include "activations/Softmax.h"
#include "activations/Linear.h"
#include "losses/MSE.h"
#include "losses/Loss.h"
#include "losses/SoftmaxCrossEntropy.h"
#include "utils/ConsoleUtils.h"
#include "core/RegressionTask.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include "core/ClassificationTask.h"
#include "utils/TrainingUtils.h"
#include "core/Layer.h"
#include "core/DenseLayer.h"

using namespace std;

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Data Processing
    Data data;
    data.setTask(new ClassificationTask());
    data.readTrain("DataFiles/mnist_train.csv", "label");
    data.readTest("DataFiles/mnist_test.csv", "label");
    data.setScalars(new Greyscale());                // setScalars(featureScalar, targetScalar); only featureScalar used for classification           
    data.fitScalars();
    size_t numFeatures = data.getTrainFeatures().getNumCols();

    // Architecture
    Loss *loss = new SoftmaxCrossEntropy();
    vector<Layer*> layers = {
        new DenseLayer(64, numFeatures, new ReLU()), // Hidden layer 1: 64 neurons
        new DenseLayer(32, 64, new ReLU()),          // Hidden layer 2: 32 neurons
        new DenseLayer(10, 32, new Softmax())        // Output layer: 10 classes
    };

    // Instantiation
    NeuralNet nn(layers, loss);

    // Train
    nn.train(
        data,
        0.01,  // learningRate
        0.01,  // decayRate
        3,     // epochs
        32     // batchSize
    );

    // Save
    nn.saveToBin("model");

    // Test
    Matrix probs = nn.predict(data);
    vector<double> predictions = TrainingUtils::getPredictions(probs);
    double accuracy = TrainingUtils::getAccuracy(predictions, data.getTestTargets());
    cout << "Test Accuracy: " << accuracy << endl;
    
    return 0;
}