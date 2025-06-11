#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "core/NeuralNet.h"
#include "core/Data.h"
#include <iomanip>
#include "activations/Relu.h"
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

using namespace std;

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Data Processing
    Data data;
    data.setTask(new ClassificationTask());
    data.readTrain("DataFiles/mnist_train.csv", "label");
    data.readTest("DataFiles/mnist_test.csv", "label");
    data.setScalars(new Greyscale());
    data.fitScalars();
    size_t numFeatures = data.getTrainFeatures().getNumCols();

    // Architecture
    Loss *loss = new SoftmaxCrossEntropy();
    vector<Activation*> activations = {new Relu(), new Relu(), new Softmax()};
    vector<size_t> layerSizes = {numFeatures, 64, 32, 10};
    NeuralNet nn(layerSizes, activations, loss);

    // Train
    nn.train(data, 0.01, 0.01, 50, 32);
    Matrix probs = nn.predict(data);
    vector<double> predictions = TrainingUtils::getPredictions(probs);
    double accuracy = TrainingUtils::getAccuracy(predictions, data.getTestTargets());
    cout << "Test Accuracy: " << accuracy << endl;

    return 0;
}