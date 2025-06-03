#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "core/NeuralNet.h"
#include "core/Data.h"
#include <iomanip>
#include "activations/Relu.h"
#include "activations/Softmax.h"
#include "losses/CrossEntropy.h"
#include "utils/ConsoleUtils.h"
#include <omp.h>

using namespace std;

int main() {
    ConsoleUtils::printTitle();
    Data reader;
    reader.readTrain("mnist_train.csv", "LABEL");
    reader.readTest("mnist_test.csv", "label");
    reader.minmax();

    CrossEntropy *loss = new CrossEntropy();
    vector<Activation*> activations = {new Relu(), new Relu(), new Softmax()};

    vector<int> layerSizes = {784, 64, 32, 10};
    NeuralNet nn(layerSizes, activations, loss);

    nn.train(reader, 0.01, 0.01, 20, 32);
    double accuracy = nn.test(reader.getTestFeatures(), reader.getTestTarget());

    cout << endl << "Test Accuracy: " << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;

    return 0;
}