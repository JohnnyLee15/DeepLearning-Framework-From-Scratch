#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "NeuralNet.h"
#include "Data.h"
#include <iomanip>
#include "Relu.h"
#include "Softmax.h"
#include "CrossEntropy.h"

using namespace std;

int main() {
    Data reader;
    reader.readTrain("mnist_train.csv", 0);
    reader.readTest("mnist_test.csv", 0);
    reader.minmax();

    CrossEntropy *loss = new CrossEntropy();
    vector<Activation*> activations = {new Relu(), new Relu(), new Softmax()};
    vector<int> layerSizes = {784, 16, 8, 10};

    NeuralNet nn(layerSizes, activations, loss);

    nn.train(reader, 0.0004, 0.02, 30);
    double accuracy = nn.test(reader.getTestFeatures(), reader.getTestTarget());

    cout << endl << "Test Accuracy: " << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;

    return 0;
}

