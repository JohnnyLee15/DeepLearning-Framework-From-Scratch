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
#include "core/RegressionTask.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include "core/ClassificationTask.h"
#include "utils/TrainingUtils.h"
#include "core/Layer.h"
#include "core/DenseLayer.h"
#include "core/Matrix.h"

using namespace std;

int main() {
    // Welcome Message
    ConsoleUtils::printTitle();

    // Data Processing
        TabularData data;
        data.setTask(new ClassificationTask());
        data.readTrain("DataFiles/MNIST/mnist_train.csv", "label");
        data.readTest("DataFiles/MNIST/mnist_test.csv", "label");
        data.setScalars(new Greyscale());                
        data.fitScalars();
        data.transformTrain();
        data.transformTest();
        size_t numFeatures = data.getTrainFeatures().getShape()[1]; 

        // Architecture
        Loss *loss = new SoftmaxCrossEntropy();
            vector<Layer*> layers = {
            new DenseLayer(64, numFeatures, new ReLU()), 
            new DenseLayer(32, 64, new ReLU()),         
            new DenseLayer(10, 32, new Softmax())        
        };

        // Instantiation
        NeuralNet nn(layers, loss);

        // Train
        nn.train(
            data,
            0.01,  
            0.01,  
            3,  
            32     
        );

        // Save
        nn.saveToBin("modelTest.nn", data);

        // Test
        // NeuralNet nn = NeuralNet::loadFromBin("modelTest", data);
        // data.readTest("DataFiles/MNIST/mnist_test.csv", "label");
        // data.transformTest();
        Tensor probs = nn.predict(data);
        vector<double> predictions = TrainingUtils::getPredictions(probs);
        double accuracy = TrainingUtils::getAccuracy(predictions, data.getTestTargets());
        cout << "Test Accuracy: " << accuracy << endl;
}