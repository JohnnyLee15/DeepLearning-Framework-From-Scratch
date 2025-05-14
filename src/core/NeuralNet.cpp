#include "core/NeuralNet.h"
#include "core/Data.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils/TrainingUtils.h"
#include "utils/ConsoleUtils.h"
#include "losses/CrossEntropy.h"

using namespace std;

NeuralNet::NeuralNet(
    const vector<int> &neuronsPerLayer,
    const vector<Activation*> &activations,
    CrossEntropy *loss
) : loss(loss) {
    for (int i = 1; i < neuronsPerLayer.size() - 1; i++) {
        layers.push_back(new Layer(neuronsPerLayer[i], neuronsPerLayer[i-1], activations[i-1]));
    }
    layers.push_back(new Layer(
        neuronsPerLayer[neuronsPerLayer.size() - 1], 
        neuronsPerLayer[neuronsPerLayer.size() - 2], 
        activations.back()
    ));
}

void NeuralNet::train(
    Data data,
    double learningRate,
    double learningDecay,
    int numEpochs
) {
    double initialLR = learningRate;
    for (int k = 0; k < numEpochs; k++) {
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;

        vector<int> predictions(data.getTrainFeatureSize(), -1);
        vector<int> shuffledIndices = data.generateShuffledIndices();
        double avgLoss = runEpoch(data, learningRate, predictions, shuffledIndices);

        double accuracy = TrainingUtils::getAccuracy(data.getTrainTarget(), predictions);
        ConsoleUtils::reportEpochProgress(k+1, numEpochs, avgLoss, accuracy);
        avgLosses.push_back(avgLoss);
        learningRate = initialLR/(1 + learningDecay*k);
    }
}

double NeuralNet::runEpoch(
    Data data,
    double learningRate,
    vector<int> &predictions,
    const vector<int> &shuffledIndices
) {
    double totalLoss = 0.0;
    int numSamples = data.getTrainFeatureSize();
    const vector<vector<double> > &train = data.getTrainFeatures();
    const vector<double> &labels = data.getTrainTarget();

    for (int i = 0; i < numSamples; i++) {
        int rdIdx = shuffledIndices[i];

        forwardPass(train[rdIdx]);
        predictions[rdIdx] = TrainingUtils::getPrediction(layers.back()->getActivations());

        totalLoss += loss->calculateLoss(labels[rdIdx], layers.back()->getActivations());
        backprop(labels[rdIdx], learningRate, train[rdIdx]);
        
        ConsoleUtils::printProgressBar(i, numSamples);
    }

    return totalLoss/train.size();
}

void NeuralNet::forwardPass(const vector<double>& inputData) {
    vector<double> prevActivations = inputData;
    for (int j = 0; j < layers.size(); j++) {
        layers[j]->calActivations(prevActivations);
        prevActivations = layers[j]->getActivations();
    }
}

void NeuralNet::backprop(int label, double learningRate, const vector<double> &inputData) {
    vector<double> outputGradient = loss->calculateGradient(label, layers.back()->getActivations());
    for (int i = layers.size() - 1; i >= 0; i--) {
        vector<double> prevActivations = getPrevActivations(i, inputData);
        layers[i]->updateLayerParameters(prevActivations, learningRate, outputGradient);

        if (i > 0) {
            Activation *prevActivation = layers[i - 1]->getActivation();
            outputGradient = layers[i]->updateOutputGradient(outputGradient, prevActivations, prevActivation);
        }
    }
}

vector<double> NeuralNet::getPrevActivations(int layerIdx, const vector<double> &inputData) const {
    vector<double> prevActivations = inputData;
    if (layerIdx != 0) {
        prevActivations = layers[layerIdx-1]->getActivations();
    }
    return prevActivations;
}

double NeuralNet::test(const vector<vector<double> > &data, const vector<double> &labels) {
    vector<int> predictions(data.size(), -1);
    for (int i = 0; i < data.size(); i++) {
        forwardPass(data[i]);
        predictions[i] = TrainingUtils::getPrediction(layers.back()->getActivations());
    }
    return TrainingUtils::getAccuracy(labels, predictions);
}

NeuralNet::~NeuralNet() {
    delete loss;
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}