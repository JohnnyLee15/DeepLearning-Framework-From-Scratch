#include "NeuralNet.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace std;

const double NeuralNet::GRADIENT_THRESHOLD = 1.0;
const double NeuralNet::LOSS_EPSILON = 1e-10;
const int NeuralNet::PROGRESS_BAR_LENGTH = 50;

NeuralNet::NeuralNet(vector<int> neuronsPerLayer):
    outputActivations(neuronsPerLayer[neuronsPerLayer.size() - 1], 0) {
    for (int i = 1; i < neuronsPerLayer.size() - 1; i++) {
        layers.push_back(Layer(neuronsPerLayer[i], neuronsPerLayer[i-1]));
    }
    layers.push_back(Layer(
        neuronsPerLayer[neuronsPerLayer.size() - 1], 
        neuronsPerLayer[neuronsPerLayer.size() - 2], 
        true
    ));
}

vector<int> NeuralNet::generateIndices(const vector<vector<double> > &data) const {
    vector<int> indices(data.size(), -1);

    for (int i = 0; i < data.size(); i++) {
        indices[i] = i;
    }
    return indices;
}

void NeuralNet::shuffleTrainIndices(vector<int> &indices) const {
    random_device rd;
    mt19937 generator(rd());
    shuffle(indices.begin(), indices.end(), generator);
}

void NeuralNet::train(
    const vector<vector<double> > &data,
    const vector<double> &labels,
    double learningRate,
    double learningDecay,
    int numEpochs
) {
    vector<int> shuffledIndices = generateIndices(data);
    double initialLR = learningRate;
    for (int k = 0; k < numEpochs; k++) {
        vector<int> predictions(data.size(), -1);
        shuffleTrainIndices(shuffledIndices);
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;
        double avgLoss = runEpoch(data, labels, learningRate, predictions, shuffledIndices);
        double accuracy = getAccurary(labels, predictions);
        reportEpochProgress(k+1, numEpochs, avgLoss, accuracy);
        avgLosses.push_back(avgLoss);
        learningRate = initialLR/(1 + learningDecay*k);
    }
}

void NeuralNet::reportEpochProgress(int epoch, int numEpochs, double avgLoss, double accuracy) const {
    cout << endl << "Average Loss: " << avgLoss << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << (100 * accuracy) << "%" << endl;
    cout << defaultfloat << setprecision(6);
}

void NeuralNet::printProgressBar(int currentSample, int totalSamples) const {
    double progress = (double) currentSample / totalSamples;
    int progressChar = (int) (progress * PROGRESS_BAR_LENGTH);

    const string GREEN = "\033[32m";
    const string RESET = "\033[0m";

    cout << "|";
    for (int i = 0; i < PROGRESS_BAR_LENGTH; i++) {
        if (i <= progressChar) {
            cout << GREEN << "█" << RESET;
        } else {
            cout << " ";
        }
    }

    cout << "| " << fixed << setprecision(2) << (progress * 100) <<"%\r";
    cout << defaultfloat << setprecision(6);
    cout.flush();
}

double NeuralNet::runEpoch(
    const vector<vector<double> >&data,
    const vector<double> &labels,
    double learningRate,
    vector<int> &predictions,
    const vector<int> &shuffledIndices
) {
    double totalLoss = 0.0;
    int numSamples = data.size();
    for (int i = 0; i < numSamples; i++) {
        int rdIdx = shuffledIndices[i];
        forwardPass(data[rdIdx]);
        applySoftmax();
        predictions[rdIdx] = getPrediction();
        totalLoss += calculateLoss(labels[rdIdx]);
        backprop(labels[rdIdx], learningRate, data[rdIdx]);
        printProgressBar(i, numSamples);
    }
    return totalLoss/data.size();
}

void NeuralNet::forwardPass(const vector<double>& inputData) {
    vector<double> prevActivations = inputData;
    for (int j = 0; j < layers.size(); j++) {
        layers[j].calActivations(prevActivations);
        prevActivations = layers[j].getActivations();
    }
}

void NeuralNet::applySoftmax() {
    vector<double> outputLayer = layers[layers.size() - 1].getActivations();
    double totalSum = 0;
    double maxAct = layers[layers.size() - 1].getMaxActivation();
    for (int i = 0; i <  outputLayer.size(); i++) {
        totalSum += exp(outputLayer[i] - maxAct);
    }

    for (int i = 0; i < outputLayer.size(); i++) { 
        outputActivations[i] = exp(outputLayer[i] - maxAct)/totalSum;
    }
}

double NeuralNet::calculateLoss(int label) {
    return(-log(max(LOSS_EPSILON, outputActivations[label])));
}

void NeuralNet::backprop(int label, double learningRate, const vector<double> &inputData) {
    vector<double> outputGradient = calculateOutputGradient(label);
    for (int i = layers.size() - 1; i >= 0; i--) {
        vector<double> prevActivations = getPrevActivations(i, inputData);
        updateLayerParameters(layers[i], prevActivations, learningRate, outputGradient);

        if (i > 0) {
            outputGradient = updateOutputGradient(layers[i], outputGradient, prevActivations);
        }
    }
}

vector<double> NeuralNet::getPrevActivations(int layerIdx, const vector<double> &inputData) const {
    vector<double> prevActivations = inputData;
    if (layerIdx != 0) {
        prevActivations = layers[layerIdx-1].getActivations();
    }
    return prevActivations;
}

vector<double> NeuralNet::calculateOutputGradient(int label) {
    vector<double> gradient(outputActivations.size(), 0.0);
    for (int i = 0; i < outputActivations.size(); i++) {
        if (i == label) {
            gradient[i] = clipDerivative(outputActivations[i] - 1);
        } else {
            gradient[i] = clipDerivative(outputActivations[i]);
        }
    }
    return gradient;
}

void NeuralNet::updateLayerParameters(
    Layer &layer,
    const vector<double> &prevActivations,
    double learningRate,
    const vector<double> &outputGradient) {
    for (int i = 0; i < layer.getNumNeurons(); i++) {
        double deltaBias = outputGradient[i]*learningRate;
        layer.updateNeuronBias(deltaBias, i);

        for (int j = 0; j < layer.getNumNeuronWeights(i); j++) {
            double deltaWeight = learningRate*outputGradient[i]*prevActivations[j];
            layer.updateNeuronWeight(i, j, deltaWeight);
        }
    }
}

vector<double> NeuralNet::updateOutputGradient(
    const Layer &layer,
    const vector<double> &prevOutputGradient,
    const vector<double> &prevActivations
) {

    vector<double> outputGradient(prevActivations.size(), 0.0);
    for (int j = 0; j < prevActivations.size(); j++) {
        if (prevActivations[j] <= 0) {
            outputGradient[j] = 0;
        } else {
            outputGradient[j] = updateOutputDerivative(layer, prevOutputGradient, j);
        }
    }
    return outputGradient;
}

double NeuralNet::updateOutputDerivative(
    const Layer &layer,
    const vector<double> &prevOutputGradient,
    int weightIdx
) {

    double dz = 0;
    for (int i = 0; i < layer.getNumNeurons(); i++) {
        dz += prevOutputGradient[i]*layer.getNeuronWeight(i, weightIdx);
    }
    return clipDerivative(dz);
}

double NeuralNet::test(const vector<vector<double> > &data, const vector<double> &labels) {
    vector<int> predictions(data.size(), -1);
    for (int i = 0; i < data.size(); i++) {
        forwardPass(data[i]);
        applySoftmax();
        predictions[i] = getPrediction();
    }
    return getAccurary(labels, predictions);
}

double NeuralNet::getAccurary(const vector<double> &labels, const vector<int> &predictions) const{
    double correct = 0;
    for (int i = 0; i < labels.size(); i++) {
        int label = int (labels[i]);
        if (label == predictions[i]) {
            correct++;
        }
    }
    return correct/predictions.size();
}

int NeuralNet::getPrediction() {
    int prediction = -1;
    double prob = -1;
    for (int i = 0; i < outputActivations.size(); i++) { 
        if (outputActivations[i] > prob) {
            prediction = i;
            prob = outputActivations[i];
        }
    }
    return prediction;
}

double NeuralNet::clipDerivative(double gradient) {
    double clip = 0.0;
    if (!isnan(gradient)) {
        clip = max(-GRADIENT_THRESHOLD, min(GRADIENT_THRESHOLD, gradient));
    }
    return clip;
}