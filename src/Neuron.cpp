#include "Neuron.h"
#include <cmath>
#include "Layer.h"

using namespace std;

random_device Neuron::rd;
mt19937 Neuron::generator(Neuron::rd());
const double Neuron::RELU_BIAS = 0.01;

Neuron::Neuron(int numWeights, bool isOutputNeuron): 
    weights(numWeights, 0.0), isOutputNeuron(isOutputNeuron)
{
    initWeights(numWeights);
    initBias();
}

double Neuron::calActivation(const vector<double> &prevActivations) {
    double activation = bias;
    for (int i = 0; i < weights.size(); i++) {
        activation += prevActivations[i] * weights[i];
    }

    if (!isOutputNeuron) {
        activation = max(0.0, activation);
    }

    return activation;
}

void Neuron::initWeights(int numWeights) {
    double std = sqrt(2.0/numWeights);

    normal_distribution<double> distribution(0, std);
    for (int i = 0; i < numWeights; i++) {
        weights[i] = distribution(generator);
    }
}

void Neuron::initBias() {
    if (isOutputNeuron) {
        bias = 0.0;
    } else {
        bias = RELU_BIAS;
    }
}

void Neuron::updateBias(double delta) {
    bias -= delta;
}

int Neuron::getNumWeights() const{
    return weights.size();
}

void Neuron::updateWeight(int idx, double delta) {
    weights[idx] -= delta;
}

double Neuron::getWeight(int idx) const{
    return weights[idx];
}

