#include "Neuron.h"
#include <cmath>
#include "Layer.h"
#include "Activation.h"

using namespace std;

random_device Neuron::rd;
mt19937 Neuron::generator(Neuron::rd());

Neuron::Neuron(int numWeights, Activation *activation): 
    weights(numWeights, 0.0) {
    initWeights(numWeights);
    bias = activation->initBias();
}

double Neuron::calPreActivation(const vector<double> &prevActivations) {
    double preActivation = bias;
    for (int i = 0; i < weights.size(); i++) {
        preActivation += prevActivations[i] * weights[i];
    }

    return preActivation;
}

void Neuron::initWeights(int numWeights) {
    double std = sqrt(2.0/numWeights);

    normal_distribution<double> distribution(0, std);
    for (int i = 0; i < numWeights; i++) {
        weights[i] = distribution(generator);
    }
}

void Neuron::updateBias(double delta) {
    bias -= delta;
}

int Neuron::getNumWeights() const{
    return weights.size();
}

void Neuron::updateWeights(
    const vector<double> &prevActivations, 
    double learningRate, 
    double outputGradient
) {
    for (int i = 0; i < weights.size(); i++) {
        double deltaWeight = learningRate*outputGradient*prevActivations[i];
        weights[i] -= deltaWeight;
    }
}

double Neuron::getWeight(int idx) const{
    return weights[idx];
}

