#include "core/Layer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/MatrixUtils.h"
#include <random>

const double Layer::HE_INT_GAIN = 2.0;

Layer::Layer(int numNeurons, int numWeights, Activation *activation) :
    activation(activation),
    weights(numNeurons, vector<double>(numWeights)),
    biases(numNeurons)
{
    Layer::initWeights();
    Layer::initBiases();
}

void Layer::initWeights() {
    int numRows = weights.size();
    int numCols = weights[0].size();

    double std = sqrt(HE_INT_GAIN/numCols);

    #pragma omp parallel for
    for (int i = 0; i < numRows; i++) {
        random_device rd;
        mt19937 generator(rd());
        normal_distribution<double> distribution(0, std);
        for (int j = 0; j < numCols; j++) {
            weights[i][j] = distribution(generator);
        }
    }
}

void Layer::initBiases() {
    int numBiases = biases.size();

    #pragma omp parallel for
    for (int i = 0; i < numBiases; i++) {
        biases[i] = activation->initBias();
    }
}

void Layer::calActivations(const vector<vector<double> >&prevActivations) {
    preActivations = MatrixUtils::multMatMatT(prevActivations, weights);
    int batchSize = preActivations.size();
    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        MatrixUtils::addVecInplace(preActivations[i], biases);
    }

    activations = vector<vector<double>>(batchSize);
    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        activations[i] = activation->activate(preActivations[i]);
    }
}

const vector<vector<double> > Layer::getActivations() const {
    return activations;
}

const vector<vector<double> > Layer::getPreActivations() const {
    return preActivations;
}

Activation* Layer::getActivation() const {
    return activation;
}

void Layer::updateLayerParameters(
    const vector<vector<double> > &prevActivations,
    double learningRate,
    const vector<vector<double> > &outputGradients
) {
    int batchSize = outputGradients.size();
    double scaleFactor = -learningRate/batchSize;

    vector<vector<double> > weightGradients = MatrixUtils::multMatTMat(outputGradients, prevActivations);
    vector<double> biasGradients = MatrixUtils::colSums(outputGradients);
    MatrixUtils::scaleMatInplace(weightGradients, scaleFactor);
    MatrixUtils::scaleVecInplace(biasGradients, scaleFactor);
    MatrixUtils::addMatInplace(weights, weightGradients);
    MatrixUtils::addVecInplace(biases, biasGradients);
}

vector<vector<double> > Layer::getActivationGradientMat(
    const vector<vector<double> >&prevPreActivations,
    Activation* prevActivation
) const {
    int size = prevPreActivations.size();
    vector<vector<double> > activationGradients(size);

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        activationGradients[i] = prevActivation->calculateGradient(prevPreActivations[i]);
    }

    return activationGradients;
}

vector<vector<double> > Layer::updateOutputGradient(
    const vector<vector<double> >&prevOutputGradients,
    const vector<vector<double> >&prevPreActivations,
    Activation *prevActivation
) {
    vector<vector<double> > activationGradients = getActivationGradientMat(prevPreActivations, prevActivation);
    vector<vector<double> > gradientsMatrix = MatrixUtils::multMatMat(prevOutputGradients, weights);
    MatrixUtils::hardamardInplace(gradientsMatrix, activationGradients);

    return gradientsMatrix;
}
