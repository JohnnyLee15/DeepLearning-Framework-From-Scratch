#include "core/Layer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/MatrixUtils.h"
#include "utils/MatrixT.h"
#include <random>

const double Layer::HE_INT_GAIN = 2.0;

Layer::Layer(int numNeurons, int numWeights, Activation *activation) :
    activation(activation), weights(numNeurons, numWeights)
{
    Layer::initWeights(numNeurons, numWeights);
    biases = activation->initBias(numNeurons);
}

void Layer::initWeights(size_t numRows, size_t numCols) {
    size_t size = numRows * numCols;
    double std = sqrt(HE_INT_GAIN/numCols);
    vector<double> &weightsFlat = weights.getFlat();

    #pragma omp parallel
    {
        random_device rd;
        mt19937 generator(rd() + omp_get_thread_num());
        normal_distribution<double> distribution(0, std);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            weightsFlat[i] = distribution(generator);
        }
    }
}

void Layer::calActivations(const Matrix&prevActivations) {
    preActivations = prevActivations * weights.T();
    preActivations.addToRows(biases);
    activations = activation->activate(preActivations);
}

const Matrix Layer::getActivations() const {
    return activations;
}

const Matrix Layer::getPreActivations() const {
    return preActivations;
}

Activation* Layer::getActivation() const {
    return activation;
}

void Layer::updateLayerParameters(
    const Matrix &prevActivations,
    double learningRate,
    const Matrix &outputGradients
) {
    size_t batchSize = outputGradients.getNumRows();
    double scaleFactor = -learningRate/batchSize;

    Matrix weightGradients = outputGradients.T() * prevActivations;
    vector<double> biasGradients = outputGradients.colSums();

    weightGradients *= scaleFactor;
    MatrixUtils::scaleVecInplace(biasGradients, scaleFactor);

    weights += weightGradients;
    MatrixUtils::addVecInplace(biases, biasGradients);
}


Matrix Layer::updateOutputGradient(
    const Matrix &prevOutputGradients,
    const Matrix &prevPreActivations,
    Activation *prevActivation
) {
    Matrix activationGradients = prevActivation->calculateGradient(prevPreActivations);
    Matrix gradientsMatrix = prevOutputGradients * weights;
    gradientsMatrix *= activationGradients;

    return gradientsMatrix;
}
