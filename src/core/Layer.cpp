#include "core/Layer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/VectorUtils.h"
#include "utils/MatrixT.h"
#include <random>

const double Layer::HE_INT_GAIN = 2.0;

Layer::Layer(int numNeurons, int numWeights, Activation *activation) :
    activation(activation), weights(numNeurons, numWeights)
{
    Layer::initWeights(numNeurons, numWeights);
    biases = activation->initBias(numNeurons);
}

vector<uint32_t> Layer::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Layer::initWeights(size_t numRows, size_t numCols) {
    size_t size = numRows * numCols;
    double std = sqrt(HE_INT_GAIN/numCols);
    vector<double> &weightsFlat = weights.getFlat();

    vector<uint32_t> seeds = generateThreadSeeds();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
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
    VectorUtils::scaleVecInplace(biasGradients, scaleFactor);

    weights += weightGradients;
    VectorUtils::addVecInplace(biases, biasGradients);
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
