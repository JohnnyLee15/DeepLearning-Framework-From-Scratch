#include "core/DenseLayer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/VectorUtils.h"
#include "core/MatrixT.h"
#include <random>

const double DenseLayer::HE_INT_GAIN = 2.0;

DenseLayer::DenseLayer(size_t numNeurons, size_t numWeights, Activation *activation) :
    weights(numNeurons, numWeights), activation(activation)
{
    DenseLayer::initWeights(numNeurons, numWeights);
    biases = activation->initBias(numNeurons);
}

vector<uint32_t> DenseLayer::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void DenseLayer::initWeights(size_t numRows, size_t numCols) {
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

void DenseLayer::calActivations(const Matrix&prevActivations) {
    preActivations = prevActivations * weights.T();
    preActivations.addToRows(biases);
    activations = activation->activate(preActivations);
}

const Matrix DenseLayer::getActivations() const {
    return activations;
}

void DenseLayer::backprop(
    const Matrix &prevActivations,
    double learningRate,
    const Matrix &outputGradients,
    bool isFirstLayer
) {
    dZ = outputGradients;
    if (!activation->isFused()) {
        dZ *= activation->calculateGradient(preActivations);
    }

    size_t batchSize = dZ.getNumRows();
    double scaleFactor = -learningRate/batchSize;

    Matrix weightGradients = dZ.T() * prevActivations;
    vector<double> biasGradients = dZ.colSums();

    weightGradients *= scaleFactor;
    VectorUtils::scaleVecInplace(biasGradients, scaleFactor);

    weights += weightGradients;
    VectorUtils::addVecInplace(biases, biasGradients);

    if (!isFirstLayer) {
        dZ = dZ * weights;
    }
}

Matrix DenseLayer::getOutputGradient() const {
    return dZ;
}

DenseLayer::~DenseLayer() {
    delete activation;
}
