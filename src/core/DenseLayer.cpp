#include "core/DenseLayer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/VectorUtils.h"
#include "core/MatrixT.h"
#include <random>
#include <sstream>

const double DenseLayer::HE_INT_GAIN = 2.0;

DenseLayer::DenseLayer(size_t numNeurons, size_t weightsPerNeuron, Activation *activation) :
    weights(numNeurons, weightsPerNeuron), activation(activation)
{
    DenseLayer::initWeights(numNeurons, weightsPerNeuron);
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

void DenseLayer::writeBin(ofstream& modelBin) const {
    uint32_t layerEncoding = Layer::Encodings::DenseLayer;
    modelBin.write((char*) &layerEncoding, sizeof(uint32_t));

    uint32_t numNeuronsWrite = (uint32_t) weights.getNumRows();
    modelBin.write((char*) &numNeuronsWrite, sizeof(uint32_t));

    uint32_t weightsPerNeuronWrite = (uint32_t) weights.getNumCols();
    modelBin.write((char*) &weightsPerNeuronWrite, sizeof(uint32_t));

    uint32_t activationEncoding = activation->getEncoding();
    modelBin.write((char*) &activationEncoding, sizeof(uint32_t));

    size_t numWeights = weights.getNumRows() * weights.getNumCols();
    modelBin.write((char*) weights.getFlat().data(), numWeights * sizeof(double));
    modelBin.write((char*) biases.data(), biases.size() * sizeof(double));
}

void DenseLayer::loadWeightsAndBiases(ifstream &modelBin) {
    size_t numWeights = weights.getNumRows() * weights.getNumCols();
    modelBin.read((char*) weights.getFlat().data(), numWeights * sizeof(double));
    modelBin.read((char*) biases.data(), biases.size() * sizeof(double));
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
