#include "core/DenseLayer.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "utils/VectorUtils.h"
#include "core/MatrixT.h"
#include <random>
#include <sstream>
#include "core/Matrix.h"
#include "activations/Linear.h"
#include "activations/ReLU.h"
#include "activations/Softmax.h"
#include "utils/ConsoleUtils.h"

const double DenseLayer::HE_INT_GAIN = 2.0;

DenseLayer::DenseLayer(size_t numNeurons, size_t weightsPerNeuron, Activation *activation) :
    weights({numNeurons, weightsPerNeuron}), activation(activation)
{
    initWeights(numNeurons, weightsPerNeuron);
    biases = activation->initBias(numNeurons);
}

DenseLayer::DenseLayer() : activation(nullptr) {}

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
    Layer::writeBin(modelBin);

    uint32_t activationEncoding = activation->getEncoding();
    modelBin.write((char*) &activationEncoding, sizeof(uint32_t));

    Matrix weightsMat = weights.M();
    uint32_t numNeuronsWrite = (uint32_t) weightsMat.getNumRows();
    modelBin.write((char*) &numNeuronsWrite, sizeof(uint32_t));

    uint32_t weightsPerNeuronWrite = (uint32_t) weightsMat.getNumCols();
    modelBin.write((char*) &weightsPerNeuronWrite, sizeof(uint32_t));

    size_t numWeights = weightsMat.getNumRows() * weightsMat.getNumCols();
    modelBin.write((char*) weights.getFlat().data(), numWeights * sizeof(double));
    modelBin.write((char*) biases.data(), biases.size() * sizeof(double));
}

void DenseLayer::loadActivation(ifstream &modelBin) {
    uint32_t activationEncoding;
    modelBin.read((char*) &activationEncoding, sizeof(uint32_t));

    if (activationEncoding == Activation::Encodings::Linear){
        activation = new Linear();
    } else if (activationEncoding == Activation::Encodings::ReLU) {
        activation = new ReLU();
    } else if (activationEncoding == Activation::Encodings::Softmax) {
        activation = new Softmax();
    } else {
        ConsoleUtils::fatalError(
            "Unsupported activation encoding \"" + to_string(activationEncoding) + "\"."
        );
    }
}

void DenseLayer::loadFromBin(ifstream &modelBin) {
    loadActivation(modelBin);

    uint32_t numNeurons;
    modelBin.read((char*) &numNeurons, sizeof(uint32_t));

    uint32_t weightsPerNeuron;
    modelBin.read((char*) &weightsPerNeuron, sizeof(uint32_t));

    weights = Tensor({numNeurons, weightsPerNeuron});
    biases = vector<double>(numNeurons);
    
    Matrix weightsMat = weights.M();
    size_t numWeights = weightsMat.getNumRows() * weightsMat.getNumCols();
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

void DenseLayer::calActivations(const Tensor&prevActivations) {
    preActivations = prevActivations.M() * weights.M().T();
    preActivations.M().addToRows(biases);
    activations = activation->activate(preActivations);
}

const Tensor DenseLayer::getActivations() const {
    return activations;
}

void DenseLayer::backprop(
    const Tensor &prevActivations,
    double learningRate,
    const Tensor &outputGradients,
    bool isFirstLayer
) {
    dZ = outputGradients;
    Matrix dZMat = dZ.M();
    Matrix weightsMat = weights.M();

    if (!activation->isFused()) {
        dZMat *= activation->calculateGradient(preActivations).M();
    }

    size_t batchSize = dZMat.getNumRows();
    double scaleFactor = -learningRate/batchSize;

    Tensor weightGradients = dZMat.T() * prevActivations.M();
    Matrix weightsGradMat = weightGradients.M();
    vector<double> biasGradients = dZMat.colSums();

    weightsGradMat *= scaleFactor;
    VectorUtils::scaleVecInplace(biasGradients, scaleFactor);

    weightsMat += weightsGradMat;
    VectorUtils::addVecInplace(biases, biasGradients);

    if (!isFirstLayer) {
        dZ = dZMat * weightsMat;
    }
}

Tensor DenseLayer::getOutputGradient() const {
    return dZ;
}

DenseLayer::~DenseLayer() {
    delete activation;
}

uint32_t DenseLayer::getEncoding() const {
    return Layer::Encodings::DenseLayer;
}
