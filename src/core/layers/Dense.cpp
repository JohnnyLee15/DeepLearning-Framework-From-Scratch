#include "core/layers/Dense.h"
#include <cmath>
#include <omp.h>
#include "utils/TrainingUtils.h"
#include "core/activations/Activation.h"
#include "core/tensor/MatrixT.h"
#include <random>
#include <sstream>
#include "core/tensor/Matrix.h"
#include "core/activations/Linear.h"
#include "core/activations/ReLU.h"
#include "core/activations/Softmax.h"
#include "utils/ConsoleUtils.h"
#include "core/gpu/GpuEngine.h"

const float Dense::HE_INT_GAIN = 2.0;

Dense::Dense(size_t numNeurons,  Activation *activation) :
    numNeurons(numNeurons), activation(activation) {}

Dense::Dense() : activation(nullptr) {}

Dense::Dense(const Dense& other)
    : numNeurons(other.numNeurons),
      activations(other.activations),
      preActivations(other.preActivations),
      weights(other.weights),
      dB(other.dB),
      dW(other.dW),
      dX(other.dX),
      dA(other.dA),
      biases(other.biases),
      activation(other.activation ? other.activation->clone() : nullptr)
{}

void Dense::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            weights.uploadToGpu();
            biases.uploadToGpu();
        #endif
    }
}

void Dense::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 2) {
        ConsoleUtils::fatalError(
            "Dense build error: Expected 2D input (batch_size, features), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void Dense::allocateForwardBuffers() {
    preActivations = Tensor({getMaxBatchSize(), numNeurons});
    activations = Tensor({getMaxBatchSize(), numNeurons});
}

void Dense::allocateGradientBuffers(size_t weightsPerNeuron, bool isInference) {
    if (isInference)
        return;
        
    dB = Tensor({numNeurons});
    dW = Tensor({numNeurons, weightsPerNeuron});
    dX = Tensor({getMaxBatchSize(), weightsPerNeuron});
    dA = Tensor({getMaxBatchSize(), numNeurons});
}

void Dense::deallocateGradientBuffers(bool isInference) {
    if (!isInference)
        return;

    dB = Tensor();
    dW = Tensor();
    dA = Tensor();
    dX = Tensor();
}

vector<uint32_t> Dense::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Dense::initWeights(size_t weightsPerNeuron) {
    if (weights.getSize() != 0)
        return;

    weights = Tensor({numNeurons, weightsPerNeuron});
    float std = sqrt(HE_INT_GAIN/weightsPerNeuron);
    vector<float> &weightsFlat = weights.getFlat();
    vector<uint32_t> seeds = generateThreadSeeds();
    size_t size = weights.getSize();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
        normal_distribution<float> distribution(0, std);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            weightsFlat[i] = distribution(generator);
        }
    }
}

void Dense::initBiases() {
    if (biases.getSize() != 0)
        return;

    biases = activation->initBias(numNeurons);
}

void Dense::initParams(size_t weightsPerNeuron, bool isInference) {
    if (isInference)
        return;

    initWeights(weightsPerNeuron);
    initBiases();
    ensureGpu();
}


void Dense::build(const vector<size_t> &inShape, bool isInference) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t weightsPerNeuron = inShape[1];

    allocateForwardBuffers();
    allocateGradientBuffers(weightsPerNeuron, isInference);
    initParams(weightsPerNeuron, isInference);
    deallocateGradientBuffers(isInference);
}


vector<size_t> Dense::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), numNeurons};
}

void Dense::syncBuffers() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            weights.downloadFromGpu();
            biases.downloadFromGpu();
        #endif
    }
}

void Dense::writeBinInternal(ofstream& modelBin) const {
    uint32_t activationEncoding = activation->getEncoding();
    modelBin.write((char*) &activationEncoding, sizeof(uint32_t));

    Matrix weightsMat = weights.M();
    uint32_t numNeuronsWrite = (uint32_t) numNeurons;
    modelBin.write((char*) &numNeuronsWrite, sizeof(uint32_t));

    uint32_t weightsPerNeuronWrite = (uint32_t) weightsMat.getNumCols();
    modelBin.write((char*) &weightsPerNeuronWrite, sizeof(uint32_t));

    size_t numWeights = weightsMat.getNumRows() * weightsMat.getNumCols();
    modelBin.write((char*) weights.getFlat().data(), numWeights * sizeof(float));
    modelBin.write((char*) biases.getFlat().data(), biases.getSize() * sizeof(float));
}

void Dense::loadActivation(ifstream &modelBin) {
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

void Dense::loadFromBin(ifstream &modelBin) {
    loadActivation(modelBin);

    uint32_t numNeuronsRead;
    modelBin.read((char*) &numNeuronsRead, sizeof(uint32_t));
    numNeurons = numNeuronsRead;

    uint32_t weightsPerNeuron;
    modelBin.read((char*) &weightsPerNeuron, sizeof(uint32_t));

    weights = Tensor({numNeurons, weightsPerNeuron});
    biases = Tensor({numNeurons});
    
    Matrix weightsMat = weights.M();
    size_t numWeights = weightsMat.getNumRows() * weightsMat.getNumCols();
    modelBin.read((char*) weights.getFlat().data(), numWeights * sizeof(float));
    modelBin.read((char*) biases.getFlat().data(), biases.getSize() * sizeof(float));
    ensureGpu();
}

void Dense::reShapeBatch(size_t currBatchSize) {
    size_t weightsPerNeuron = weights.getShape()[1];

    preActivations.reShapeInPlace({currBatchSize, numNeurons});
    activations.reShapeInPlace({currBatchSize, numNeurons});

    if (dX.getSize() > 0) {
        dX.reShapeInPlace({currBatchSize, weightsPerNeuron});
        dA.reShapeInPlace({currBatchSize, numNeurons});
    }
}

void Dense::forward(const Tensor &prevActivations) {
    if (prevActivations.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(prevActivations.getShape()[0]);
    }

    prevActivations.M().mmT(weights.M().T(), preActivations);
    preActivations.M().addToRows(biases);
    activation->activate(preActivations, activations); 
}

const Tensor& Dense::getOutput() const {
    return activations;
}

void Dense::backprop(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    if (!activation->isFused()) {
        activation->calculateGradient(preActivations, dA);
        grad.hadamard(dA);
    }

    Matrix gradMat = grad.M();
    size_t batchSize = gradMat.getNumRows();
    float scaleFactor = -learningRate/batchSize;

    gradMat.T().mTm(prevActivations.M(), dW);
    gradMat.colSums(dB);

    weights.applyGrad(dW, scaleFactor);
    biases.applyGrad(dB, scaleFactor);

    if (!isFirstLayer) {
        gradMat.mm(weights, dX);
    }
}

Tensor& Dense::getOutputGradient() {
    return dX;
}

Dense::~Dense() {
    delete activation;
}

Layer::Encodings Dense::getEncoding() const {
    return Layer::Encodings::Dense;
}

Layer* Dense::clone() const {
    return new Dense(*this);
}

const Tensor& Dense::getWeights() const {
    return weights;
}

const Tensor& Dense::getBiases() const {
    return biases;
}

const Tensor& Dense::getDeltaInputs() const {
    return dX;
}