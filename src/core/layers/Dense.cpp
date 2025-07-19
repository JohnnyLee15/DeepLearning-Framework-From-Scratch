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
    numNeurons(numNeurons), activation(activation), isLoadedDense(false) {}

Dense::Dense() : activation(nullptr), isLoadedDense(false) {}

void Dense::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            weights.uploadToGpu();
            biases.uploadToGpu();
        #endif
    }
}

void Dense::ensureCpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            weights.downloadFromGpu();
            biases.downloadFromGpu();
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

void Dense::initParams(size_t weightsPerNeuron) {
    if (isLoadedDense) 
        return;

    weights = Tensor({numNeurons, weightsPerNeuron});
    initWeights();
    biases = activation->initBias(numNeurons);
    
}

void Dense::build(const vector<size_t> &inShape) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t batchSize = getMaxBatchSize();
    size_t weightsPerNeuron = inShape[1];

    preActivations = Tensor({batchSize, numNeurons});
    activations = Tensor({batchSize, numNeurons});
    dB = Tensor({numNeurons});
    dW = Tensor({numNeurons, weightsPerNeuron});
    dX = Tensor({batchSize, weightsPerNeuron});
    dA = Tensor({batchSize, numNeurons});

    initParams(weightsPerNeuron);
    ensureGpu();
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

void Dense::initWeights() {
    size_t numRows = weights.M().getNumRows();
    size_t numCols = weights.M().getNumCols();

    size_t size = numRows * numCols;
    float std = sqrt(HE_INT_GAIN/numCols);
    vector<float> &weightsFlat = weights.getFlat();
    vector<uint32_t> seeds = generateThreadSeeds();

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

vector<size_t> Dense::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), numNeurons};
}

void Dense::writeBin(ofstream& modelBin) const {
    const_cast<Dense*>(this)->ensureCpu();
    
    Layer::writeBin(modelBin);

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

    isLoadedDense = true;
}

void Dense::reShapeBatch(size_t currBatchSize) {
    size_t weightsPerNeuron = weights.getShape()[1];

    preActivations.reShapeInPlace({currBatchSize, numNeurons});
    activations.reShapeInPlace({currBatchSize, numNeurons});
    dX.reShapeInPlace({currBatchSize, weightsPerNeuron});
    dA.reShapeInPlace({currBatchSize, numNeurons});
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