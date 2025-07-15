#include "core/Conv2D.h"
#include "activations/Activation.h"
#include <random>
#include <omp.h>
#include "utils/ConsoleUtils.h"

Conv2D::Conv2D(
    size_t numKernals, 
    size_t kRows, 
    size_t kCols, 
    size_t strideIn,
    const string &padIn, 
    Activation *activation
) : numKernals(numKernals), kRows(kRows), kCols(kCols), activation(activation) {
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

void Conv2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "Conv2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void Conv2D::build(const vector<size_t> &inShape) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t batchSize = inShape[0];
    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    vector<size_t> outShape = getBuildOutShape(inShape);
    size_t outRows = outShape[1];
    size_t outCols = outShape[2];

    kernals = Tensor({numKernals, kRows, kCols, inDepth});
    initKernals();

    biases = activation->initBias(numKernals);

    preActivations = Tensor({batchSize, outRows, outCols, numKernals});
    activations = Tensor({batchSize, outRows, outCols, numKernals});

    dB = Tensor({numKernals});
    dW = Tensor({numKernals, kRows, kCols, inDepth});
    dA = Tensor({batchSize, outRows, outCols, numKernals});
    dX = Tensor({batchSize, inRows, inCols, inDepth});
}

vector<uint32_t> Conv2D::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Conv2D::initKernals() {
    const vector<size_t> &kernalsShape = kernals.getShape();
    size_t size = kernals.getSize();
    float std = sqrt(2.0/(kernalsShape[1] * kernalsShape[2] * kernalsShape[3]));
    vector<float> &kernalsFlat = kernals.getFlat();
    vector<uint32_t> seeds = generateThreadSeeds();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
        normal_distribution<float> distribution(0, std);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            kernalsFlat[i] = distribution(generator);
        }
    }
}

void Conv2D::initStride(size_t strideIn) {
    if (strideIn == 0) {
        ConsoleUtils::fatalError(
            "Stride must be greater than zero for Conv2D layer configuration."
        );
    }
    stride = strideIn;
}

vector<size_t> Conv2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    Tensor dummy(inShape);
    WindowDims win = dummy.computeInputWindow(kRows, kCols, padding, stride);
    return {getMaxBatchSize(), win.outRows, win.outCols, numKernals};
}

void Conv2D::reShapeBatch(size_t currBatchSize) {
    vector<size_t> outShape = activations.getShape();
    vector<size_t> inShape = dX.getShape();

    size_t outRows = outShape[1];
    size_t outCols = outShape[2];

    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    preActivations.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
    activations.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
    dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});
    dA.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
}

void Conv2D::forward(const Tensor &input) {
    if (input.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    WindowDims win = input.computeInputWindow(kRows, kCols, padding, stride);
    Tensor inputFwd = input.padIfNeeded(win, padding);

    preActivations = inputFwd.conv2dForward(kernals, win, stride, biases);
    activation->activate(preActivations, activations);
}

void Conv2D::backprop(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    (void) isFirstLayer;

    float scaleFactor = -learningRate / prevActivations.getShape()[0];
    WindowDims winInput = prevActivations.computeInputWindow(kRows, kCols, padding, stride);

    activation->calculateGradient(preActivations, dA);
    grad.hadamard(dA);

    Tensor prevActProcessed = prevActivations.padIfNeeded(winInput, padding);
    Tensor dW = prevActProcessed.conv2dWeights(grad, numKernals, kRows, kCols, stride);
    grad.reduceSumBias(dB);

    kernals.applyGrad(dW, scaleFactor);
    biases.applyGrad(dB, scaleFactor);

    if (stride > 1) {
        grad = grad.gradUpsample(stride);
    }
    WindowDims winGrad = grad.computeGradWindow(
        kRows, kCols, prevActProcessed.getShape()[1], 
        prevActProcessed.getShape()[2], stride, winInput
    );
    grad = grad.padWindowInput(winGrad);
    dX = grad.conv2dInput(kernals);
}


Tensor& Conv2D::getOutput() {
    return activations;
}

Tensor& Conv2D::getOutputGradient() {
    return dX;
}

void Conv2D::writeBin(ofstream &modelBin) const {}
void Conv2D::loadFromBin(ifstream &modelBin) {}
uint32_t Conv2D::getEncoding() const {
    return Layer::Encodings::Conv2D;
}
