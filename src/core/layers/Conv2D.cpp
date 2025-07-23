#include "core/layers/Conv2D.h"
#include "core/activations/Activation.h"
#include <random>
#include <omp.h>
#include "utils/ConsoleUtils.h"
#include "core/gpu/GpuEngine.h"

Conv2D::Conv2D(
    size_t numKernals, 
    size_t kRows, 
    size_t kCols, 
    size_t strideIn,
    const string &padIn, 
    Activation *activation
) : numKernals(numKernals), kRows(kRows), kCols(kCols), activation(activation), isLoadedConv2D(false) {
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

void Conv2D::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            kernals.uploadToGpu();
            biases.uploadToGpu();
        #endif
    }
}

void Conv2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "Conv2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void Conv2D::initGradBuf() {
    size_t gradRows = winIn.outRows;
    size_t gradCols = winIn.outCols;

    if (stride > 1) {
        gradRows = stride * (gradRows - 1) + 1;
        gradCols = stride * (gradCols - 1) + 1;
    }

    winGrad = Tensor({getMaxBatchSize(), gradRows, gradCols, numKernals}).computeGradWindow(
        kRows, kCols, paddedInput.getShape()[1], 
        paddedInput.getShape()[2], stride, winIn
    );

    gradRows += winGrad.padRows;
    gradCols += winGrad.padCols;

    gradBuf = Tensor({getMaxBatchSize(), gradRows, gradCols, numKernals});
}

void Conv2D::initParams(size_t inDepth) {
    if (isLoadedConv2D)
        return;

    kernals = Tensor({numKernals, kRows, kCols, inDepth});
    initKernals();
    biases = activation->initBias(numKernals);
}

void Conv2D::build(const vector<size_t> &inShape) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t batchSize = inShape[0];
    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    winIn = Tensor({inShape}).computeInputWindow(kRows, kCols, padding, stride);

    paddedInput = Tensor({batchSize, inRows + winIn.padRows, inCols + winIn.padCols, inDepth});
    preActivations = Tensor({batchSize, winIn.outRows, winIn.outCols, numKernals});
    activations = Tensor({batchSize, winIn.outRows, winIn.outCols, numKernals});

    dB = Tensor({numKernals});
    dW = Tensor({numKernals, kRows, kCols, inDepth});
    dA = Tensor({batchSize, winIn.outRows, winIn.outCols, numKernals});
    dX = Tensor({batchSize, inRows, inCols, inDepth});

    initGradBuf();
    initParams(inDepth);
    ensureGpu();
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
    return {getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals};
}

void Conv2D::reShapeBatch(size_t currBatchSize) {
    vector<size_t> outShape = activations.getShape();
    vector<size_t> inShape = dX.getShape();
    vector<size_t> inPadShape = paddedInput.getShape();
    vector<size_t> gradShape = gradBuf.getShape();

    size_t outRows = outShape[1];
    size_t outCols = outShape[2];

    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    size_t inPadRows = inPadShape[1];
    size_t inPadCols = inPadShape[2];

    size_t gradRows = gradShape[1];
    size_t gradCols = gradShape[2];

    paddedInput.reShapeInPlace({currBatchSize, inPadRows, inPadCols, inDepth});
    preActivations.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
    activations.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
    dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});
    dA.reShapeInPlace({currBatchSize, outRows, outCols, numKernals});
    gradBuf.reShapeInPlace({currBatchSize, gradRows, gradCols, numKernals});
}

void Conv2D::forward(const Tensor &input) {
    if (input.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    const Tensor &inputFwd = input.padIfNeeded(paddedInput, winIn, padding);
    inputFwd.conv2dForward(kernals, stride, preActivations, biases);
    activation->activate(preActivations, activations);
}

void Conv2D::backprop(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    (void) isFirstLayer;
    float scaleFactor = -learningRate / input.getShape()[0];

    activation->calculateGradient(preActivations, dA);
    grad.hadamard(dA);

    const Tensor &inputBwd = input.padIfNeeded(paddedInput, winIn, padding);
    inputBwd.conv2dWeights(grad, numKernals, kRows, kCols, stride, dW);
    grad.reduceSumBias(dB);

    kernals.applyGrad(dW, scaleFactor);
    biases.applyGrad(dB, scaleFactor);

    grad.padAndUpsampleGrad(gradBuf, winGrad, stride);
    gradBuf.conv2dInput(kernals, dX);
}



const Tensor& Conv2D::getOutput() const {
    return activations;
}

Tensor& Conv2D::getOutputGradient() {
    return dX;
}

void Conv2D::writeBin(ofstream &modelBin) const {}
void Conv2D::loadFromBin(ifstream &modelBin) {}
Layer::Encodings Conv2D::getEncoding() const {
    return Layer::Encodings::Conv2D;
}
