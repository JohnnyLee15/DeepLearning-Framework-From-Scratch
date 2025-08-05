#include "core/layers/Conv2D.h"
#include "core/activations/Activation.h"
#include <random>
#include <omp.h>
#include "utils/ConsoleUtils.h"
#include "core/gpu/GpuEngine.h"
#include "utils/Im2ColUtils.h"
#include "core/activations/ReLU.h"
#include "core/activations/Linear.h"
#include "core/activations/Softmax.h"

const size_t Conv2D::GPU_FAST = 0;
const size_t Conv2D::GPU_NAIVE = 1;
const size_t Conv2D::CPU = 2;

Conv2D::Conv2D(
    size_t numKernals, 
    size_t kRows, 
    size_t kCols, 
    size_t strideIn,
    const string &padIn, 
    Activation *activation
) : numKernals(numKernals), kRows(kRows), kCols(kCols), 
    activation(activation), isInitParams(false) {
    initStride(strideIn);
    initExecutionMode(kRows, kCols);
    padding = Tensor::decodePadding(padIn);
}

Conv2D::Conv2D() : activation(nullptr), isInitParams(false) {}

void Conv2D::initStride(size_t strideIn) {
    if (strideIn == 0) {
        ConsoleUtils::fatalError(
            "Stride must be greater than zero for Conv2D layer configuration."
        );
    }
    stride = strideIn;
}

void Conv2D::initExecutionMode(size_t kRows, size_t kCols) {
    if (GpuEngine::isUsingGpu()) {
        size_t patchRows = (Im2ColUtils::getTileSize() - 1) * stride + kRows;
        size_t patchCols = (Im2ColUtils::getTileSize() - 1) * stride + kCols;
        size_t maxPatchDim = Im2ColUtils::getGpuFastSize();

        ReLU *act = dynamic_cast<ReLU*>(activation);
        
        bool fastCondition = (patchCols <= maxPatchDim && patchRows <= maxPatchDim && act != nullptr);
        executionMode = fastCondition ? GPU_FAST : GPU_NAIVE;
    } else {
        executionMode = CPU;
    }
}

void Conv2D::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            im2ColKBuf.uploadToGpu();
            kernals.uploadToGpu();
            biases.uploadToGpu();
        #endif
    }
}

void Conv2D::ensureCpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            im2ColKBuf.downloadFromGpu();
            kernals.downloadFromGpu();
            biases.downloadFromGpu();
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
    if (executionMode == GPU_FAST) {
        gradIm2ColBuf = Tensor(im2ColInBuf.getShape());
        return;
    }

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

vector<uint32_t> Conv2D::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Conv2D::initFlatKernals() {
    if (executionMode != GPU_FAST) 
        return;

    size_t inDepth = kernals.getShape()[3];

    im2ColKBuf = Tensor({kRows * kCols * inDepth, numKernals});
    vector<float> &kColFlat = im2ColKBuf.getFlat();
    const vector<float> kFlat = kernals.getFlat();

    #pragma omp parallel for collapse(4)
    for (size_t o = 0; o < numKernals; o++) {
        for (size_t i = 0; i < kRows; i++) {
            for (size_t j = 0; j < kCols; j++) {
                for (size_t d  = 0; d < inDepth; d++) {
                    size_t kIdx = ((o * kRows + i) * kCols + j) * inDepth + d;
                    size_t row = (i * kCols + j) * inDepth + d;
                    size_t col = o;

                    kColFlat[row * numKernals + col] = kFlat[kIdx];
                }
            }
        }
    }

    kernals = Tensor();
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
    initFlatKernals();
}

void Conv2D::initParams(size_t inDepth) {
    if (isInitParams)
        return;
    
    kernals = Tensor({numKernals, kRows, kCols, inDepth});
    initKernals();
    biases = activation->initBias(numKernals);
    ensureGpu();
    isInitParams = true;
}

void Conv2D::allocateForwardBuffers(size_t inRows, size_t inCols, size_t inDepth) {
    paddedInput = Tensor({getMaxBatchSize(), inRows + winIn.padRows, inCols + winIn.padCols, inDepth});

    if (executionMode != GPU_FAST) {
        preActivations = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals});

    } else {
        im2ColInBuf = Tensor({getMaxBatchSize() * winIn.outRows * winIn.outCols, kRows * kCols * inDepth});
        preActTensorShape = {getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals};
        im2ColPreActShape = {getMaxBatchSize() * winIn.outRows * winIn.outCols, numKernals};
    }

    activations = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals});
}

void Conv2D::allocateGradientBuffers(size_t inRows, size_t inCols, size_t inDepth) {
    if (executionMode == CPU) {
        dB = Tensor({numKernals});
    }
    
    if (executionMode != GPU_FAST) {
        dW = Tensor({numKernals, kRows, kCols, inDepth});
        dA = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals});
    }

    dX = Tensor({getMaxBatchSize(), inRows, inCols, inDepth});
}

void Conv2D::build(const vector<size_t> &inShape, bool isInference) {
    checkBuildSize(inShape);

    Layer::build(inShape);
    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    winIn = Tensor({inShape}).computeInputWindow(kRows, kCols, padding, stride);
    allocateForwardBuffers(inRows, inCols, inDepth);
    allocateGradientBuffers(inRows, inCols, inDepth);
    initGradBuf();
    initParams(inDepth);
}

vector<size_t> Conv2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernals};
}

void Conv2D::reShapeBatch(size_t currBatchSize) {
    const vector<size_t> &inShape = dX.getShape();
    const vector<size_t> &inPadShape = paddedInput.getShape();

    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    size_t inPadRows = inPadShape[1];
    size_t inPadCols = inPadShape[2];

    paddedInput.reShapeInPlace({currBatchSize, inPadRows, inPadCols, inDepth});
    activations.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernals});
    dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});

    if (executionMode != GPU_FAST) {
        const vector<size_t> &gradShape = gradBuf.getShape();
        size_t gradRows = gradShape[1];
        size_t gradCols = gradShape[2];

        preActivations.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernals});
        dA.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernals});
        gradBuf.reShapeInPlace({currBatchSize, gradRows, gradCols, numKernals});

    } else {
        im2ColInBuf.reShapeInPlace({currBatchSize * winIn.outRows * winIn.outCols, kRows * kCols * inDepth});
        im2ColPreActShape = {currBatchSize * winIn.outRows * winIn.outCols, numKernals};
        preActTensorShape = {currBatchSize, winIn.outRows, winIn.outCols, numKernals};
        gradIm2ColBuf.reShapeInPlace(im2ColInBuf.getShape());
    }
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
    float scaleFactor = -learningRate / input.getShape()[0];

    activation->calculateGradient(preActivations, dA);
    grad.hadamard(dA);

    const Tensor &inputBwd = input.padIfNeeded(paddedInput, winIn, padding);
    inputBwd.conv2dWeights(grad, numKernals, kRows, kCols, stride, dW);
    grad.reduceSumBias(dB);

    kernals.applyGrad(dW, scaleFactor);
    biases.applyGrad(dB, scaleFactor);

    if (!isFirstLayer) {
        grad.padAndUpsampleGrad(gradBuf, winGrad, stride);
        gradBuf.conv2dInput(kernals, dX);
    }
}

const Tensor& Conv2D::getOutput() const {
    return activations;
}

Tensor& Conv2D::getOutputGradient() {
    return dX;
}

void Conv2D::writeBin(ofstream &modelBin) const {
    const_cast<Conv2D*>(this)->ensureCpu();
    
    Layer::writeBin(modelBin);

    uint32_t activationEncoding = activation->getEncoding();
    modelBin.write((char*) &activationEncoding, sizeof(uint32_t));

    uint32_t numKernalsWrite = (uint32_t) numKernals;
    uint32_t kRowsWrite = (uint32_t) kRows;
    uint32_t kColsWrite = (uint32_t) kCols;
    uint32_t inDepthWrite = (uint32_t) paddedInput.getShape()[3];

    modelBin.write((char*) &numKernalsWrite, sizeof(uint32_t));
    modelBin.write((char*) &kRowsWrite, sizeof(uint32_t));
    modelBin.write((char*) &kColsWrite, sizeof(uint32_t));
    modelBin.write((char*) &inDepthWrite, sizeof(uint32_t));

    uint32_t executionModeWrite = (uint32_t) executionMode;
    modelBin.write((char*) &executionModeWrite, sizeof(uint32_t));

    uint32_t strideWrite = (uint32_t) stride;
    modelBin.write((char*) &strideWrite, sizeof(uint32_t));
    
    uint32_t paddingWrite = (uint32_t) padding;
    modelBin.write((char*) &paddingWrite, sizeof(uint32_t));


    if (executionMode == GPU_FAST) {
        modelBin.write((char*) im2ColKBuf.getFlat().data(), im2ColKBuf.getSize() * sizeof(float));
    } else {
        modelBin.write((char*) kernals.getFlat().data(), kernals.getSize() * sizeof(float));
    }

    modelBin.write((char*) biases.getFlat().data(), biases.getSize() * sizeof(float));
}

void Conv2D::loadActivation(ifstream &modelBin) {
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

void Conv2D::loadFromBin(ifstream &modelBin) {
    loadActivation(modelBin);

    uint32_t numKernalsRead;
    uint32_t kRowsRead;
    uint32_t kColsRead;
    uint32_t inDepthRead;

    modelBin.read((char*) &numKernalsRead, sizeof(uint32_t));
    modelBin.read((char*) &kRowsRead, sizeof(uint32_t));
    modelBin.read((char*) &kColsRead, sizeof(uint32_t));
    modelBin.read((char*) &inDepthRead, sizeof(uint32_t));

    numKernals = numKernalsRead;
    kRows = kRowsRead;
    kCols = kColsRead;
    size_t inDepth = inDepthRead;

    biases = Tensor({numKernals});

    uint32_t executionModeRead;
    modelBin.read((char*) &executionModeRead, sizeof(uint32_t));
    executionMode = executionModeRead;

    uint32_t strideRead;
    modelBin.read((char*) &strideRead, sizeof(uint32_t));
    stride = strideRead;

    uint32_t paddingRead;
    modelBin.read((char*) &paddingRead, sizeof(uint32_t));
    padding = (Tensor::Paddings) paddingRead;

    if (executionMode == GPU_FAST) {
        im2ColKBuf = Tensor({kRows * kCols * inDepth, numKernals});
        modelBin.read((char*) im2ColKBuf.getFlat().data(), sizeof(float) * im2ColKBuf.getSize());
    } else {
        kernals = Tensor({numKernals, kRows, kCols, inDepth});
        modelBin.read((char*) kernals.getFlat().data(), sizeof(float) * kernals.getSize());
    }

    modelBin.read((char*) biases.getFlat().data(), sizeof(float) * numKernals);
    ensureGpu();
    isInitParams = true;
}

Layer::Encodings Conv2D::getEncoding() const {
    return Layer::Encodings::Conv2D;
}

const Tensor& Conv2D::getWeights() const {
    if (executionMode == GPU_FAST) {
        return im2ColKBuf;
    }
    
    return kernals;
}

const Tensor& Conv2D::getBiases() const {
    return biases;
}

const Tensor& Conv2D::getDeltaInputs() const {
    return dX;
}
