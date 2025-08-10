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

const float Conv2D::HE_INT_GAIN = 2.0;

const size_t Conv2D::GPU_FAST = 0;
const size_t Conv2D::GPU_NAIVE = 1;
const size_t Conv2D::CPU = 2;

Conv2D::Conv2D(
    size_t numKernels, 
    size_t kRows, 
    size_t kCols, 
    size_t strideIn,
    const string &padIn, 
    Activation *activation
) : numKernels(numKernels), kRows(kRows), kCols(kCols), activation(activation){
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

Conv2D::Conv2D() : activation(nullptr) {}

Conv2D::Conv2D(const Conv2D &other) 
    : numKernels(other.numKernels),
      kRows(other.kRows),
      kCols(other.kCols),
      paddedInput(other.paddedInput),
      im2ColInBuf(other.im2ColInBuf),
      kernels(other.kernels),
      fastKernels(other.fastKernels),
      activations(other.activations),
      preActivations(other.preActivations),
      im2ColPreActShape(other.im2ColPreActShape),
      preActTensorShape(other.preActTensorShape),
      dB(other.dB),
      dW(other.dW),
      dA(other.dA),
      dX(other.dX),
      gradIm2ColBuf(other.gradIm2ColBuf),
      gradBuf(other.gradBuf),
      biases(other.biases),
      winIn(other.winIn),
      winGrad(other.winGrad),
      activation(other.activation ? other.activation->clone() : nullptr),
      padding(other.padding),
      stride(other.stride),
      executionMode(other.executionMode) 
{}

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
            fastKernels.uploadToGpu();
            kernels.uploadToGpu();
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

void Conv2D::initGradBuf(bool isInference) {
    if (isInference) 
        return;

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

    winGrad = Tensor({getMaxBatchSize(), gradRows, gradCols, numKernels}).computeGradWindow(
        kRows, kCols, paddedInput.getShape()[1], 
        paddedInput.getShape()[2], stride, winIn
    );

    gradRows += winGrad.padRows;
    gradCols += winGrad.padCols;

    gradBuf = Tensor({getMaxBatchSize(), gradRows, gradCols, numKernels});
}

void Conv2D::unflattenKernels() {
    if (executionMode != GPU_FAST)
        return;

    size_t inDepth = fastKernels.getShape()[0] / (kRows * kCols);
    kernels = Tensor({numKernels, kRows, kCols, inDepth});
    vector<float> &kFlat = kernels.getFlat();
    const vector<float> &kColFlat = fastKernels.getFlat();

    #pragma omp parallel for collapse(4)
    for (size_t o = 0; o < numKernels; o++) {
        for (size_t i = 0; i < kRows; i++) {
            for (size_t j = 0; j < kCols; j++) {
                for (size_t d  = 0; d < inDepth; d++) {
                    size_t kIdx = ((o * kRows + i) * kCols + j) * inDepth + d;
                    size_t row = (i * kCols + j) * inDepth + d;
                    size_t col = o;

                    kFlat[kIdx] = kColFlat[row * numKernels + col];
                }
            }
        }
    }
}

void Conv2D::flattenKernels() {
    if (executionMode != GPU_FAST || fastKernels.getSize() != 0) 
        return;

    size_t inDepth = kernels.getShape()[3];

    fastKernels = Tensor({kRows * kCols * inDepth, numKernels});
    vector<float> &kColFlat = fastKernels.getFlat();
    const vector<float> kFlat = kernels.getFlat();

    #pragma omp parallel for collapse(4)
    for (size_t o = 0; o < numKernels; o++) {
        for (size_t i = 0; i < kRows; i++) {
            for (size_t j = 0; j < kCols; j++) {
                for (size_t d  = 0; d < inDepth; d++) {
                    size_t kIdx = ((o * kRows + i) * kCols + j) * inDepth + d;
                    size_t row = (i * kCols + j) * inDepth + d;
                    size_t col = o;

                    kColFlat[row * numKernels + col] = kFlat[kIdx];
                }
            }
        }
    }

    kernels = Tensor();
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

void Conv2D::initKernels(size_t inDepth) {
    if (kernels.getSize() != 0 || fastKernels.getSize() != 0)
        return;

    kernels = Tensor({numKernels, kRows, kCols, inDepth});
    const vector<size_t> &kernelsShape = kernels.getShape();
    size_t size = kernels.getSize();
    float std = sqrt(HE_INT_GAIN/(kernelsShape[1] * kernelsShape[2] * kernelsShape[3]));
    vector<float> &kernelsFlat = kernels.getFlat();
    vector<uint32_t> seeds = generateThreadSeeds();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
        normal_distribution<float> distribution(0, std);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            kernelsFlat[i] = distribution(generator);
        }
    }
}

void Conv2D::initBiases() {
    if (biases.getSize() !=  0)
        return;

    biases = activation->initBias(numKernels);
}

void Conv2D::initParams(size_t inDepth) {
    initKernels(inDepth);
    flattenKernels();
    initBiases();
    ensureGpu();
}

void Conv2D::allocateForwardBuffers(size_t inRows, size_t inCols, size_t inDepth) {
    paddedInput = Tensor({getMaxBatchSize(), inRows + winIn.padRows, inCols + winIn.padCols, inDepth});

    if (executionMode != GPU_FAST) {
        preActivations = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernels});

    } else {
        im2ColInBuf = Tensor({getMaxBatchSize() * winIn.outRows * winIn.outCols, kRows * kCols * inDepth});
        preActTensorShape = {getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernels};
        im2ColPreActShape = {getMaxBatchSize() * winIn.outRows * winIn.outCols, numKernels};
    }

    activations = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernels});
}

void Conv2D::allocateGradientBuffers(
    size_t inRows, 
    size_t inCols, 
    size_t inDepth, 
    bool isInference
) {
    if (isInference)
        return;

    if (executionMode == CPU) {
        dB = Tensor({numKernels});
    }
    
    if (executionMode != GPU_FAST) {
        dW = Tensor({numKernels, kRows, kCols, inDepth});
        dA = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernels});
    }

    dX = Tensor({getMaxBatchSize(), inRows, inCols, inDepth});
}

void Conv2D::deallocateGradientBuffers(bool isInference) {
    if (!isInference)
        return;

    dB = Tensor();
    dW = Tensor();
    dA = Tensor();
    dX = Tensor();
}

void Conv2D::build(const vector<size_t> &inShape, bool isInference) {
    checkBuildSize(inShape);

    initExecutionMode(kRows, kCols);

    Layer::build(inShape);
    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];
    
    winIn = Tensor({inShape}).computeInputWindow(kRows, kCols, padding, stride);
    allocateForwardBuffers(inRows, inCols, inDepth);
    allocateGradientBuffers(inRows, inCols, inDepth, isInference);
    initGradBuf(isInference);
    initParams(inDepth);
    deallocateGradientBuffers(isInference);
}

vector<size_t> Conv2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), winIn.outRows, winIn.outCols, numKernels};
}

void Conv2D::reShapeGpuFastBuffers(size_t currBatchSize, size_t inDepth) {
    if (executionMode != GPU_FAST)
        return;
    
    im2ColInBuf.reShapeInPlace({currBatchSize * winIn.outRows * winIn.outCols, kRows * kCols * inDepth});
    im2ColPreActShape = {currBatchSize * winIn.outRows * winIn.outCols, numKernels};
    preActTensorShape = {currBatchSize, winIn.outRows, winIn.outCols, numKernels};

    if (gradIm2ColBuf.getSize() > 0) {
        gradIm2ColBuf.reShapeInPlace(im2ColInBuf.getShape());
    }
}

void Conv2D::reShapeCpuBuffers(size_t currBatchSize, size_t inDepth) {
    if (executionMode == GPU_FAST)
        return;

    preActivations.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernels});

    if (gradBuf.getSize() > 0) {
        const vector<size_t> &gradShape = gradBuf.getShape();
        size_t gradRows = gradShape[1];
        size_t gradCols = gradShape[2];
        gradBuf.reShapeInPlace({currBatchSize, gradRows, gradCols, numKernels});
    }
    
    if (dX.getSize() > 0) {
        const vector<size_t> &inShape = dX.getShape();
        size_t inRows = inShape[1];
        size_t inCols = inShape[2];
        dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});        
    }

    if (dA.getSize() > 0) {
        dA.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernels});
    }
}

void Conv2D::reShapeBatch(size_t currBatchSize) {
    const vector<size_t> &inPadShape = paddedInput.getShape();
    size_t inPadRows = inPadShape[1];
    size_t inPadCols = inPadShape[2];
    size_t inDepth = inPadShape[3];

    paddedInput.reShapeInPlace({currBatchSize, inPadRows, inPadCols, inDepth});
    activations.reShapeInPlace({currBatchSize, winIn.outRows, winIn.outCols, numKernels});

    reShapeCpuBuffers(currBatchSize, inDepth);
    reShapeGpuFastBuffers(currBatchSize, inDepth);
}

void Conv2D::forward(const Tensor &input) {
    if (input.getShape()[0] != activations.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    const Tensor &inputFwd = input.padIfNeeded(paddedInput, winIn, padding);
    inputFwd.conv2dForward(kernels, stride, preActivations, biases);
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
    inputBwd.conv2dWeights(grad, numKernels, kRows, kCols, stride, dW);
    grad.reduceSumBias(dB);

    kernels.applyGrad(dW, scaleFactor);
    biases.applyGrad(dB, scaleFactor);

    if (!isFirstLayer) {
        grad.padAndUpsampleGrad(gradBuf, winGrad, stride);
        gradBuf.conv2dInput(kernels, dX);
    }
}

const Tensor& Conv2D::getOutput() const {
    return activations;
}

Tensor& Conv2D::getOutputGradient() {
    return dX;
}

void Conv2D::syncBuffers() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            kernels.downloadFromGpu();
            fastKernels.downloadFromGpu();
            biases.downloadFromGpu();
            unflattenKernels();
        #endif
    }
}

void Conv2D::writeBinInternal(ofstream &modelBin) const {
    uint32_t activationEncoding = activation->getEncoding();
    modelBin.write((char*) &activationEncoding, sizeof(uint32_t));

    uint32_t numKernelsWrite = (uint32_t) numKernels;
    uint32_t kRowsWrite = (uint32_t) kRows;
    uint32_t kColsWrite = (uint32_t) kCols;
    uint32_t inDepthWrite = (uint32_t) paddedInput.getShape()[3];

    modelBin.write((char*) &numKernelsWrite, sizeof(uint32_t));
    modelBin.write((char*) &kRowsWrite, sizeof(uint32_t));
    modelBin.write((char*) &kColsWrite, sizeof(uint32_t));
    modelBin.write((char*) &inDepthWrite, sizeof(uint32_t));

    uint32_t strideWrite = (uint32_t) stride;
    modelBin.write((char*) &strideWrite, sizeof(uint32_t));
    
    uint32_t paddingWrite = (uint32_t) padding;
    modelBin.write((char*) &paddingWrite, sizeof(uint32_t));
    
    modelBin.write((char*) kernels.getFlat().data(), kernels.getSize() * sizeof(float));
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

    uint32_t numKernelsRead;
    uint32_t kRowsRead;
    uint32_t kColsRead;
    uint32_t inDepthRead;

    modelBin.read((char*) &numKernelsRead, sizeof(uint32_t));
    modelBin.read((char*) &kRowsRead, sizeof(uint32_t));
    modelBin.read((char*) &kColsRead, sizeof(uint32_t));
    modelBin.read((char*) &inDepthRead, sizeof(uint32_t));

    numKernels = numKernelsRead;
    kRows = kRowsRead;
    kCols = kColsRead;
    size_t inDepth = inDepthRead;

    uint32_t strideRead;
    modelBin.read((char*) &strideRead, sizeof(uint32_t));
    stride = strideRead;

    uint32_t paddingRead;
    modelBin.read((char*) &paddingRead, sizeof(uint32_t));
    padding = (Tensor::Paddings) paddingRead;

    kernels = Tensor({numKernels, kRows, kCols, inDepth});
    modelBin.read((char*) kernels.getFlat().data(), sizeof(float) * kernels.getSize());
    
    biases = Tensor({numKernels});
    modelBin.read((char*) biases.getFlat().data(), sizeof(float) * numKernels);
    ensureGpu();
}

Layer::Encodings Conv2D::getEncoding() const {
    return Layer::Encodings::Conv2D;
}

const Tensor& Conv2D::getWeights() const {
    if (executionMode == GPU_FAST) {
        return fastKernels;
    }
    
    return kernels;
}

const Tensor& Conv2D::getBiases() const {
    return biases;
}

const Tensor& Conv2D::getDeltaInputs() const {
    return dX;
}

Layer* Conv2D::clone() const {
    return new Conv2D(*this);
}
