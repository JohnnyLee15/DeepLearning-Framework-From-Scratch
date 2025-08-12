#include "core/layers/MaxPooling2D.h"
#include "utils/ConsoleUtils.h"
#include "core/gpu/GpuEngine.h"

MaxPooling2D::MaxPooling2D(
    size_t kRows,
    size_t kCols,
    size_t strideIn,
    const string &padIn
) : kRows(kRows), kCols(kCols) {
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

MaxPooling2D::MaxPooling2D() {}

void MaxPooling2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "MaxPooling2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void MaxPooling2D::build(const vector<size_t> &inShape, bool isInference) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    winIn = Tensor({inShape}).computeInputWindow(kRows, kCols, padding, stride);

    paddedInput = Tensor({getMaxBatchSize(), inRows + winIn.padRows, inCols + winIn.padCols, inDepth});
    pooledOutput = Tensor({getMaxBatchSize(), winIn.outRows, winIn.outCols, inDepth});

    if (!isInference) {
        dX = Tensor({getMaxBatchSize(), inRows, inCols, inDepth});
    }
    
    initMaxIndices();
}

void MaxPooling2D::initStride(size_t strideIn) {
    if (strideIn == 0) {
        ConsoleUtils::fatalError(
            "Stride must be greater than zero for MaxPool2D layer configuration."
        );
    }
    stride = strideIn;
}

void MaxPooling2D::initMaxIndices() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            size_t bytes = pooledOutput.getSize() * sizeof(uint32_t);
            maxIndicesGpu = MetalBuffer(bytes);
        #endif
    }
}

vector<size_t> MaxPooling2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), winIn.outRows, winIn.outCols, inShape[3]};
}

void MaxPooling2D::reShapeBatch(size_t currBatchSize) {
    vector<size_t> outShape = pooledOutput.getShape();
    vector<size_t> inPadShape = paddedInput.getShape();

    size_t outRows = outShape[1];
    size_t outCols = outShape[2];

    size_t inPadRows = inPadShape[1];
    size_t inPadCols = inPadShape[2];
    size_t inDepth = inPadShape[3];

    paddedInput.reShapeInPlace({currBatchSize, inPadRows, inPadCols, inDepth});
    pooledOutput.reShapeInPlace({currBatchSize, outRows, outCols, inDepth});

    if (dX.getSize() > 0) {
        vector<size_t> inShape = dX.getShape();
        size_t inRows = inShape[1];
        size_t inCols = inShape[2];
        dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});
    }
}

void MaxPooling2D::forward(const Tensor &input) {
    // Add error checking
    if (input.getShape()[0] != paddedInput.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    const Tensor &inputFwd = input.padIfNeeded(paddedInput, winIn, padding, numeric_limits<float>::lowest());
    inputFwd.maxPool2d(maxIndices, kRows, kCols, stride, pooledOutput, winIn);
}

void MaxPooling2D::backprop(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &outputGradients,
    bool isFirstLayer
) {
    // Add error checking
    (void)learningRate;
    (void)isFirstLayer;
    outputGradients.maxPool2dGrad(maxIndices, dX);
}

const Tensor& MaxPooling2D::getOutput() const {
    return pooledOutput;
}

Tensor& MaxPooling2D::getOutputGradient() {
    return dX;
}

void MaxPooling2D::writeBinInternal(ofstream &modelBin) const {
    uint32_t kRowsWrite = (uint32_t) kRows;
    uint32_t kColsWrite = (uint32_t) kCols;

    modelBin.write((char*) &kRowsWrite, sizeof(uint32_t));
    modelBin.write((char*) &kColsWrite, sizeof(uint32_t));

    uint32_t strideWrite = (uint32_t) stride;
    modelBin.write((char*) &strideWrite, sizeof(uint32_t));

    uint32_t paddingWrite = (uint32_t) padding;
    modelBin.write((char*) &paddingWrite, sizeof(uint32_t));

}

void MaxPooling2D::loadFromBin(ifstream &modelBin) {
    uint32_t kRowsRead;
    uint32_t kColsRead;

    modelBin.read((char*) &kRowsRead, sizeof(uint32_t));
    modelBin.read((char*) &kColsRead, sizeof(uint32_t));

    kRows = kRowsRead;
    kCols = kColsRead;

    uint32_t strideRead;
    modelBin.read((char*) &strideRead, sizeof(uint32_t));
    stride = strideRead;

    uint32_t paddingRead;
    modelBin.read((char*) &paddingRead, sizeof(uint32_t));
    padding = (Tensor::Paddings) paddingRead;
}

Layer::Encodings MaxPooling2D::getEncoding() const {
    return Layer::Encodings::MaxPooling2D;
}

Layer* MaxPooling2D::clone() const {
    return new MaxPooling2D(*this);
}