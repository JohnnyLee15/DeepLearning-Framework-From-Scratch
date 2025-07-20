#include "core/layers/MaxPooling2D.h"
#include "utils/ConsoleUtils.h"

MaxPooling2D::MaxPooling2D(
    size_t kRows,
    size_t kCols,
    size_t strideIn,
    const string &padIn
) : kRows(kRows), kCols(kCols) {
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

void MaxPooling2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "MaxPooling2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void MaxPooling2D::build(const vector<size_t> &inShape) {
    checkBuildSize(inShape);

    Layer::build(inShape);

    size_t batchSize = inShape[0];
    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    winIn = Tensor({inShape}).computeInputWindow(kRows, kCols, padding, stride);

    paddedInput = Tensor({batchSize, inRows + winIn.padRows, inCols + winIn.padCols, inDepth});
    pooledOutput = Tensor({batchSize, winIn.outRows, winIn.outCols, inDepth});
    dX = Tensor(inShape);
}

void MaxPooling2D::initStride(size_t strideIn) {
    if (strideIn == 0) {
        ConsoleUtils::fatalError(
            "Stride must be greater than zero for MaxPool2D layer configuration."
        );
    }
    stride = strideIn;
}

vector<size_t> MaxPooling2D::getBuildOutShape(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "MaxPooling2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }

    return {getMaxBatchSize(), winIn.outRows, winIn.outCols, inShape[3]};
}

void MaxPooling2D::reShapeBatch(size_t currBatchSize) {
    vector<size_t> outShape = pooledOutput.getShape();
    vector<size_t> inShape = dX.getShape();
    vector<size_t> inPadShape = paddedInput.getShape();

    size_t outRows = outShape[1];
    size_t outCols = outShape[2];

    size_t inRows = inShape[1];
    size_t inCols = inShape[2];
    size_t inDepth = inShape[3];

    size_t inPadRows = inPadShape[1];
    size_t inPadCols = inPadShape[2];

    paddedInput.reShapeInPlace({currBatchSize, inPadRows, inPadCols, inDepth});
    pooledOutput.reShapeInPlace({currBatchSize, outRows, outCols, inDepth});
    dX.reShapeInPlace({currBatchSize, inRows, inCols, inDepth});
}

void MaxPooling2D::forward(const Tensor &input) {
    // Add error checking
    if (input.getShape()[0] != paddedInput.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }

    const Tensor &inputFwd = input.padIfNeeded(paddedInput, winIn, padding, -numeric_limits<float>::max());
    inputFwd.maxPool2d(winIn, maxIndices, kRows, kCols, stride, padding, pooledOutput);
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
    prevActivations.maxPool2dGrad(outputGradients, maxIndices, dX);
}

const Tensor& MaxPooling2D::getOutput() const {
    return pooledOutput;
}

Tensor& MaxPooling2D::getOutputGradient() {
    return dX;
}

void MaxPooling2D::writeBin(ofstream &modelBin) const {}
void MaxPooling2D::loadFromBin(ifstream &modelBin) {}
Layer::Encodings MaxPooling2D::getEncoding() const {
    return Layer::Encodings::MaxPooling2D;
}