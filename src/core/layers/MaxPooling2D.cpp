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

    Tensor dummy(inShape);
    WindowDims win = dummy.computeInputWindow(kRows, kCols, padding, stride);
    return {getMaxBatchSize(), win.outRows, win.outCols, inShape[3]};
}

void MaxPooling2D::forward(const Tensor &input) {
    // Add error checking
    WindowDims win = input.computeInputWindow(kRows, kCols, padding, stride);
    Tensor inputProcessed = input.padIfNeeded(win, padding);
    pooledOutput = inputProcessed.maxPool2d(win, maxIndices, kRows, kCols, stride, padding);
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
    dZ = prevActivations.maxPool2dGrad(outputGradients, maxIndices);
}

const Tensor& MaxPooling2D::getOutput() const {
    return pooledOutput;
}

Tensor& MaxPooling2D::getOutputGradient() {
    return dZ;
}

void MaxPooling2D::writeBin(ofstream &modelBin) const {}
void MaxPooling2D::loadFromBin(ifstream &modelBin) {}
Layer::Encodings MaxPooling2D::getEncoding() const {
    return Layer::Encodings::MaxPooling2D;
}