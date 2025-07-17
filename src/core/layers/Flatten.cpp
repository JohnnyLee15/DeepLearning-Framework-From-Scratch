#include "core/layers/Flatten.h"
#include "utils/ConsoleUtils.h"

void Flatten::checkInputSize(const vector<size_t> &givenShape) const {
    if (givenShape.size() < 2) {
        ConsoleUtils::fatalError(
            "Flatten build error: Expected input rank at least 2, "
            "but got tensor with " + to_string(givenShape.size()) + " dimensions."
        );
    }
}

void Flatten::forward(const Tensor &input) {
    const vector<size_t> &shape = input.getShape();
    size_t batchSize = shape[0];
    checkInputSize(shape);
    inShape[0] = batchSize;
    outShape[0] = batchSize;
    output = input.reShape(outShape);
}

void Flatten::backprop(
    const Tensor &prevActivations,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    // Add error checking
    (void)prevActivations;
    (void)learningRate;
    (void)isFirstLayer;
    dZ = grad.reShape(inShape);
}

Tensor& Flatten::getOutput() {
    return output;
}

Tensor& Flatten::getOutputGradient() {
    return dZ;
}

void Flatten::build(const vector<size_t> &givenShape) {
    checkInputSize(givenShape);

    Layer::build(givenShape);

    inShape = givenShape;
    outShape = getBuildOutShape(givenShape);
}

vector<size_t> Flatten::getBuildOutShape(const vector<size_t> &givenShape) const {
    checkInputSize(givenShape);

    size_t inSize = givenShape.size();
    size_t flatSize = 1;
    for (size_t i = 1; i < inSize; i++) {
        flatSize *= givenShape[i];
    }

    return {getMaxBatchSize(), flatSize};
}

void Flatten::writeBin(ofstream &modelBin) const {}
void Flatten::loadFromBin(ifstream &modelBin) {}
uint32_t Flatten::getEncoding() const {
    return Layer::Encodings::Flatten;
}
