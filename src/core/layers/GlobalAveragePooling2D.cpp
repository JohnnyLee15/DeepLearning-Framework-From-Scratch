#include "core/layers/GlobalAveragePooling2D.h"
#include "utils/ConsoleUtils.h"
#include <omp.h>

void GlobalAveragePooling2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "GlobalAveragePooling2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void GlobalAveragePooling2D::build(const vector<size_t> &inShape, bool isInference) {
    checkBuildSize(inShape);
    Layer::build(inShape, isInference);

    size_t depth = inShape[3];

    output= Tensor({getMaxBatchSize(), depth});

    if (isInference) {
        dX = Tensor();
    } else {
        dX = Tensor(inShape);
    }
}   

vector<size_t> GlobalAveragePooling2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    return {getMaxBatchSize(), inShape[3]};
}

void GlobalAveragePooling2D::reShapeBatch(size_t currBatchSize) {
    vector<size_t> outShape = output.getShape();
    outShape[0] = currBatchSize;
    output.reShapeInPlace(outShape);

    if (dX.getSize() > 0) {
        vector<size_t> dxShape = dX.getShape();
        dxShape[0] = currBatchSize;
        dX.reShapeInPlace(dxShape);
    }
}
        
void GlobalAveragePooling2D::forward(const Tensor &input) {
    if (input.getShape()[0] != output.getShape()[0]) {
        reShapeBatch(input.getShape()[0]);
    }
    input.globalAvgPool2d(output);
}

void GlobalAveragePooling2D::backprop(
    const Tensor &input,
    float learningRate,
    Tensor &grad,
    bool isFirstLayer
) {
    (void)input;
    (void)learningRate;
    (void)isFirstLayer;
    grad.globalAvgPool2dGrad(dX);
}

const Tensor& GlobalAveragePooling2D::getOutput() const {
    return output;
}

Tensor& GlobalAveragePooling2D::getOutputGradient() {
    return dX;
}

Layer::Encodings GlobalAveragePooling2D::getEncoding() const {
    return Layer::Encodings::GlobalAveragePooling2D;
}

Layer* GlobalAveragePooling2D::clone() const {
    return new GlobalAveragePooling2D(*this);
}

void GlobalAveragePooling2D::writeBinInternal(ofstream &modelBin) const {}