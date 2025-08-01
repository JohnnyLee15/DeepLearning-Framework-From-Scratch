#include "core/layers/Layer.h"
#include "core/tensor/Tensor.h"

Layer::Layer() : maxBatchSize(0), isMaxBatchSet(false) {}

void Layer::writeBin(ofstream &modelBin) const {
    uint32_t layerEncoding = getEncoding();
    modelBin.write((char*) &layerEncoding, sizeof(uint32_t));
}

void Layer::build(const vector<size_t> &inShape) {
    if (isMaxBatchSet)
        return;

    maxBatchSize = inShape[0];
    isMaxBatchSet = true;
}

void Layer::downloadOutputFromGpu() {}

size_t Layer::getMaxBatchSize() const {
    return maxBatchSize;
}

const Tensor& Layer::getDeltaWeights() const {
    return Tensor();
}

const Tensor& Layer::getDeltaWeightsIm2Col() const {
    return Tensor();
}


const Tensor& Layer::getDeltaBiases() const {
    return Tensor();
}

const Tensor& Layer::getDeltaInputs() const {
    return Tensor();
}