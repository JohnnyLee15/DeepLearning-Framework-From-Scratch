#include "core/layers/Layer.h"
#include "core/tensor/Tensor.h"

Layer::Layer() : maxBatchSize(0) {}

void Layer::syncBuffers() {}

void Layer::writeBin(ofstream &modelBin) {
    uint32_t layerEncoding = getEncoding();
    modelBin.write((char*) &layerEncoding, sizeof(uint32_t));
    
    syncBuffers();
    writeBinInternal(modelBin);
}

void Layer::loadFromBin(ifstream &modeBin) {}

void Layer::build(const vector<size_t> &inShape, bool isInference) {
    if (inShape[0] <= maxBatchSize)
        return;

    maxBatchSize = inShape[0];
}

void Layer::downloadOutputFromGpu() {}

size_t Layer::getMaxBatchSize() const {
    return maxBatchSize;
}

const Tensor& Layer::getWeights() const {
    return Tensor();
}

const Tensor& Layer::getBiases() const {
    return Tensor();
}

const Tensor& Layer::getDeltaInputs() const {
    return Tensor();
}