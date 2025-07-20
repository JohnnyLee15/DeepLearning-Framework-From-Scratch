#include "core/layers/Layer.h"

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