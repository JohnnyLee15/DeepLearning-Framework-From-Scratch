#include "core/layers/Layer.h"

Layer::Layer() : maxBatchSize(0) {}

void Layer::writeBin(ofstream &modelBin) const {
    uint32_t layerEncoding = getEncoding();
    modelBin.write((char*) &layerEncoding, sizeof(uint32_t));
}

void Layer::build(const vector<size_t> &inShape) {
    maxBatchSize = inShape[0];
}

size_t Layer::getMaxBatchSize() const {
    return maxBatchSize;
}