#include "core/Layer.h"

void Layer::writeBin(ofstream &modelBin) const {
    uint32_t layerEncoding = getEncoding();
    modelBin.write((char*) &layerEncoding, sizeof(uint32_t));
}

void Layer::build(const vector<size_t> &inShape) {}