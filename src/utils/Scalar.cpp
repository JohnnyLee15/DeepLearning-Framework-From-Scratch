#include "utils/Scalar.h"
#include "utils/ConsoleUtils.h"
#include <iostream>

void Scalar::fit(const Tensor &data) {
    fitted = true;
}

void Scalar::fit(const vector<double> &data) {
    fitted = true;
}

void Scalar::checkFitted() {
    if (!fitted) {
        ConsoleUtils::fatalError("Cannot transform data before calling fit().");
    }
}

void Scalar::transform(Tensor &data) {
    checkFitted();
}

void Scalar::transform(vector<double> &data) {
    checkFitted();
}

void Scalar::resetToRaw() {
    fitted = false;
}

void Scalar::writeBin(ofstream &modelBin) const {
    uint32_t scalarEncoding = getEncoding();
    modelBin.write((char*) &scalarEncoding, sizeof(uint32_t));

    uint8_t fittedByte = fitted ? 1 : 0;
    modelBin.write((char*) &fittedByte, sizeof(uint8_t));
}

void Scalar::loadFromBin(ifstream &modelBin) {
    uint8_t fittedByte;
    modelBin.read((char*) &fittedByte, sizeof(uint8_t));
    fitted = (fittedByte == 1);
}