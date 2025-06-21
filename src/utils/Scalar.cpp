#include "utils/Scalar.h"
#include "utils/ConsoleUtils.h"
#include <iostream>

void Scalar::fit(const Matrix &data) {
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

void Scalar::checkTransformed() const {
    if (!transformed) {
        ConsoleUtils::fatalError("Cannot reverse transform before calling transform().");
    }
}

void Scalar::transform(Matrix &data) {
    checkFitted();
    transformed = true;
}

void Scalar::transform(vector<double> &data) {
    checkFitted();
    transformed = true;
}

void Scalar::reverseTransform(Matrix &data) const {
    checkTransformed();
}

void Scalar::reverseTransform(vector<double> &data) const {
    checkTransformed();
}

void Scalar::resetToRaw() {
    fitted = false;
    transformed = false;
}

bool Scalar::isTransformed() const {
    return transformed;
}