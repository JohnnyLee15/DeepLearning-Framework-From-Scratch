#include "utils/Scalar.h"
#include <iostream>

void Scalar::fit(const Matrix &data) {
    fitted = true;
}

void Scalar::fit(const vector<double> &data) {
    fitted = true;
}

void Scalar::checkFitted() {
    if (!fitted) {
        cerr << "Fatal Error: Cannot transform data before calling fit()." << endl;
        exit(1);
    }
}

void Scalar::checkTransformed() const {
    if (!transformed) {
        cerr << "Fatal Error: Cannot reverse transform before calling transform()." << endl;
        exit(1);
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