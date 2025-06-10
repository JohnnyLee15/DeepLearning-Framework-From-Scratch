#include "utils/Scalar.h"
#include <iostream>

void Scalar::fit(const Matrix &data) {
    fitted = true;
}

void Scalar::fit(const vector<double> &data) {
    fitted = true;
}

void Scalar::transform(Matrix &data) {
    if (!fitted) {
        cout << "Error: Must call fit() before transform()." << endl;
    }
    transformed = true;
}

void Scalar::transform(vector<double> &data) {
    if (!fitted) {
        cout << "Error: Must call fit() before transform()." << endl;
    }
    transformed = true;
}

void Scalar::reverseTransform(Matrix &data) const {
    if (!transformed) {
        cout << "Error: Must call transform() before reverseTransform()." << endl;
    }
}

void Scalar::reverseTransform(vector<double> &data) const {
    if (!transformed) {
        cout << "Error: Must call transform() before reverseTransform()." << endl;
    }
}

void Scalar::setFitted(bool newFitted) {
    fitted = newFitted;
}

void Scalar::setTransforemd(bool newTransformed) {
    transformed = newTransformed;
}

void Scalar::resetToRaw() {
    fitted = false;
    transformed = false;
}

bool Scalar::isTransformed() const {
    return transformed;
}