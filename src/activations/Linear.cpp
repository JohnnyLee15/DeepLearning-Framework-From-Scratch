#include "activations/Linear.h"
#include <algorithm>
#include "core/Matrix.h"

const double Linear::LINEAR_BIAS = 0.01;

Matrix Linear::activate(const Matrix& z) const{
    return z;
}

vector<double> Linear::initBias(size_t numBiases) const {
    vector<double> biases(numBiases);
    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biases[i] = LINEAR_BIAS;
    }

    return biases;
}

Matrix Linear::calculateGradient(const Matrix &preActivations) const {
    size_t numRows = preActivations.getNumRows();
    size_t numCols = preActivations.getNumCols();
    size_t size = numRows * numCols;

    Matrix gradients(numRows, numCols);
    vector<double> &gradientsFlat = gradients.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        gradientsFlat[i] = 1;
    }
    return gradients;
}

uint32_t Linear::getEncoding() const {
    return Activation::Encodings::Linear;
}