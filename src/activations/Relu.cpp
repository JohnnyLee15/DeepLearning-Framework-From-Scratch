#include "activations/Relu.h"
#include <algorithm>
#include "utils/Matrix.h"

const double Relu::RELU_BIAS = 0.01;

Matrix Relu::activate(const Matrix& z) const{
    size_t numRows = z.getNumRows();
    size_t numCols = z.getNumCols();
    size_t size = numRows * numCols;
    const vector<double> &zFlat = z.getFlat();

    Matrix activations(numRows, numCols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        activations.setValue(i, max(0.0, zFlat[i]));
    }
    return activations;
}

vector<double> Relu::initBias(size_t numBiases) const {
    vector<double> biases(numBiases);
    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biases[i] = RELU_BIAS;
    }

    return biases;
}

Matrix Relu::calculateGradient(const Matrix &preActivations) const {
    size_t numRows = preActivations.getNumRows();
    size_t numCols = preActivations.getNumCols();
    size_t size = numRows * numCols;
    const vector<double> &preFlat = preActivations.getFlat();

    Matrix gradients(numRows, numCols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (preFlat[i] > 0) {
            gradients.setValue(i, 1);
        } else {
            gradients.setValue(i, 0);
        }
    }
    return gradients;
}