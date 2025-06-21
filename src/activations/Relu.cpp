#include "activations/ReLU.h"
#include <algorithm>
#include "core/Matrix.h"

const double ReLU::RELU_BIAS = 0.01;

Matrix ReLU::activate(const Matrix& z) const{
    size_t numRows = z.getNumRows();
    size_t numCols = z.getNumCols();
    size_t size = numRows * numCols;
    const vector<double> &zFlat = z.getFlat();

    Matrix activations(numRows, numCols);
    vector<double> &activationsFlat = activations.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        activationsFlat[i] = max(0.0, zFlat[i]);
    }
    return activations;
}

vector<double> ReLU::initBias(size_t numBiases) const {
    vector<double> biases(numBiases);
    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biases[i] = RELU_BIAS;
    }

    return biases;
}

Matrix ReLU::calculateGradient(const Matrix &preActivations) const {
    size_t numRows = preActivations.getNumRows();
    size_t numCols = preActivations.getNumCols();
    size_t size = numRows * numCols;
    const vector<double> &preFlat = preActivations.getFlat();

    Matrix gradients(numRows, numCols);
    vector<double> &gradientsFlat = gradients.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (preFlat[i] > 0) {
            gradientsFlat[i] = 1;
        } else {
            gradientsFlat[i] = 0;
        }
    }
    return gradients;
}

uint32_t ReLU::getEncoding() const {
    return Activation::Encodings::ReLU;
}