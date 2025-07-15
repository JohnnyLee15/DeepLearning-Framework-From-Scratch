#include "activations/Softmax.h"
#include "core/Tensor.h"
#include "core/Matrix.h"
#include <cmath>
#include "losses/SoftmaxCrossEntropy.h"
#include <limits>

const float Softmax::SOFTMAX_BIAS = 0.0;

void Softmax::activate(const Tensor &z, Tensor &a) const {
    Matrix zMat = z.M();
    size_t numCols = zMat.getNumCols();
    size_t numRows = zMat.getNumRows();

    vector<float> &aFlat = a.getFlat();
    const vector<float> &zFlat = z.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        activateRow(aFlat, zFlat, i, numCols);
    }

}

void Softmax::activateRow(
    vector<float> &activations, 
    const vector<float> &z, 
    size_t row,
    size_t numCols
) const {

    vector<float> exps(numCols, 0.0);
    float totalSum = 0;
    float maxPreAct = getMaxPreActivation(z, row, numCols);

    for (size_t j = 0; j < numCols; j++) {
        exps[j] = exp(z[row * numCols + j] - maxPreAct);
        totalSum += exps[j];
    }

    for (size_t j = 0; j < numCols; j++) { 
        activations[row * numCols + j] =  exps[j]/totalSum;
    }
}

float Softmax::getMaxPreActivation(
    const vector<float> &z, 
    size_t row, 
    size_t numCols
) const {
    float maxVal = -numeric_limits<float>::max();

    for (size_t j = 0; j < numCols; j++) {
        float value = z[row * numCols + j];
        if (value > maxVal) {
            maxVal = value;
        }
    }

    return maxVal;
}

Tensor Softmax::initBias(size_t numBiases) const {
    Tensor biases({numBiases});
    vector<float> &biasFlat = biases.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biasFlat[i] = SOFTMAX_BIAS;
    }

    return biases;
}


void Softmax::calculateGradient(const Tensor &z, Tensor &dZ) const {
    SoftmaxCrossEntropy::checkInvalidGradientCall();
}

bool Softmax::isFused() const {
    return true;
}

uint32_t Softmax::getEncoding() const {
    return Activation::Encodings::Softmax;
}
