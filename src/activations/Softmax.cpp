#include "activations/Softmax.h"
#include "utils/VectorUtils.h"
#include "core/Tensor.h"
#include "core/Matrix.h"
#include <cmath>
#include "losses/SoftmaxCrossEntropy.h"

const double Softmax::SOFTMAX_BIAS = 0.0;

Tensor Softmax::activate(const Tensor &z) const {
    Matrix zMat = z.M();
    size_t numCols = zMat.getNumCols();
    size_t numRows = zMat.getNumRows();

    Tensor activations({numRows, numCols});
    vector<double> &activationsFlat = activations.getFlat();
    const vector<double> &zFlat = z.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        activateRow(activationsFlat, zFlat, i, numCols);
    }

    return activations;
}

void Softmax::activateRow(
    vector<double> &activations, 
    const vector<double> &z, 
    size_t row,
    size_t numCols
) const {

    vector<double> exps(numCols, 0.0);
    double totalSum = 0;
    double maxPreAct = getMaxPreActivation(z, row, numCols);

    for (size_t j = 0; j < numCols; j++) {
        exps[j] = exp(z[row * numCols + j] - maxPreAct);
        totalSum += exps[j];
    }

    for (size_t j = 0; j < numCols; j++) { 
        activations[row * numCols + j] =  exps[j]/totalSum;
    }
}

double Softmax::getMaxPreActivation(
    const vector<double> &z, 
    size_t row, 
    size_t numCols
) const {
    double maxVal = -VectorUtils::INF;

    for (size_t j = 0; j < numCols; j++) {
        double value = z[row * numCols + j];
        if (value > maxVal) {
            maxVal = value;
        }
    }

    return maxVal;
}

vector<double> Softmax::initBias(size_t numBiases) const {
    vector<double> biases(numBiases);
    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biases[i] = SOFTMAX_BIAS;
    }

    return biases;
}

Tensor Softmax::calculateGradient(const Tensor& preActivations) const {
    SoftmaxCrossEntropy::checkInvalidGradientCall();
    return Tensor();
}

bool Softmax::isFused() const {
    return true;
}

uint32_t Softmax::getEncoding() const {
    return Activation::Encodings::Softmax;
}
