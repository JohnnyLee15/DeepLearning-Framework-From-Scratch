#include "activations/Softmax.h"
#include "utils/VectorUtils.h"
#include "utils/Matrix.h"
#include <cmath>
#include <iostream>

const double Softmax::SOFTMAX_BIAS = 0.0;

Matrix Softmax::activate(const Matrix &z) const {
    size_t numCols = z.getNumCols();
    size_t numRows = z.getNumRows();

    Matrix activations(numRows, numCols);
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

Matrix Softmax::calculateGradient(const Matrix& preActivations) const {
    cout << "Error. Softmax gradient calculation should never be called." << endl;
    exit(1);
    return Matrix(0,0);
}
