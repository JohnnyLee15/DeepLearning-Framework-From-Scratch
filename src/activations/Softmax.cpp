#include "activations/Softmax.h"
#include "utils/MatrixUtils.h"
#include "utils/Matrix.h"
#include <cmath>
#include <iostream>

const double Softmax::SOFTMAX_BIAS = 0.0;

Matrix Softmax::activate(const Matrix &z) const {
    size_t numCols = z.getNumCols();
    size_t numRows = z.getNumRows();

    Matrix activations(numRows, numCols);
    vector<double> &activationsFlat = activations.getFlat();
    

    #pragma omp parallel for
    for (int i = 0; i < numRows; i++) {
        activateRow(activationsFlat, z, i);
    }

    return activations;
}

void Softmax::activateRow(vector<double> &activations, const Matrix &z, int row) const {
    size_t numCols = z.getNumCols();
    const vector<double> &zFlat = z.getFlat();
    vector<double> exps(numCols, 0.0);
    double totalSum = 0;
    double maxPreAct = getMaxPreActivation(z, row);

    for (size_t j = 0; j < numCols; j++) {
        exps[j] = exp(zFlat[row * numCols + j] - maxPreAct);
        totalSum += exps[j];
    }

    for (size_t j = 0; j < numCols; j++) { 
        activations[row * numCols + j] =  exps[j]/totalSum;
    }
}

double Softmax::getMaxPreActivation(const Matrix &z, int row) const {
    double maxVal = -MatrixUtils::INF;
    size_t numCols = z.getNumCols();
    const vector<double> &zFlat = z.getFlat();

    for (int j = 0; j < numCols; j++) {
        double value = zFlat[row * numCols + j];
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
