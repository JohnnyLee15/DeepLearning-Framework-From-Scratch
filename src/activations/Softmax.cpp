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

    #pragma omp parallel for
    for (int i = 0; i < numRows; i++) {
        activateRow(activations, z, i);
    }

    return activations;
}

void Softmax::activateRow(Matrix& activations, const Matrix &z, int row) const {
    size_t numCols = z.getNumCols();
    vector<double> exps(numCols, 0.0);
    double totalSum = 0;
    double maxPreAct = getMaxPreActivation(z, row);

    for (size_t j = 0; j < numCols; j++) {
        exps[j] = exp(z.getValue(row, j) - maxPreAct);
        totalSum += exps[j];
    }

    for (size_t j = 0; j < numCols; j++) { 
        activations.setValue(row, j, exps[j]/totalSum);
    }
}

double Softmax::getMaxPreActivation(const Matrix &z, int row) const {
    double maxVal = -MatrixUtils::INF;
    size_t numCols = z.getNumCols();

    for (int j = 0; j < numCols; j++) {
        double value = z.getValue(row, j);
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
