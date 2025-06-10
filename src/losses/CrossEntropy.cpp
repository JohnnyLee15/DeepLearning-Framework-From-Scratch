#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include "core/Matrix.h"
#include <iostream>
#include <omp.h>
#include <cmath>

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateTotalLoss(
    const vector<double> &labels,
    const Matrix &probs
) const {
    size_t numRows = probs.getNumRows();
    size_t numCols = probs.getNumCols();

    double totalLoss = 0.0;
    const vector<double> &probsFlat = probs.getFlat();

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < numRows; i++) {
        totalLoss += -log(max(CROSS_ENTROPY_EPSILON, probsFlat[i * numCols + (int) labels[i]]));
    }

    return totalLoss;
}

Matrix CrossEntropy::calculateGradient(
    const vector<double> &labels, 
    const Matrix &activations
) const {
    cout << "Error. Cross Entropy gradient calculation should never be called." << endl;
    exit(1);
    return Matrix(0,0);
}