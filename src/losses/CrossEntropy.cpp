#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include <omp.h>
#include <cmath>

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateLoss(
    const vector<int> &labels,
    const Matrix &probs
) const {
    size_t size = probs.getNumRows();
    double totalLoss = 0.0;

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < size; i++) {
        totalLoss += -log(max(CROSS_ENTROPY_EPSILON, probs.getValue(i, labels[i])));
    }

    return totalLoss;
}

double CrossEntropy::calculateDerivative(
    const Matrix &activations,
    size_t row,
    size_t col,
    size_t labelIdx
) const {
    double value;
    if (col == labelIdx) {
        value = TrainingUtils::clipDerivative(activations.getValue(row, col) - 1);
    } else {
        value = TrainingUtils::clipDerivative(activations.getValue(row, col));
    }

    return value;
}

Matrix CrossEntropy::calculateGradient(
    const vector<int> &labels, 
    const Matrix &activations
) const {
    size_t numRows = activations.getNumRows();
    size_t numCols = activations.getNumCols();

    Matrix gradients(numRows, numCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            size_t labelIdx = (size_t) labels[i];
            gradients.setValue(i, j, calculateDerivative(activations, i, j, labelIdx));
        }
    }
    
    return gradients;
}
