#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include "core/Matrix.h"
#include <omp.h>
#include <cmath>

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateLoss(
    const vector<int> &labels,
    const Matrix &probs
) const {
    size_t numRows = probs.getNumRows();
    size_t numCols = probs.getNumCols();

    double totalLoss = 0.0;
    const vector<double> &probsFlat = probs.getFlat();

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < numRows; i++) {
        totalLoss += -log(max(CROSS_ENTROPY_EPSILON, probsFlat[i * numCols + labels[i]]));
    }

    return totalLoss;
}

double CrossEntropy::calculateDerivative(
    double prob,
    size_t col,
    size_t labelIdx
) const {
    double value;
    if (col == labelIdx) {
        value = TrainingUtils::clipDerivative(prob - 1);
    } else {
        value = TrainingUtils::clipDerivative(prob);
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
    vector<double> &gradientsFlat = gradients.getFlat();
    const vector<double> &activationsFlat = activations.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            size_t labelIdx = (size_t) labels[i];
            gradientsFlat[i * numCols + j] = calculateDerivative(activationsFlat[i * numCols + j], j, labelIdx);
        }
    }
    
    return gradients;
}
