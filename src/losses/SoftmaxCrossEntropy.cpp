#include "losses/SoftmaxCrossEntropy.h"
#include "utils/TrainingUtils.h"
#include "core/Matrix.h"
#include <iostream>

double SoftmaxCrossEntropy::calculateDerivative(
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

Matrix SoftmaxCrossEntropy::calculateGradient(
    const vector<double> &labels, 
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

bool SoftmaxCrossEntropy::isFused() const {
    return true;
}

void SoftmaxCrossEntropy::checkInvalidGradientCall() {
    cerr << "Fatal Error: Softmax::calculateGradient() should never be called.\n"
         << "Softmax must be used only with the fused SoftmaxCrossEntropy loss,\n"
         << "not as a standalone activation. Please replace the CrossEntropy loss\n"
         << "with SoftmaxCrossEntropy to ensure correct gradient computation." << endl;
    exit(1);
}
