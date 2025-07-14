#include "losses/SoftmaxCrossEntropy.h"
#include "utils/TrainingUtils.h"
#include "core/Matrix.h"
#include "utils/ConsoleUtils.h"
#include <cmath>

const float SoftmaxCrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

float SoftmaxCrossEntropy::calculateTotalLoss(
    const vector<float> &labels,
    const Tensor &probs
) const {
    Matrix probsMat = probs.M();
    size_t numRows = probsMat.getNumRows();
    size_t numCols = probsMat.getNumCols();

    float totalLoss = 0.0;
    const vector<float> &probsFlat = probs.getFlat();

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < numRows; i++) {
        totalLoss += -log(max(CROSS_ENTROPY_EPSILON, probsFlat[i * numCols + (int) labels[i]]));
    }

    return totalLoss;
}

float SoftmaxCrossEntropy::calculateDerivative(
    float prob,
    size_t col,
    size_t labelIdx
) const {
    float value;
    if (col == labelIdx) {
        value = TrainingUtils::clipDerivative(prob - 1);
    } else {
        value = TrainingUtils::clipDerivative(prob);
    }

    return value;
}

Tensor SoftmaxCrossEntropy::calculateGradient(
    const vector<float> &labels, 
    const Tensor &activations
) const {
    Matrix actMat = activations.M();
    size_t numRows = actMat.getNumRows();
    size_t numCols = actMat.getNumCols();

    Tensor gradients({numRows, numCols});
    vector<float> &gradientsFlat = gradients.getFlat();
    const vector<float> &activationsFlat = activations.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            size_t labelIdx = (size_t) labels[i];
            gradientsFlat[i * numCols + j] = calculateDerivative(activationsFlat[i * numCols + j], j, labelIdx);
        }
    }
    
    return gradients;
}

uint32_t SoftmaxCrossEntropy::getEncoding() const {
    return Loss::Encodings::SoftmaxCrossEntropy;
}

void SoftmaxCrossEntropy::checkInvalidGradientCall() {
    ConsoleUtils::fatalError(
        "Softmax::calculateGradient() should never be called.\n"
        "Softmax must be used only with the fused SoftmaxCrossEntropy loss,\n"
        "not as a standalone activation. Please replace the CrossEntropy loss\n"
        "with SoftmaxCrossEntropy to ensure correct gradient computation."
    );
}
