#include "core/losses/MSE.h"
#include "core/tensor/Tensor.h"
#include <cmath>

float MSE::calculateTotalLoss(
    const Tensor& targets, 
    const Tensor& activations
) const {
    const vector<float> &actFlat = activations.getFlat();
    const vector<float> &targetsFlat = targets.getFlat();

    size_t size = activations.getSize();
    float totalLoss = 0.0;

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < size; i++) {
        float diff = targetsFlat[i] - actFlat[i];
        totalLoss += (diff * diff);
    }

    return totalLoss;
}

void MSE::calculateGradient(
    const Tensor&targets, 
    const Tensor &a,
    Tensor &dL
) const {
    vector<float> &dlFlat = dL.getFlat();
    const vector<float> &aFlat = a.getFlat();
    const vector<float> &targetsFlat = targets.getFlat();
    size_t size = a.getSize();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        dlFlat[i] = 2*(aFlat[i] - targetsFlat[i]);
    }
}

float MSE::formatLoss(float avgLoss) const {
    return sqrt(avgLoss);
}

uint32_t MSE::getEncoding() const {
    return Loss::Encodings::MSE;
}

Loss* MSE::clone() const {
    return new MSE(*this);
}

