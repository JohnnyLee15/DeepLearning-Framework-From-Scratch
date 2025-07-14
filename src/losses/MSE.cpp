#include "losses/MSE.h"
#include "core/Tensor.h"
#include <cassert>
#include <cmath>

float MSE::calculateTotalLoss(
    const vector<float>& targets, 
    const Tensor& activations
) const {
    const vector<float> &actFlat = activations.getFlat();
    assert(actFlat.size() == targets.size());

    size_t size = actFlat.size();
    float totalLoss = 0.0;

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < size; i++) {
        float diff = targets[i] - actFlat[i];
        totalLoss += (diff * diff);
    }

    return totalLoss;
}

Tensor MSE::calculateGradient(
    const vector<float> &targets, 
    const Tensor &activations
) const {
    const vector<float> &actFlat = activations.getFlat();
    assert(actFlat.size() == targets.size());

    size_t size = actFlat.size();
    Tensor gradients({size, 1});

    vector<float> &gradientsFlat = gradients.getFlat();
    const vector<float> &activationsFlat = activations.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        gradientsFlat[i] = 2*(activationsFlat[i] - targets[i]);
    }
    
    return gradients;
}

float MSE::formatLoss(float avgLoss) const {
    return sqrt(avgLoss);
}

uint32_t MSE::getEncoding() const {
    return Loss::Encodings::MSE;
}

