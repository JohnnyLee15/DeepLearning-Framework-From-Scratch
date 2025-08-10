#include "core/activations/Linear.h"
#include "core/tensor/Tensor.h"
#include <algorithm>

const float Linear::LINEAR_BIAS = 0.0;

void Linear::activate(const Tensor& z, Tensor &a) const {
    memcpy(a.getFlat().data(), z.getFlat().data(), z.getSize() * sizeof(float));
}

Tensor Linear::initBias(size_t numBiases) const {
    Tensor biases({numBiases});
    vector<float> &biasFlat = biases.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biasFlat[i] = LINEAR_BIAS;
    }
    
    return biases;
}

void Linear::calculateGradient(const Tensor &z, Tensor &dZ) const {
    size_t size = z.getSize();

    vector<float> &dzFlat = dZ.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        dzFlat[i] = 1;
    }
}

Activation::Encodings Linear::getEncoding() const {
    return Activation::Encodings::Linear;
}

Activation* Linear::clone() const {
    return new Linear(*this);
}