#include "core/activations/ReLU.h"
#include "core/tensor/Tensor.h"
#include <algorithm>

const float ReLU::RELU_BIAS = 0.01;

void ReLU::activate(const Tensor &z, Tensor &a) const{
    size_t size = z.getSize();
    const vector<float> &zFlat = z.getFlat();
    vector<float> &aFlat = a.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        aFlat[i] = max(0.0f, zFlat[i]);
    }
}

Tensor ReLU::initBias(size_t numBiases) const {
    Tensor biases({numBiases});
    vector<float> &biasFlat = biases.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biasFlat[i] = RELU_BIAS;
    }

    return biases;
}

void ReLU::calculateGradient(const Tensor &z, Tensor &dZ) const {
    size_t size = z.getSize();
    const vector<float> &preFlat = z.getFlat();
    vector<float> &dzFlat = dZ.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (preFlat[i] > 0) {
            dzFlat[i] = 1.0;
        } else {
            dzFlat[i] = 0.0;
        }
    }
    
}

Activation::Encodings ReLU::getEncoding() const {
    return Activation::Encodings::ReLU;
}

Activation* ReLU::clone() const {
    return new ReLU(*this);
}