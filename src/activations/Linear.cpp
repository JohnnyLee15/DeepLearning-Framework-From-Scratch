#include "activations/Linear.h"
#include <algorithm>
#include "core/Tensor.h"

const double Linear::LINEAR_BIAS = 0.01;

Tensor Linear::activate(const Tensor& z) const{
    return z;
}

vector<double> Linear::initBias(size_t numBiases) const {
    vector<double> biases(numBiases);
    #pragma omp parallel for
    for (size_t i = 0; i < numBiases; i++) {
        biases[i] = LINEAR_BIAS;
    }

    return biases;
}

Tensor Linear::calculateGradient(const Tensor &preActivations) const {
    size_t size = preActivations.getSize();

    Tensor gradients(preActivations.getShape());
    vector<double> &gradientsFlat = gradients.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        gradientsFlat[i] = 1;
    }
    return gradients;
}

uint32_t Linear::getEncoding() const {
    return Activation::Encodings::Linear;
}