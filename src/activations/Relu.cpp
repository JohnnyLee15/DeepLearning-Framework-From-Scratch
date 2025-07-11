#include "activations/ReLU.h"
#include <algorithm>
#include "core/Tensor.h"

const double ReLU::RELU_BIAS = 0.01;

Tensor ReLU::activate(const Tensor& z) const{
    size_t size = z.getSize();
    const vector<double> &zFlat = z.getFlat();
    Tensor activations(z.getShape());
    vector<double> &activationsFlat = activations.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        activationsFlat[i] = max(0.0, zFlat[i]);
    }
    return activations;
}

vector<double> ReLU::initBias(size_t numBiases) const {
    return vector<double>(numBiases, RELU_BIAS);
}

Tensor ReLU::calculateGradient(const Tensor &preActivations) const {
    size_t size = preActivations.getSize();
    const vector<double> &preFlat = preActivations.getFlat();

    Tensor gradients(preActivations.getShape());
    vector<double> &gradientsFlat = gradients.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (preFlat[i] > 0) {
            gradientsFlat[i] = 1;
        } else {
            gradientsFlat[i] = 0;
        }
    }
    return gradients;
}

uint32_t ReLU::getEncoding() const {
    return Activation::Encodings::ReLU;
}