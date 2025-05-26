#include "activations/Relu.h"
#include <algorithm>

const double Relu::RELU_BIAS = 0.01;

vector<double> Relu::activate(const vector<double>& z) const{
    size_t size = z.size();
    vector<double> activations(size, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        activations[i] = max(0.0, z[i]);
    }
    return activations;
}

double Relu::initBias() const {
    return RELU_BIAS;
}

vector<double> Relu::calculateGradient(const vector<double> &preActivations) const {
    size_t size = preActivations.size();
    vector<double> gradient(size, 0.0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (preActivations[i] > 0) {
            gradient[i] = 1;
        } else {
            gradient[i] = 0;
        }
    }
    return gradient;
}