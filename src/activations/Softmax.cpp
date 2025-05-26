#include "activations/Softmax.h"
#include "utils/MatrixUtils.h"
#include <cmath>
#include <iostream>

const double Softmax::SOFTMAX_BIAS = 0.0;

vector<double> Softmax::activate(const vector<double> &z) const {
    size_t size = z.size();
    vector<double> exps(size, 0.0);
    vector<double> activations(size, 0.0);

    double totalSum = 0;
    double maxPreAct = getMaxPreActivation(z);

    #pragma omp parallel for reduction(+:totalSum)
    for (size_t i = 0; i < size; i++) {
        exps[i] = exp(z[i] - maxPreAct);
        totalSum += exps[i];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) { 
        activations[i] = exps[i]/totalSum;
    }

    return activations;
}

double Softmax::getMaxPreActivation(const vector<double> &z) const {
    double maxVal = -MatrixUtils::INF;
    size_t size = z.size();

    #pragma omp parallel for reduction(max:maxVal)
    for (size_t i = 0; i < size; i++) {
        if (z[i] > maxVal) {
            maxVal = z[i];
        }
    }

    return maxVal;
}

double Softmax::initBias() const {
    return SOFTMAX_BIAS;
}

vector<double> Softmax::calculateGradient(const vector<double>& preActivations) const {
    cout << "Error. Softmax gradient calculation should never be called." << endl;
    exit(1);
    return {};
}
