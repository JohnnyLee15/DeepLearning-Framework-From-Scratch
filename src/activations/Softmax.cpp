#include "activations/Softmax.h"
#include <cmath>
#include <iostream>

const double Softmax::SOFTMAX_BIAS = 0.0;

vector<double> Softmax::activate(const vector<double> &z) const {
    vector<double> activations(z.size(), 0.0);
    double totalSum = 0;
    double maxPreAct = getMaxPreActivation(z);
    for (int i = 0; i <  z.size(); i++) {
        totalSum += exp(z[i] - maxPreAct);
    }

    for (int i = 0; i < z.size(); i++) { 
        activations[i] = exp(z[i] - maxPreAct)/totalSum;
    }

    return activations;
}

double Softmax::getMaxPreActivation(const vector<double> &z) const {
    double maxVal = -INFINITY;
    for (int i = 0; i < z.size(); i++) {
        if (z[i] > maxVal) {
            maxVal = z[i];
        }
    }

    return maxVal;
}

double Softmax::initBias() const {
    return SOFTMAX_BIAS;
}

vector<double> Softmax::calculateGradient(const vector<double>& activations) const {
    cout << "Error. Softmax gradient calculation should never be called." << endl;
    exit(1);
    return {};
}
