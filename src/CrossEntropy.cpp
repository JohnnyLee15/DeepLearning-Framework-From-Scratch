#include "CrossEntropy.h"
#include "TrainingUtils.h"

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateLoss(int label, const vector<double> &probabilities) {
    return(-log(max(CROSS_ENTROPY_EPSILON, probabilities[label])));
}

vector<double> CrossEntropy::calculateGradient(int label, const vector<double> &activations) const {
    vector<double> gradient(activations.size(), 0.0);
    for (int i = 0; i < activations.size(); i++) {
        if (i == label) {
            gradient[i] = TrainingUtils::clipDerivative(activations[i] - 1);
        } else {
            gradient[i] = TrainingUtils::clipDerivative(activations[i]);
        }
    }
    return gradient;
}
