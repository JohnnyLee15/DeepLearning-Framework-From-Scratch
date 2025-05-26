#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include <omp.h>
#include <cmath>

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateLoss(int label, const vector<double> &probabilities) {
    return(-log(max(CROSS_ENTROPY_EPSILON, probabilities[label])));
}

vector<double> CrossEntropy::calculateGradient(int label, const vector<double> &activations) const {
    int size = activations.size();
    vector<double> gradient(size, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < activations.size(); i++) {
        if (i == label) {
            gradient[i] = TrainingUtils::clipDerivative(activations[i] - 1);
        } else {
            gradient[i] = TrainingUtils::clipDerivative(activations[i]);
        }
    }
    return gradient;
}
