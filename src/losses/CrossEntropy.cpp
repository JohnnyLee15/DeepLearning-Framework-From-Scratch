#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include <omp.h>
#include <cmath>

const double CrossEntropy::CROSS_ENTROPY_EPSILON = 1e-10;

double CrossEntropy::calculateLoss(int label, const vector<double> &probabilities) {
    return(-log(max(CROSS_ENTROPY_EPSILON, probabilities[label])));
}

vector<double> CrossEntropy::calculateGradient(int label, const vector<double> &activations) const {
    size_t size = activations.size();
    size_t label = (size_t) label;
    vector<double> gradient(size, 0.0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        if (i == label) {
            gradient[i] = TrainingUtils::clipDerivative(activations[i] - 1);
        } else {
            gradient[i] = TrainingUtils::clipDerivative(activations[i]);
        }
    }
    return gradient;
}
