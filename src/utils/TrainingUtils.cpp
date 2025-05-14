#include "utils/TrainingUtils.h"
#include <cmath>

const double TrainingUtils::GRADIENT_THRESHOLD = 1.0;

double TrainingUtils::getAccuracy(const vector<double> &labels, const vector<int> &predictions) {
    double correct = 0;
    for (int i = 0; i < labels.size(); i++) {
        int label = int (labels[i]);
        if (label == predictions[i]) {
            correct++;
        }
    }
    return correct/predictions.size();
}

double TrainingUtils::clipDerivative(double gradient) {
    double clip = 0.0;
    if (!isnan(gradient)) {
        clip = max(-GRADIENT_THRESHOLD, min(GRADIENT_THRESHOLD, gradient));
    }
    return clip;
}

int TrainingUtils::getPrediction(const vector<double> &probabilities) {
    int prediction = -1;
    double prob = -1;
    for (int i = 0; i < probabilities.size(); i++) { 
        if (probabilities[i] > prob) {
            prediction = i;
            prob = probabilities[i];
        }
    }
    return prediction;
}
