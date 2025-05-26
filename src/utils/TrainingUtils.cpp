#include "utils/TrainingUtils.h"
#include <cmath>

const double TrainingUtils::GRADIENT_THRESHOLD = 1.0;

double TrainingUtils::getAccuracy(const vector<int> &labels, const vector<int> &predictions) {
    double correct = 0;
    size_t size = labels.size();

    for (size_t i = 0; i < size; i++) {
        if (labels[i] == predictions[i]) {
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
    size_t size = probabilities.size();

    for (size_t i = 0; i < size; i++) { 
        if (probabilities[i] > prob) {
            prediction = i;
            prob = probabilities[i];
        }
    }
    return prediction;
}

vector<int> TrainingUtils::getPredictions(const vector<vector<double> > &probabilities) {
    size_t numPreds = probabilities.size();
    vector<int> predictions(numPreds);

    #pragma omp parallel for
    for (size_t i = 0; i < numPreds; ++i) {
        predictions[i] = getPrediction(probabilities[i]);
    }

    return predictions;

}

