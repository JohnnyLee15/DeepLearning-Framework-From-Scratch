#include "utils/TrainingUtils.h"
#include <cmath>
#include "utils/Matrix.h"

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

int TrainingUtils::getPrediction(
    const Matrix &probs,
    size_t row
) {
    size_t numChoices = probs.getNumCols();
    int prediction = -1;
    double maxProb = -1;

    for (size_t j = 0; j < numChoices; j++) {
        double prob = probs.getValue(row, j);
        if (prob > maxProb) {
            prediction = (int) j;
            maxProb = prob;
        }
    }

    return prediction;
}

vector<int> TrainingUtils::getPredictions(const Matrix &probs) {
    size_t numPreds = probs.getNumRows();
    vector<int> predictions(numPreds);

    #pragma omp parallel for
    for (size_t i = 0; i < numPreds; i++) {
        predictions[i] = getPrediction(probs, i);
    }

    return predictions;

}

