#include "utils/TrainingUtils.h"
#include <cmath>
#include "core/Matrix.h"
#include <cmath>

const double TrainingUtils::GRADIENT_THRESHOLD = 1.0;

double TrainingUtils::getAccuracy(const vector<double> &labels, const vector<double> &predictions) {
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

double TrainingUtils::getPrediction(
    const vector<double> &probsFlat,
    size_t row,
    size_t numCols
) {
    double prediction = -1.0;
    double maxProb = -1;

    for (size_t j = 0; j < numCols; j++) {
        double prob = probsFlat[row * numCols + j];
        if (prob > maxProb) {
            prediction = j;
            maxProb = prob;
        }
    }

    return prediction;
}

vector<double> TrainingUtils::getPredictions(const Tensor &probs) {
    Matrix probsMat = probs.M();
    size_t numRows = probsMat.getNumRows();
    size_t numCols = probsMat.getNumCols();
    vector<double> predictions(numRows);
    const vector<double> &probsFlat = probs.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        predictions[i] = getPrediction(probsFlat, i, numCols);
    }

    return predictions;
}


double TrainingUtils::getRMSE(
    const Tensor &predicted,
    const vector<double> &actual
) {
    const vector<double> &predictedFlat = predicted.getFlat();
    size_t size = predictedFlat.size();
    // check size

    double total = 0.0;
    for (size_t i = 0; i < size; i++) {
        double diff = (predictedFlat[i] - actual[i]);
        total += (diff * diff);
    }

    return sqrt(total / size);
}

