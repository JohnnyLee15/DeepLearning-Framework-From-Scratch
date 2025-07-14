#include "utils/TrainingUtils.h"
#include <cmath>
#include "core/Matrix.h"
#include <cmath>

const float TrainingUtils::GRADIENT_THRESHOLD = 1.0;

float TrainingUtils::getAccuracy(const vector<float> &labels, const vector<float> &predictions) {
    float correct = 0;
    size_t size = labels.size();

    for (size_t i = 0; i < size; i++) {
        if (labels[i] == predictions[i]) {
            correct++;
        }
    }
    return correct/predictions.size();
}

float TrainingUtils::clipDerivative(float gradient) {
    float clip = 0.0;
    if (!isnan(gradient)) {
        clip = max(-GRADIENT_THRESHOLD, min(GRADIENT_THRESHOLD, gradient));
    }
    return clip;
}

float TrainingUtils::getPrediction(
    const vector<float> &probsFlat,
    size_t row,
    size_t numCols
) {
    float prediction = -1.0;
    float maxProb = -1;

    for (size_t j = 0; j < numCols; j++) {
        float prob = probsFlat[row * numCols + j];
        if (prob > maxProb) {
            prediction = j;
            maxProb = prob;
        }
    }

    return prediction;
}

vector<float> TrainingUtils::getPredictions(const Tensor &probs) {
    Matrix probsMat = probs.M();
    size_t numRows = probsMat.getNumRows();
    size_t numCols = probsMat.getNumCols();
    vector<float> predictions(numRows);
    const vector<float> &probsFlat = probs.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        predictions[i] = getPrediction(probsFlat, i, numCols);
    }

    return predictions;
}


float TrainingUtils::getRMSE(
    const Tensor &predicted,
    const vector<float> &actual
) {
    const vector<float> &predictedFlat = predicted.getFlat();
    size_t size = predictedFlat.size();
    // check size

    float total = 0.0;
    for (size_t i = 0; i < size; i++) {
        float diff = (predictedFlat[i] - actual[i]);
        total += (diff * diff);
    }

    return sqrt(total / size);
}

