#include "core/metrics/ProgressAccuracy.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/data/Batch.h"
#include "utils/TrainingUtils.h"

const string ProgressAccuracy::NAME = "Accuracy";

ProgressAccuracy::ProgressAccuracy(size_t numSamples) : 
    ProgressMetric(numSamples), predictions(numSamples) {}

void ProgressAccuracy::init() {
    ProgressMetric::init();
    correctPredictions = 0;
}

string ProgressAccuracy::getName() const {
    return NAME;
}

void ProgressAccuracy::update(
    const Batch &batch,
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    ProgressMetric::update(batch, loss, outputActivations, batchTotalLoss);
    updateCorrectPredictions(
        batch.getData(), batch.getTargets().getFlat(),
        outputActivations, &batch.getIndices()
    );
}

void ProgressAccuracy::update(
    const Tensor &features,
    const vector<float> &targets,
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    ProgressMetric::update(features, targets, loss, outputActivations, batchTotalLoss);
    updateCorrectPredictions(features, targets, outputActivations);
}

void ProgressAccuracy::updateCorrectPredictions(
    const Tensor &features,
    const vector<float> &targets,
    const Tensor &outputActivations,
    const vector<size_t> *indices
) {
    size_t batchSize = features.getShape()[0];
    Matrix probsMat = outputActivations.M();
    size_t numCols = probsMat.getNumCols();
    const vector<float> &probsFlat = outputActivations.getFlat();
    size_t localCorrect = 0;

    #pragma omp parallel for reduction(+:localCorrect)
    for (size_t i = 0; i < batchSize; i++) {
        float prediction = TrainingUtils::getPrediction(probsFlat, i, numCols);

        if (indices == nullptr) {
            predictions[i] = prediction;
        } else {
            predictions[(*indices)[i]] = prediction;
        }
        
        if (prediction == targets[i]) {
            localCorrect++;
        }
    }
    
    correctPredictions += localCorrect;
}


float ProgressAccuracy::calculate() const {
    if (getSamplesProcessed() == 0) {
        return 0.0;
    }

    return 100 * (float) correctPredictions/getSamplesProcessed();
}