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

    size_t batchSize = batch.getSize();
    Matrix probsMat = outputActivations.M();
    size_t numCols = probsMat.getNumCols();
    const vector<float> &probsFlat = outputActivations.getFlat();
    const vector<size_t> &indices = batch.getIndices();
    const vector<float> &batchTargets = batch.getTargets().getFlat();
    size_t localCorrect = 0;

    #pragma omp parallel for reduction(+:localCorrect)
    for (size_t i = 0; i < batchSize; i++) {
        float prediction = TrainingUtils::getPrediction(probsFlat, i, numCols);
        predictions[indices[i]] = prediction;

        if (prediction == batchTargets[i]) {
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