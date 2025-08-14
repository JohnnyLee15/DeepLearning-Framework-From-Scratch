#include "core/metrics/ProgressAccuracy.h"
#include "core/data/Batch.h"
#include "core/losses/Loss.h"

void ProgressMetric::init(size_t n) {
    numSamples = n;
    totalLoss = 0.0;
    avgLoss = 0.0;
    timeElapsed = 0.0;
    startTime = chrono::steady_clock::now();
    progressMetricName = getName();
    samplesProcessed = 0;
}

void ProgressMetric::update(
    const Batch &batch,
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    samplesProcessed += batch.getSize();
    updateCommon(loss, outputActivations, batchTotalLoss);
}

void ProgressMetric::update(
    const Tensor &features,
    const vector<float> &targets,
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    samplesProcessed += features.getShape()[0];
    updateCommon(loss, outputActivations, batchTotalLoss);
}

void ProgressMetric::updateCommon(
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    totalLoss += batchTotalLoss;
    avgLoss = loss->formatLoss(totalLoss/samplesProcessed);
    timeElapsed = chrono::duration<float>(chrono::steady_clock::now() - startTime).count();
}

size_t ProgressMetric::getSamplesProcessed() const {
    return samplesProcessed;
}

float ProgressMetric::getTotalLoss() const {
    return totalLoss;
}

size_t ProgressMetric::getNumSamples() const {
    return numSamples;
}

float ProgressMetric::getAvgLoss() const {
    return avgLoss;
}

float ProgressMetric::getTimeElapsed() const {
    return timeElapsed;
}