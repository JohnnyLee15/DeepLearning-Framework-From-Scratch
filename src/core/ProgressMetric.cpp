#include "core/ProgressAccuracy.h"
#include "core/Batch.h"
#include "losses/Loss.h"

ProgressMetric::ProgressMetric(size_t numSamples) : numSamples(numSamples) {}

void ProgressMetric::init() {
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
    double batchTotalLoss
) {
    totalLoss += batchTotalLoss;
    samplesProcessed += batch.getSize();
    avgLoss = loss->formatLoss(totalLoss/samplesProcessed);
    timeElapsed = chrono::duration<double>(chrono::steady_clock::now() - startTime).count();
}

size_t ProgressMetric::getSamplesProcessed() const {
    return samplesProcessed;
}

double ProgressMetric::getTotalLoss() const {
    return totalLoss;
}

size_t ProgressMetric::getNumSamples() const {
    return numSamples;
}

double ProgressMetric::getAvgLoss() const {
    return avgLoss;
}

double ProgressMetric::getTimeElapsed() const {
    return timeElapsed;
}