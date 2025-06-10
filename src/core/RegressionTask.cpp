#include "core/RegressionTask.h"
#include <iostream>
#include "core/Matrix.h"
#include "utils/ConsoleUtils.h"
#include <cassert>  
#include "core/Batch.h"
#include "losses/Loss.h"
#include "utils/Scalar.h"
#include <cmath>

const string RegressionTask::PROGRESS_METRIC_NAME = "MAPE";

RegressionTask::RegressionTask() : Task(PROGRESS_METRIC_NAME) {} 

vector<double> RegressionTask::getTarget(
    const vector<string> &targetRaw
) {
    ConsoleUtils::loadMessage("Extracting Targets.");
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        target[i] = stod(targetRaw[i]);
    }
    ConsoleUtils::completeMessage();

    return target;
}

void RegressionTask::setTargetScalar(Scalar *scalar) {
    if (targetScalar != nullptr) {
        delete targetScalar;
    }
    targetScalar = scalar;
}

vector<double> RegressionTask::parsePredictions(const Matrix& rawOutput) const {
    assert(rawOutput.getNumCols() == 1);
    return rawOutput.getFlat();
}

double RegressionTask::processBatch(
    Batch& batch,
    vector<double>& predictions,
    const Matrix& rawOutput,
    const Loss* loss
) const {
    if (targetScalar && targetScalar->isTransformed()) {
        Matrix rescaledOutput = rawOutput;
        targetScalar->reverseTransform(rescaledOutput);

        vector<double> rescaledTargets = batch.getTargets();
        targetScalar->reverseTransform(rescaledTargets);
        
        batch.setRescaledOutput(rescaledOutput);
        batch.setRescaledTargets(rescaledTargets);

        return loss->calculateTotalLoss(rescaledTargets, rescaledOutput);
    }

    return loss->calculateTotalLoss(batch.getTargets(), rawOutput);
}

double RegressionTask::calculateProgressMetric(
    const Batch &batch,
    const Matrix &output,
    const vector<double> &predictions,
    EpochStats &stats
) const {
    assert(output.getNumCols() == 1);

    vector<double> outputFlat = output.getFlat();
    vector<double> batchTargets = batch.getTargets();

    if (targetScalar && targetScalar->isTransformed()) {
        outputFlat = batch.getRescaledOutput().getFlat();
        batchTargets = batch.getRescaledTargets();
    }

    return computeMAPE(outputFlat, batchTargets, stats);
}

double RegressionTask::computeMAPE(
    const vector<double> &outputFlat,
    const vector<double> &batchTargets,
    EpochStats &stats
) const {
    size_t numSamples = outputFlat.size();
    double localMapeSum = 0.0;
    #pragma omp parallel for reduction(+:localMapeSum)
    for (size_t i = 0; i < numSamples; i++) {
        double actual = batchTargets[i];
        if (actual != 0) {
            localMapeSum += abs(outputFlat[i] - actual)/actual;
        }
    }
    stats.mapeSum += localMapeSum;
    return 100 * stats.mapeSum/stats.samplesProcessed;
}

Matrix RegressionTask::predict(const Matrix &activations) const {
    Matrix rescaledOutput = activations;
    if (targetScalar && targetScalar->isTransformed()) {
        targetScalar->reverseTransform(rescaledOutput);
    }

    return rescaledOutput;
}

void RegressionTask::fitScalars(
    Matrix &trainFeatures,
    vector<double> &trainTargets,
    Matrix &testFeatures,
    vector<double> &testTargets
) {
    Task::fitScalars(trainFeatures, trainTargets, testFeatures, testTargets);

    targetScalar->fit(trainTargets);
    targetScalar->transform(trainTargets);
    targetScalar->transform(testTargets);
}

void RegressionTask::resetToRaw() {
    Task::resetToRaw();
    targetScalar->resetToRaw();
}

RegressionTask::~RegressionTask() {
    delete targetScalar;
}