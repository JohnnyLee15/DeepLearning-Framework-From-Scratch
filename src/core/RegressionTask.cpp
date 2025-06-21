#include "core/RegressionTask.h"
#include <iostream>
#include "core/Matrix.h"
#include "utils/ConsoleUtils.h"
#include "core/Batch.h"
#include "losses/Loss.h"
#include "utils/Scalar.h"
#include <cmath>

const string RegressionTask::PROGRESS_METRIC_NAME = "MAPE";

RegressionTask::RegressionTask() : Task(PROGRESS_METRIC_NAME), targetScalar(nullptr) {} 

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

void RegressionTask::checkNumOutputNeurons(size_t numOutputNeurons) const {
    if (numOutputNeurons != 1) {
        ConsoleUtils::fatalError(
            string("Regression tasks expect 1 output neuron.\n") +
            "Got " + to_string(numOutputNeurons) + " neurons instead."
        );
    }
}

void RegressionTask::setTargetScalar(Scalar *scalar) {
    if (targetScalar != nullptr) {
        delete targetScalar;
    }
    targetScalar = scalar;
}

vector<double> RegressionTask::parsePredictions(const Matrix& rawOutput) const {
    checkNumOutputNeurons(rawOutput.getNumCols());
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
    checkNumOutputNeurons(output.getNumCols());
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
    size_t numBatchSamples = outputFlat.size();
    double localMapeSum = 0.0;
    size_t localNonZero = 0;

    #pragma omp parallel for reduction(+:localMapeSum, localNonZero)
    for (size_t i = 0; i < numBatchSamples; i++) {
        double actual = batchTargets[i];
        if (actual != 0) {
            localNonZero ++;
            localMapeSum += abs(outputFlat[i] - actual)/actual;
        }
    }

    stats.mapeSum += localMapeSum;
    stats.nonZeroTargets += localNonZero;

    if (stats.nonZeroTargets == 0) {
        cerr << "Warning: All target values were 0. MAPE is undefined." << endl;
        return 0.0;
    }

    return 100 * stats.mapeSum/stats.nonZeroTargets;
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
    if (!targetScalar) {
        ConsoleUtils::fatalError("Target scalar must be set before calling fitScalars() in RegressionTask.");
    }

    Task::fitScalars(trainFeatures, trainTargets, testFeatures, testTargets);
    targetScalar->fit(trainTargets);
    targetScalar->transform(trainTargets);
    targetScalar->transform(testTargets);
}

void RegressionTask::resetToRaw() {
    if (!targetScalar) {
        ConsoleUtils::fatalError("Target scalar not set before resetToRaw().");
    }
    Task::resetToRaw();
    targetScalar->resetToRaw();
}

RegressionTask::~RegressionTask() {
    delete targetScalar;
}