#include "core/RegressionTask.h"
#include <iostream>
#include "core/Matrix.h"
#include "core/Tensor.h"
#include "utils/ConsoleUtils.h"
#include "core/Batch.h"
#include "losses/Loss.h"
#include "utils/Scalar.h"
#include "utils/Greyscale.h"
#include "utils/Minmax.h"
#include <cmath>

const string RegressionTask::PROGRESS_METRIC_NAME = "MAPE";

RegressionTask::RegressionTask() : Task(PROGRESS_METRIC_NAME), targetScalar(nullptr) {} 

RegressionTask::~RegressionTask() {
    delete targetScalar;
}

uint32_t RegressionTask::getEncoding() const {
    return Task::Encodings::Regression;
}

vector<double> RegressionTask::getTarget(
    const vector<string> &targetRaw
) {
    ConsoleUtils::loadMessage("Extracting Targets.");
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        try {
            target[i] = stod(targetRaw[i]);
        } catch (const invalid_argument &ia) {
            ConsoleUtils::fatalError(
                "RegressionTask cannot parse non-numeric target: \"" + targetRaw[i] + "\"."
            );
        }
        
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

double RegressionTask::processBatch(
    Batch& batch,
    vector<double>& predictions,
    const Tensor& rawOutput,
    const Loss* loss
) const {
    if (targetScalar) {
        Tensor rescaledOutput = rawOutput;
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
    const Tensor &output,
    const vector<double> &predictions,
    EpochStats &stats
) const {
    checkNumOutputNeurons(output.M().getNumCols());
    vector<double> outputFlat = output.getFlat();
    vector<double> batchTargets = batch.getTargets();

    if (targetScalar) {
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

Tensor RegressionTask::predict(const Tensor &activations) const {
    Tensor rescaledOutput = activations;
    if (targetScalar) {
        targetScalar->reverseTransform(rescaledOutput);
    }

    return rescaledOutput;
}

void RegressionTask::fitScalars(
    Tensor &features,
    vector<double> &targets
) {
    if (!targetScalar) {
        ConsoleUtils::fatalError("Target scalar must be set before calling fitScalars() in RegressionTask.");
    }

    Task::fitScalars(features, targets);
    targetScalar->fit(targets);
}

void RegressionTask::transformScalars(
    Tensor &features,
    vector<double> &targets
) {
    if (!targetScalar) {
        ConsoleUtils::fatalError(
            "Feature scalar must be set and fit before calling transformScalars()."
        );
    }

    Task::transformScalars(features, targets);
    targetScalar->transform(targets);
}

void RegressionTask::reverseTransformScalars(
    Tensor &features,
    vector<double> &targets
) {
    if (!targetScalar) {
        ConsoleUtils::fatalError(
            "Feature scalar must be set and fit before calling reverseTransformScalars()."
        );
    }

    Task::reverseTransformScalars(features, targets);
    targetScalar->reverseTransform(targets);
}

void RegressionTask::resetToRaw() {
    if (!targetScalar) {
        ConsoleUtils::fatalError("Target scalar not set before resetToRaw().");
    }
    Task::resetToRaw();
    targetScalar->resetToRaw();
}

void RegressionTask::writeBin(ofstream &modelBin) const {
    Task::writeBin(modelBin);

    if (targetScalar) {
        targetScalar->writeBin(modelBin);
    } else {
        uint32_t targetScalarEncodng = Scalar::Encodings::None;
        modelBin.write((char*) &targetScalarEncodng, sizeof(uint32_t));
    }
}

void RegressionTask::loadFromBin(ifstream &modelBin) {
    Task::loadFromBin(modelBin);

    uint32_t scalarEncoding;
    modelBin.read((char*) &scalarEncoding, sizeof(uint32_t));

    if (scalarEncoding == Scalar::Encodings::Greyscale) {
        targetScalar = new Greyscale();
    } else if(scalarEncoding == Scalar::Encodings::Minmax)  {
        targetScalar = new Minmax();
    } else if (scalarEncoding == Scalar::Encodings::None) {
        targetScalar = nullptr;
    } else {
        ConsoleUtils::fatalError(
            "Unsupported scalar encoding \"" + to_string(scalarEncoding) + "\"."
        ); 
    }

    if (targetScalar) {
        targetScalar->loadFromBin(modelBin);
    }
}