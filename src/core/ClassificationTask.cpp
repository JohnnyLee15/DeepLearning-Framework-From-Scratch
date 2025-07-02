#include "core/ClassificationTask.h"
#include "utils/ConsoleUtils.h"
#include "core/Tensor.h"
#include "utils/TrainingUtils.h"
#include "core/Batch.h"
#include "losses/Loss.h"
#include "utils/Scalar.h"
#include <iostream>

const string ClassificationTask::PROGRESS_METRIC_NAME = "Accuracy";

ClassificationTask::ClassificationTask() : Task(PROGRESS_METRIC_NAME) {}

void ClassificationTask::createLabelMap(
    const vector<string> &targetRaw
) {
    size_t numSamples = targetRaw.size();
    int nextIdx = 0;

    for (size_t i = 0; i < numSamples; i++) {
        const string &val = targetRaw[i];
        if (labelMap.find(val) == labelMap.end()) {
            labelMap[val] = nextIdx++;
        }
    }
}

vector<double> ClassificationTask::getTarget(
    const vector<string> &targetRaw
) {
    ConsoleUtils::loadMessage("Extracting Targets.");

    if (labelMap.empty()) {
        createLabelMap(targetRaw);
    }
    
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        if (labelMap.find(targetRaw[i]) == labelMap.end()) {
            ConsoleUtils::fatalError(
                "Unknown label \"" + targetRaw[i] + "\' encountered."
            );
        }

        target[i] = (double) labelMap[targetRaw[i]];
    }
    ConsoleUtils::completeMessage();

    return target;
}

double ClassificationTask::processBatch(
    Batch& batch,
    vector<double>& predictions,
    const Tensor& rawOutput,
    const Loss* loss
) const  {
    batch.writeBatchPredictions(predictions, rawOutput);
    return loss->calculateTotalLoss(batch.getTargets(), rawOutput);
}

double ClassificationTask::calculateProgressMetric(
    const Batch &batch,
    const Tensor &output,
    const vector<double> &predictions,
    EpochStats &stats
) const {
    stats.correctPredictions += batch.getCorrectPredictions(predictions);
    return 100 * (double) stats.correctPredictions/stats.samplesProcessed;
}

void ClassificationTask::writeBin(ofstream &modelBin) const {
    Task::writeBin(modelBin);

    uint32_t mapSize = labelMap.size();
    modelBin.write((char*) &mapSize, sizeof(uint32_t));
    for (const pair<const string, int > &pair : labelMap) {
        uint32_t keyLen = pair.first.size();
        modelBin.write((char*) &keyLen, sizeof(uint32_t));
        modelBin.write(pair.first.c_str(), keyLen);

        uint32_t mapVal = pair.second;
        modelBin.write((char*) &mapVal, sizeof(uint32_t));
    }
}

uint32_t ClassificationTask::getEncoding() const {
    return Task::Encodings::Classification;
}

void ClassificationTask::loadFromBin(ifstream &modelBin) {
    Task::loadFromBin(modelBin);

    uint32_t mapSize;
    modelBin.read((char*) &mapSize, sizeof(uint32_t));

    for (uint32_t i = 0; i < mapSize; i++) {
        uint32_t keyLen;
        modelBin.read((char*) &keyLen, sizeof(uint32_t));

        string key(keyLen, '\0');
        modelBin.read(key.data(), keyLen);

        uint32_t value;
        modelBin.read((char*) &value, sizeof(uint32_t));

        labelMap[key] = value;
    }
}
