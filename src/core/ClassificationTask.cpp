#include "core/ClassificationTask.h"
#include "utils/ConsoleUtils.h"
#include "core/Matrix.h"
#include "utils/TrainingUtils.h"
#include "core/Batch.h"
#include "losses/Loss.h"
#include "utils/Scalar.h"

ClassificationTask::ClassificationTask() : Task(PROGRESS_METRIC_NAME) {}

const string ClassificationTask::PROGRESS_METRIC_NAME = "Accuracy";

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
    createLabelMap(targetRaw);
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        target[i] = (double) labelMap[targetRaw[i]];
    }
    ConsoleUtils::completeMessage();

    return target;
}

vector<double> ClassificationTask::parsePredictions(const Matrix &rawOutput) const {
    return TrainingUtils::getPredictions(rawOutput);
}

double ClassificationTask::processBatch(
    Batch& batch,
    vector<double>& predictions,
    const Matrix& rawOutput,
    const Loss* loss
) const  {
    batch.writeBatchPredictions(predictions, rawOutput);
    return loss->calculateTotalLoss(batch.getTargets(), rawOutput);
}

double ClassificationTask::calculateProgressMetric(
    const Batch &batch,
    const Matrix &output,
    const vector<double> &predictions,
    EpochStats &stats
) const {
    stats.correctPredictions += batch.getCorrectPredictions(predictions);
    return 100 * (double) stats.correctPredictions/stats.samplesProcessed;
}
