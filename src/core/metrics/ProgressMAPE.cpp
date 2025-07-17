#include "core/metrics/ProgressMAPE.h"
#include "core/tensor/Tensor.h"
#include "core/data/Batch.h"
#include <iostream>

const string ProgressMAPE::NAME = "MAPE";

ProgressMAPE::ProgressMAPE(size_t numSamples) : ProgressMetric(numSamples) {}

void ProgressMAPE::init() {
    ProgressMetric::init();
    numNonZeroTargets = 0;
    runningSum = 0.0;
}

string ProgressMAPE::getName() const {
    return NAME;
}

void ProgressMAPE::update(
    const Batch &batch,
    const Loss *loss,
    const Tensor &outputActivations,
    float batchTotalLoss
) {
    ProgressMetric::update(batch, loss, outputActivations, batchTotalLoss);

    const vector<float> &outputFlat = outputActivations.getFlat();
    const vector<float> &batchTargets = batch.getTargets().getFlat();

    size_t numBatchSamples = outputActivations.getSize();
    float localMapeSum = 0.0;
    size_t localNonZero = 0;

    #pragma omp parallel for reduction(+:localMapeSum, localNonZero)
    for (size_t i = 0; i < numBatchSamples; i++) {
        float actual = batchTargets[i];
        if (actual != 0) {
            localNonZero ++;
            localMapeSum += abs(outputFlat[i] - actual)/actual;
        }
    }

    runningSum += localMapeSum;
    numNonZeroTargets += localNonZero;
}

float ProgressMAPE::calculate() const {
    if (numNonZeroTargets == 0) {
        cerr << "Warning: All target values were 0. MAPE is undefined." << endl;
        return 0.0;
    }

    return 100 * runningSum/numNonZeroTargets;
}