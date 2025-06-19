#include "core/Batch.h"
#include <omp.h>
#include "core/DenseLayer.h"
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"

Batch::Batch(size_t numLayers, size_t batchSize) :
    batchSize(batchSize),
    indices(batchSize),
    targets(batchSize)
{}

void Batch::setBatchIndices(
    size_t start,
    size_t end,
    const vector<size_t> &shuffledIndices
) {
    #pragma omp parallel for
    for (size_t i = start; i < end; i++) {
        indices[i - start] = shuffledIndices[i];
    }
}

void Batch::setBatch(
    const Matrix &train,
    const vector<double> &trainLabels
) {
    size_t trainCols = train.getNumCols();

    data = Matrix(batchSize, trainCols);
    vector<double> &batchFlat = data.getFlat();
    const vector<double> &trainFlat = train.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        size_t rdIdx = indices[i];
        for (size_t j = 0; j < trainCols; j++) {
            batchFlat[i*trainCols + j] = trainFlat[rdIdx * trainCols + j];
        }

        targets[i] = trainLabels[rdIdx];
    }
}

void Batch::writeBatchPredictions(
    vector<double> &predictions,
    const Matrix &probs
) const {
    size_t numCols = probs.getNumCols();
    const vector<double> &probsFlat = probs.getFlat();
    
    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        predictions[indices[i]] = TrainingUtils::getPrediction(probsFlat, i, numCols);
    }
}

size_t Batch::getCorrectPredictions(
    const vector<double> &predictions
) const {
    size_t correct = 0;
    size_t batchSize = data.getNumRows();

    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < batchSize; i++) {
        if (predictions[indices[i]] == targets[i]){ 
            correct++;
        }
    }

    return correct;
}

const Matrix& Batch::getData() const {
    return data;
}

const vector<double>& Batch::getTargets() const {
    return targets;
}

void Batch::setRescaledOutput(const Matrix &outputs) {
    rescaledOutput = outputs;
}

void Batch::setRescaledTargets(const vector<double> &targets) {
    rescaledTargets = targets;
}

const Matrix& Batch::getRescaledOutput() const {
    return rescaledOutput;
}

const vector<double>& Batch::getRescaledTargets() const {
    return rescaledTargets;
}
