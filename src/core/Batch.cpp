#include "core/Batch.h"
#include <omp.h>
#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"

Batch::Batch(int numLayers, int batchSize) :
    batchSize(batchSize),
    layerActivations(numLayers),
    layerPreActivations(numLayers),
    indices(batchSize),
    labels(batchSize),
    writeActivationIdx(0),
    writePreActivationIdx(0) 
{}

void Batch::setBatchIndices(
    int start,
    int end,
    const vector<int> &shuffledIndices
) {
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        indices[i - start] = shuffledIndices[i];
    }
}

void Batch::setBatch(
    const Matrix &train,
    const vector<int> &trainLabels
) {
    size_t trainCols = train.getNumCols();

    data = Matrix(batchSize, trainCols);
    vector<double> &batchFlat = data.getFlat();
    const vector<double> &trainFlat = train.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        int rdIdx = indices[i];
        for (size_t j = 0; j < trainCols; j++) {
            batchFlat[i*trainCols + j] = trainFlat[rdIdx * trainCols + j];
        }

        labels[i] = trainLabels[rdIdx];
    }
}

void Batch::calculateOutputGradients(
    const Matrix &probs,
    CrossEntropy *loss
) {
    outputGradients = loss->calculateGradient(labels, probs);
}

double Batch::calculateBatchLoss(
    const Matrix &probs,
    CrossEntropy *loss
) {
    return loss->calculateLoss(labels, probs);
}

void Batch::writeBatchPredictions(
    vector<int> &predictions,
    const Matrix &probs
) const {
    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        predictions[indices[i]] = TrainingUtils::getPrediction(probs, i);
    }
}

int Batch::getCorrectPredictions(
    const vector<int> &predictions
) const {
    int correct = 0;
    size_t batchSize = data.getNumRows();

    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < batchSize; i++) {
        if (predictions[indices[i]] == labels[i]){ 
            correct++;
        }
    }

    return correct;
}

const Matrix& Batch::getData() const {
    return data;
}

void Batch::addLayerActivations(
    const Matrix &activations
) {
    layerActivations[writeActivationIdx++] = activations;
}

void Batch::addLayerPreActivations(
    const Matrix &preActivations
) {
    layerPreActivations[writePreActivationIdx++] = preActivations;
}

const Matrix& Batch::getLayerActivation(int idx) const {
    return layerActivations[idx];
}

const Matrix& Batch::getLayerPreActivation(int idx) const {
    return layerPreActivations[idx];
}

const Matrix& Batch::getOutputGradients() const {
    return outputGradients;
}

void Batch::updateOutputGradients(const Matrix &newOutputGradients) {
    outputGradients = newOutputGradients;
}

