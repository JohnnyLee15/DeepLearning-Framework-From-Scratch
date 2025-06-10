#include "core/Batch.h"
#include <omp.h>
#include "core/Layer.h"
#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"

Batch::Batch(int numLayers, int batchSize) :
    batchSize(batchSize),
    layerActivations(numLayers),
    layerPreActivations(numLayers),
    indices(batchSize),
    targets(batchSize),
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
    const vector<double> &trainLabels
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

        targets[i] = trainLabels[rdIdx];
    }
}

void Batch::calculateOutputGradients(
    const Layer &outputLayer,
    const Loss *loss
) {
    const Matrix &activations = outputLayer.getActivations();
    const Matrix &preActivations = outputLayer.getPreActivations();
    outputGradients = loss->calculateGradient(targets, activations);
    if (!loss->isFused()) {
        outputGradients *= outputLayer.getActivation()->calculateGradient(preActivations);
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

int Batch::getCorrectPredictions(
    const vector<double> &predictions
) const {
    int correct = 0;
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
