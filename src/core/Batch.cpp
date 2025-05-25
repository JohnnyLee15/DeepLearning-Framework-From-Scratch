#include "core/Batch.h"
#include <omp.h>
#include "losses/CrossEntropy.h"
#include "utils/TrainingUtils.h"

Batch::Batch(int numLayers, int batchSize) :
    outputGradients(batchSize),
    layerActivations(numLayers),
    layerPreActivations(numLayers),
    indices(batchSize),
    data(batchSize),
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
    const vector<vector<double> > &train,
    const vector<int> &trainLabels
) {
    int batchSize = data.size();

    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        data[i] = train[indices[i]];
        labels[i] = trainLabels[indices[i]];
    }
}

void Batch::calculateOutputGradients(
    const vector<vector<double> > &probs,
    CrossEntropy *loss
) {
    int batchSize = data.size();

    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        outputGradients[i] = loss->calculateGradient(labels[i], probs[i]);
    }
}

double Batch::calculateBatchLoss(
    const vector<vector<double> > &probs,
    CrossEntropy *loss
) {
    int batchSize = data.size();
    double batchLoss = 0.0;

    #pragma omp parallel for reduction(+:batchLoss)
    for (int i = 0; i < batchSize; i++) {
        batchLoss += loss->calculateLoss(labels[i], probs[i]);
    }

    return batchLoss;
}

void Batch::writeBatchPredictions(
    vector<int> &predictions,
    const vector<vector<double> > &probs
) const {
    int batchSize = data.size();

    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        predictions[indices[i]] = TrainingUtils::getPrediction(probs[i]);
    }
}

int Batch::getCorrectPredictions(
    const vector<int> &predictions
) const {
    int correct = 0;
    int batchSize = data.size();

    #pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < batchSize; i++) {
        if (predictions[indices[i]] == labels[i]){ 
            correct++;
        }
    }

    return correct;
}

const vector<vector<double> >& Batch::getData() const {
    return data;
}

void Batch::addLayerActivations(
    const vector<vector<double> > &activations
) {
    layerActivations[writeActivationIdx++] = activations;
}

void Batch::addLayerPreActivations(
    const vector<vector<double> > &preActivations
) {
    layerPreActivations[writePreActivationIdx++] = preActivations;
}

const vector<vector<double> >& Batch::getLayerActivation(int idx) const {
    return layerActivations[idx];
}

const vector<vector<double> >& Batch::getLayerPreActivation(int idx) const {
    return layerPreActivations[idx];
}

const vector<vector<double> >& Batch::getOutputGradients() const {
    return outputGradients;
}

void Batch::updateOutputGradients(const vector<vector<double> > &newOutputGradients) {
    outputGradients = newOutputGradients;
}

