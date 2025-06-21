#include "core/NeuralNet.h"
#include "core/Data.h"
#include <cmath>
#include <iostream>
#include "core/Matrix.h"
#include <algorithm>
#include "utils/TrainingUtils.h"
#include "utils/ConsoleUtils.h"
#include "losses/Loss.h"
#include "core/Batch.h"
#include "activations/Activation.h"
#include "utils/BinUtils.h"

using namespace std;

NeuralNet::NeuralNet(vector<Layer*> layers, Loss *loss) : 
    layers(layers), loss(loss) {}

const vector<Layer*>& NeuralNet::getLayers() const {
    return layers;
}

const Loss* NeuralNet::getLoss() const {
    return loss;
}

void NeuralNet::train(
    const Data &data,
    double learningRate,
    double learningDecay,
    size_t numEpochs,
    size_t batchSize
) {
    double initialLR = learningRate;
    avgLosses.resize(numEpochs);
    for (size_t k = 0; k < numEpochs; k++) {
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;

        vector<double> predictions(data.getNumTrainSamples(), -1);
        double avgLoss = runEpoch(data, learningRate, predictions, batchSize);

        avgLosses[k] = avgLoss;
        learningRate = initialLR/(1 + learningDecay*k);
    }
    ConsoleUtils::printSepLine();
}

double NeuralNet::runEpoch(
    const Data &data,
    double learningRate,
    vector<double> &predictions,
    size_t batchSize
) {
    EpochStats stats = initEpochStats(data);
    size_t numBatches = (stats.numSamples + batchSize - 1)/batchSize;
    vector<size_t> shuffledIndices = data.generateShuffledIndices();

    for (size_t b = 0; b < numBatches; b++) {
        size_t start = b * batchSize;
        size_t end = min((b + 1) * batchSize, stats.numSamples);
        size_t currBatchSize = end - start;
        Batch batch = makeBatch(start, end, data, shuffledIndices);

        stats.totalLoss += processBatch(data, batch, predictions);
        backprop(batch, learningRate);
        updateEpochStats(stats, data, batch, predictions, currBatchSize);
    }

    return stats.totalLoss/stats.numSamples;
}

void NeuralNet::updateEpochStats(
    EpochStats& stats, 
    const Data &data,
    const Batch &batch,
    const vector<double> &predictions,
    size_t batchSize
) const {
    stats.samplesProcessed += (batchSize);
    stats.avgLoss = loss->formatLoss(stats.totalLoss/stats.samplesProcessed);
    stats.timeElapsed = chrono::duration<double>(chrono::steady_clock::now() - stats.startTime).count();
    stats.progressMetric = data.getTask()->calculateProgressMetric(
        batch, layers.back()->getActivations(), predictions, stats
    );
    ConsoleUtils::printProgressBar(stats);
}

EpochStats NeuralNet::initEpochStats(const Data &data) const{
    return EpochStats {
        .totalLoss = 0.0,
        .timeElapsed = 0.0,
        .avgLoss = 0.0,
        .mapeSum = 0.0,
        .progressMetric = 0.0,
        .samplesProcessed = 0,
        .correctPredictions = 0,
        .numSamples = data.getNumTrainSamples(),
        .nonZeroTargets = 0,
        .startTime = chrono::steady_clock::now(),
        .progressMetricName = data.getTask()->getProgressMetricName()
    };
}

Batch NeuralNet::makeBatch(
    size_t start,
    size_t end,
    const Data &data,
    const vector<size_t> &shuffledIndices
) const {
    size_t batchSize = end - start;
    Batch batch = Batch(layers.size() + 1, batchSize);
    batch.setBatchIndices(start, end, shuffledIndices);
    batch.setBatch(data.getTrainFeatures(), data.getTrainTargets());

    return batch;
}

double NeuralNet::processBatch(
    const Data &data,
    Batch &batch,
    vector<double> &predictions
) {
    forwardPass(batch);
    const Matrix &output = layers.back()->getActivations();
    return  data.getTask()->processBatch(batch, predictions, output, loss);
}

void NeuralNet::forwardPass(Batch &batch) {
    Matrix prevActivations = batch.getData();
    size_t numLayers = layers.size();

    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->calActivations(prevActivations);
        prevActivations = layers[j]->getActivations();
    }
}

void NeuralNet::backprop(Batch &batch, double learningRate) {
    Matrix outputGradients = loss->calculateGradient(batch.getTargets(),layers.back()->getActivations());
    size_t numLayers = (int) layers.size();
    
    for (int i = numLayers - 1; i >= 0; i--) {
        bool isFirstLayer = (i == 0);
        const Matrix &prevActivations = ((i == 0) ? batch.getData() : layers[i-1]->getActivations());
        layers[i]->backprop(prevActivations, learningRate, outputGradients, isFirstLayer);
        outputGradients = layers[i]->getOutputGradient();
    }
}

Matrix NeuralNet::predict(const Data &data) {
    forwardPassInference(data.getTestFeatures());
    return data.getTask()->predict(layers.back()->getActivations());
}

void NeuralNet::forwardPassInference(const Matrix& data) {
    Matrix prevActivations = data;
    size_t numLayers = layers.size();
    
    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->calActivations(prevActivations);
        prevActivations = layers[j]->getActivations();
    }
}

NeuralNet::~NeuralNet() {
    delete loss;
    size_t numLayers = layers.size();
    for (size_t i = 0; i < numLayers; i++) {
        delete layers[i];
    }
}

void NeuralNet::saveToBin(const string &filename) const {
    BinUtils::saveModel(*this, filename);
}

NeuralNet NeuralNet::loadFromBin(const string &filename) {
    return BinUtils::loadModel(filename);
}