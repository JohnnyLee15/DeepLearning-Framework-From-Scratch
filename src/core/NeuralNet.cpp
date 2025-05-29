#include "core/NeuralNet.h"
#include "core/Data.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils/TrainingUtils.h"
#include "utils/ConsoleUtils.h"
#include "losses/CrossEntropy.h"
#include "core/Batch.h"
#include <chrono>
#include "activations/Activation.h"

using namespace std;

NeuralNet::NeuralNet(
    const vector<int> &neuronsPerLayer,
    const vector<Activation*> &activations,
    CrossEntropy *loss
) : loss(loss) {
    size_t numNeurons = neuronsPerLayer.size();
    layers.reserve(numNeurons - 1);

    for (size_t i = 1; i < numNeurons - 1; i++) {
        layers.emplace_back(Layer(neuronsPerLayer[i], neuronsPerLayer[i-1], activations[i-1]));
    }

    layers.emplace_back(Layer(
        neuronsPerLayer[numNeurons - 1], 
        neuronsPerLayer[numNeurons - 2], 
        activations.back()
    ));
}

void NeuralNet::train(
    Data data,
    double learningRate,
    double learningDecay,
    int numEpochs,
    int batchSize
) {
    double initialLR = learningRate;
    avgLosses.resize(numEpochs);
    for (int k = 0; k < numEpochs; k++) {
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;

        vector<int> predictions(data.getTrainFeatureSize(), -1);
        double avgLoss = runEpoch(data, learningRate, predictions, batchSize);

        avgLosses[k] = avgLoss;
        learningRate = initialLR/(1 + learningDecay*k);
    }
}

double NeuralNet::runEpoch(
    Data &data,
    double learningRate,
    vector<int> &predictions,
    int batchSize
) {
    int numSamples = data.getTrainFeatureSize();
    int numBatches = (numSamples + batchSize - 1)/batchSize;

    const vector<int> &labels = data.getTrainTarget();
    const Matrix &train = data.getTrainFeatures();
    const vector<int> shuffledIndices = data.generateShuffledIndices();
    
    double totalLoss = 0.0;
    int samplesProcessed = 0;
    int correctPredictions = 0;
    auto startTime = chrono::steady_clock::now();
    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int end = min((b + 1) * batchSize, numSamples);
        int currBatchSize = end - start;
        Batch batch = makeBatch(start, end, shuffledIndices, labels, train);

        totalLoss += processBatch(batch, numSamples, predictions);
        backprop(batch, learningRate);

        samplesProcessed += (currBatchSize);
        correctPredictions += batch.getCorrectPredictions(predictions);
        auto batchTime = chrono::steady_clock::now();
        double timeElapsed = chrono::duration<double>(batchTime - startTime).count();
        double accuracy = 100 * (double) correctPredictions/samplesProcessed;
        double avgLoss = totalLoss/samplesProcessed;
        ConsoleUtils::printProgressBar(samplesProcessed, numSamples, accuracy, avgLoss, timeElapsed);
    }

    return totalLoss/numSamples;
}

Batch NeuralNet::makeBatch(
    int start,
    int end,
    const vector<int> &shuffledIndices,
    const vector<int> &labels,
    const Matrix &train
) const {
    int batchSize = end - start;
    Batch batch = Batch(layers.size() + 1, batchSize);
    batch.setBatchIndices(start, end, shuffledIndices);
    batch.setBatch(train, labels);

    return batch;
}

double NeuralNet::processBatch(
    Batch &batch,
    int numSamples,
    vector<int> &predictions
) {
    forwardPass(batch);
    const Matrix &probs = layers.back().getActivations();

    batch.calculateOutputGradients(probs, loss);
    batch.writeBatchPredictions(predictions, probs);
    return batch.calculateBatchLoss(probs, loss);
}

void NeuralNet::forwardPass(Batch &batch) {
    Matrix prevActivations = batch.getData();
    size_t numLayers = layers.size();

    batch.addLayerActivations(batch.getData());
    for (size_t j = 0; j < numLayers; j++) {
        layers[j].calActivations(prevActivations);
        prevActivations = layers[j].getActivations();
        batch.addLayerPreActivations(layers[j].getPreActivations());
        batch.addLayerActivations(layers[j].getActivations());
    }
}

void NeuralNet::backprop(Batch &batch, double learningRate) {
    int numLayers = (int) layers.size();
    for (int i = numLayers - 1; i >= 0; i--) {
        const Matrix &prevActivations = batch.getLayerActivation(i);
        layers[i].updateLayerParameters(prevActivations, learningRate, batch.getOutputGradients());
        updateOutputGradients(batch, i);
    }
}

void NeuralNet::updateOutputGradients(Batch &batch, int currLayer) {
    if (currLayer > 0) {
        Activation *prevActivation = layers[currLayer - 1].getActivation();
        const Matrix &prevPreActivations = batch.getLayerPreActivation(currLayer - 1);

        batch.updateOutputGradients(
            layers[currLayer].updateOutputGradient(
                batch.getOutputGradients(), 
                prevPreActivations, 
                prevActivation
            )
        );
    }
}

double NeuralNet::test(const Matrix &data, const vector<int> &labels) {
    forwardPassInference(data);
    vector<int> predictions = TrainingUtils::getPredictions(layers.back().getActivations());
    return TrainingUtils::getAccuracy(labels, predictions);
}

void NeuralNet::forwardPassInference(const Matrix& data) {
    Matrix prevActivations = data;
    size_t numLayers = layers.size();
    
    for (size_t j = 0; j < numLayers; j++) {
        layers[j].calActivations(prevActivations);
        prevActivations = layers[j].getActivations();
    }
}

NeuralNet::~NeuralNet() {
    delete loss;
    size_t numLayers = layers.size();
    for (size_t i = 0; i < numLayers; i++) {
        delete layers[i].getActivation();
    }
}