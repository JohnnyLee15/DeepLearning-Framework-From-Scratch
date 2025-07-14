#include "core/NeuralNet.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils/TrainingUtils.h"
#include "utils/ConsoleUtils.h"
#include "losses/Loss.h"
#include "core/Batch.h"
#include "activations/Activation.h"
#include "utils/BinUtils.h"
#include "core/ProgressMetric.h"
#include "losses/MSE.h"
#include "losses/SoftmaxCrossEntropy.h"
#include "core/DenseLayer.h"

random_device NeuralNet::rd;
mt19937 NeuralNet::generator(NeuralNet::rd());

NeuralNet::NeuralNet(vector<Layer*> layers, Loss *loss) : 
    layers(layers), loss(loss) {}

NeuralNet::NeuralNet() : loss(nullptr) {}

const vector<Layer*>& NeuralNet::getLayers() const {
    return layers;
}

const Loss* NeuralNet::getLoss() const {
    return loss;
}

void NeuralNet::fit(
    const Tensor &features,
    const vector<float> &targets,
    float learningRate,
    float learningDecay,
    size_t numEpochs,
    size_t batchSize,
    ProgressMetric &metric
) {
    build(batchSize, features);
    float initialLR = learningRate;
    avgLosses.resize(numEpochs);
    for (size_t k = 0; k < numEpochs; k++) {
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;

        float avgLoss = runEpoch(features, targets, learningRate, batchSize, metric);

        avgLosses[k] = avgLoss;
        learningRate = initialLR/(1 + learningDecay*k);
    }
    ConsoleUtils::printSepLine();
}

void NeuralNet::build(size_t batchSize, const Tensor &features) {
    size_t numLayers = layers.size();

    maxBatchSize = batchSize;
    vector<size_t> inShape = features.getShape();
    inShape[0] = maxBatchSize;

    for (size_t i = 0; i < numLayers; i++) {
        layers[i]->build(inShape);
        inShape = layers[i]->getBuildOutShape(inShape);
    }
}

float NeuralNet::runEpoch(
    const Tensor &features,
    const vector<float> &targets,
    float learningRate,
    size_t batchSize,
    ProgressMetric &metric
) {
    metric.init();
    size_t numBatches = (targets.size() + batchSize - 1)/batchSize;
    vector<size_t> shuffledIndices = generateShuffledIndices(features);

    for (size_t b = 0; b < numBatches; b++) {
        size_t start = b * batchSize;
        size_t end = min((b + 1) * batchSize, targets.size());
        Batch batch = makeBatch(start, end, features, targets, shuffledIndices);

        forwardPass(batch);
        float batchTotalLoss = loss->calculateTotalLoss(batch.getTargets(), layers.back()->getOutput());
        backprop(batch, learningRate);
        metric.update(batch, loss, layers.back()->getOutput(), batchTotalLoss);
        ConsoleUtils::printProgressBar(metric);
    }

    return metric.getTotalLoss()/targets.size();
}

Batch NeuralNet::makeBatch(
    size_t start,
    size_t end,
    const Tensor &features,
    const vector<float> &targets,
    const vector<size_t> &shuffledIndices
) const {
    size_t batchSize = end - start;
    Batch batch = Batch(layers.size() + 1, batchSize);
    batch.setBatchIndices(start, end, shuffledIndices);
    batch.setBatch(features, targets);

    return batch;
}

void NeuralNet::forwardPass(Batch &batch) {
    Tensor prevActivations = batch.getData();
    size_t numLayers = layers.size();

    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->forward(prevActivations);
        prevActivations = layers[j]->getOutput();
    }
}

void NeuralNet::backprop(Batch &batch, float learningRate) {
    Tensor outputGradients = loss->calculateGradient(batch.getTargets(),layers.back()->getOutput());
    size_t numLayers = (int) layers.size();
    
    for (int i = numLayers - 1; i >= 0; i--) {
        bool isFirstLayer = (i == 0);
        const Tensor &prevActivations = ((i == 0) ? batch.getData() : layers[i-1]->getOutput());
        layers[i]->backprop(prevActivations, learningRate, outputGradients, isFirstLayer);
        outputGradients = layers[i]->getOutputGradient();
    }
}

Tensor NeuralNet::predict(const Tensor &features) {
    forwardPassInference(features);
    return layers.back()->getOutput();
}

void NeuralNet::forwardPassInference(const Tensor& data) {
    Tensor prevActivations = data;
    size_t numLayers = layers.size();
    
    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->forward(prevActivations);
        prevActivations = layers[j]->getOutput();
    }
}

vector<size_t> NeuralNet::generateShuffledIndices(const Tensor &features) const {
    if (features.getShape().size() == 0) {
        return vector<size_t>();
    }

    size_t size = features.getShape()[0];
    vector<size_t> indices(size, 0);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

NeuralNet::~NeuralNet() {
    delete loss;
    size_t numLayers = layers.size();
    for (size_t i = 0; i < numLayers; i++) {
        delete layers[i];
    }
}

void NeuralNet::writeBin(ofstream &modelBin) const {
    uint32_t lossEncoding = loss->getEncoding();
    modelBin.write((char*) &lossEncoding, sizeof(uint32_t));

    uint32_t numActiveLayers = layers.size();
    modelBin.write((char*) &numActiveLayers, sizeof(uint32_t));

    for (uint32_t i = 0; i < numActiveLayers; i++) {
        layers[i]->writeBin(modelBin);
    }
}

void NeuralNet::loadLoss(ifstream &modelBin) {
    uint32_t lossEncoding;
    modelBin.read((char*) &lossEncoding, sizeof(uint32_t));

    if (lossEncoding == Loss::Encodings::MSE) {
        loss = new MSE();
    } else if (lossEncoding == Loss::Encodings::SoftmaxCrossEntropy) {
        loss = new SoftmaxCrossEntropy();
    } else {
        ConsoleUtils::fatalError(
            "Unsupported loss encoding \"" + to_string(lossEncoding) + "\" in model file."
        );
    } 
}

void NeuralNet::loadLayer(ifstream &modelBin) {
    uint32_t layerEncoding;
    modelBin.read((char*) &layerEncoding, sizeof(uint32_t));

    Layer *layer = nullptr;
    if (layerEncoding == Layer::Encodings::DenseLayer) {
        layer = new DenseLayer();
    } else{
        ConsoleUtils::fatalError(
            "Unsupported layer encoding \"" + to_string(layerEncoding) + "\"."
        );
    }

    if (layer) {
        layer->loadFromBin(modelBin);
        layers.push_back(layer);
    }
}

void NeuralNet::loadFromBin(ifstream &modelBin) {
    loadLoss(modelBin);
    uint32_t numActiveLayers;
    modelBin.read((char*) &numActiveLayers, sizeof(uint32_t));

    for (uint32_t i = 0; i < numActiveLayers; i++) {
        loadLayer(modelBin);
    }
}