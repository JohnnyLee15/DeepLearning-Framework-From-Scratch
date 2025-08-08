#include "core/model/NeuralNet.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils/ConsoleUtils.h"
#include "core/losses/Loss.h"
#include "core/data/Batch.h"
#include "core/activations/Activation.h"
#include "utils/BinUtils.h"
#include "core/metrics/ProgressMetric.h"
#include "core/losses/MSE.h"
#include "core/losses/SoftmaxCrossEntropy.h"
#include "core/layers/Dense.h"
#include "core/layers/Conv2D.h"
#include "core/layers/MaxPooling2D.h"
#include "core/layers/Flatten.h"
#include "core/gpu/GpuEngine.h"
#include <cstring>

const size_t NeuralNet::INFERENCE_BATCH_SIZE = 8;

random_device NeuralNet::rd;
mt19937 NeuralNet::generator(NeuralNet::rd());

NeuralNet::NeuralNet(vector<Layer*> layers, Loss *loss) : 
    layers(layers), loss(loss) {}

NeuralNet::NeuralNet() : loss(nullptr) {}

NeuralNet::NeuralNet(const NeuralNet &other)
    : avgLosses(other.avgLosses),
      loss(other.loss ? other.loss->clone() : nullptr),
      maxBatchSize(other.maxBatchSize),
      dL(other.dL)
{
    layers.reserve(other.layers.size());
    for (const Layer *layer : other.layers) {
        layers.push_back(layer->clone());
    }
}

NeuralNet* NeuralNet::clone() const {
    return new NeuralNet(*this);
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

void NeuralNet::build(size_t batchSize, const Tensor &features, bool isInference) {
    size_t numLayers = layers.size();
    maxBatchSize = batchSize;
    vector<size_t> inShape = features.getShape();
    inShape[0] = maxBatchSize;

    for (size_t i = 0; i < numLayers; i++) {
        layers[i]->build(inShape, isInference);
        inShape = layers[i]->getBuildOutShape(inShape);
    }

    if (!isInference) {
        vector<size_t> lossShape = layers.back()->getOutput().getShape();
        lossShape[0] = maxBatchSize;
        dL = Tensor(lossShape);

    } else {
        dL = Tensor();
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
        
        fitBatch(batch, learningRate);
        float batchTotalLoss = loss->calculateTotalLoss(batch.getTargets(), layers.back()->getOutput());
        
        metric.update(batch, loss, layers.back()->getOutput(), batchTotalLoss);
        ConsoleUtils::printProgressBar(metric);
    }

    return metric.getTotalLoss()/targets.size();
}

void NeuralNet::fitBatch(const Batch &batch, float learningRate) {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            fitBatchGpu(batch, learningRate);
        #endif
    } else {
        forwardPass(batch.getData());
        backprop(batch, learningRate);
    }
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

void NeuralNet::forwardPass(const Tensor &batch) {
    const Tensor *prevActivations = &batch;
    size_t numLayers = layers.size();

    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->forward(*prevActivations);
        prevActivations = &layers[j]->getOutput();
    }
}

void NeuralNet::reShapeDL(size_t currBatchSize) {
    if (dL.getSize() == 0)
        return;

    vector<size_t> lossShape = layers.back()->getOutput().getShape();
    lossShape[0] = currBatchSize;
    dL.reShapeInPlace(lossShape);
}

void NeuralNet::backprop(const Batch &batch, float learningRate) {
    if (batch.getSize() != dL.getShape()[0]) {
        reShapeDL(batch.getSize());
    }

    loss->calculateGradient(batch.getTargets(),layers.back()->getOutput(), dL);
    size_t numLayers = (int) layers.size();
    
    Tensor *grad = &dL;
    for (int i = numLayers - 1; i >= 0; i--) {
        bool isFirstLayer = (i == 0);
        const Tensor &prevActivations = ((i == 0) 
            ? batch.getData() 
            : layers[i-1]->getOutput());

        layers[i]->backprop(prevActivations, learningRate, *grad, isFirstLayer);

        grad = &layers[i]->getOutputGradient();
    }
}

Tensor NeuralNet::makeInferenceBatch(
    size_t start,
    size_t batchSize,
    size_t sampleFloats,
    const Tensor &features
) const {
    vector<size_t> batchShape = features.getShape();
    batchShape[0] = batchSize;
    Tensor batch = Tensor(batchShape);

    size_t sampleStartFloat = start * sampleFloats;
    size_t bytes = batchSize * sampleFloats * sizeof(float);
    memcpy(batch.getFlat().data(), features.getFlat().data() + sampleStartFloat, bytes);

    return batch;
}

void NeuralNet::forwardPassInference(const Tensor &batch) {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            forwardPassGpuSync(batch);
        #endif
    } else {
        forwardPass(batch);
    }
}

void NeuralNet::cpyBatchToOutput(
    size_t start,
    size_t batchSize,
    size_t batchIdx,
    size_t numSamples,
    const Tensor &batch,
    Tensor &output
) const {
    const Tensor &endLayerOutput = layers.back()->getOutput();
    if (batchIdx == 0){
        vector<size_t> outputShape = endLayerOutput.getShape();
        outputShape[0] = numSamples;
        output = Tensor(outputShape);
    }

    size_t outputFloats = endLayerOutput.getSize() / batchSize;
    size_t outputStartFloat = start * outputFloats;
    size_t outBytes = batchSize * outputFloats * sizeof(float);
    memcpy(output.getFlat().data() + (outputStartFloat), endLayerOutput.getFlat().data(), outBytes);
}

Tensor NeuralNet::predict(const Tensor &features) {
    build(INFERENCE_BATCH_SIZE, features, true);

    size_t numSamples = features.getShape()[0];
    size_t numBatches = (numSamples + INFERENCE_BATCH_SIZE - 1) / INFERENCE_BATCH_SIZE;
    size_t sampleFloats = features.getSize() / numSamples;

    Tensor output;
    for (size_t i = 0; i < numBatches; i++) {
        size_t start = i * INFERENCE_BATCH_SIZE;
        size_t end = min((i + 1) * INFERENCE_BATCH_SIZE, numSamples);
        size_t batchSize = end - start;

        Tensor batch = makeInferenceBatch(start, batchSize, sampleFloats, features);
        forwardPassInference(batch);
        cpyBatchToOutput(start, batchSize, i, numSamples, batch, output);
    }

    return output;
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
    if (layerEncoding == Layer::Encodings::Dense) {
        layer = new Dense();
    } else if (layerEncoding == Layer::Encodings::Conv2D) {
        layer = new Conv2D();
    } else if (layerEncoding == Layer::Encodings::MaxPooling2D) {
        layer = new MaxPooling2D;
    } else if (layerEncoding == Layer::Encodings::Flatten) {
        layer = new Flatten();
    } else {
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