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
#include "core/layers/Dropout.h"
#include "core/layers/GlobalAveragePooling2D.h"
#include <cerrno>
#include "utils/EarlyStop.h"
#include "core/data/Dataset.h"

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
      inputShape(other.inputShape),
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
    ProgressMetric &metric,
    const Tensor &xVal,
    const vector<float> &yVal,
    EarlyStop *stop
) {
    Dataset train(features, targets);
    Dataset val(xVal, yVal);
    fitInternal(
        train, val, learningRate, learningDecay,
        numEpochs, batchSize, metric, stop
    );
}

void NeuralNet::fit(
    const BinLoader &train,
    float learningRate,
    float learningDecay,
    size_t numEpochs,
    size_t batchSize,
    ProgressMetric &metric,
    const BinLoader &val,
    EarlyStop *stop
) {
    Dataset trainBin(train);
    Dataset valBin(val);
    fitInternal(
        trainBin, valBin, learningRate, learningDecay,
        numEpochs, batchSize, metric, stop
    );
}

void NeuralNet::fitInternal(
    const Dataset &train,
    const Dataset &val,
    float learningRate,
    float learningDecay,
    size_t numEpochs,
    size_t batchSize,
    ProgressMetric &metric,
    EarlyStop *stop
) {
    if (batchSize == 0 || train.sampleCount() == 0)
        return;

    float initialLR = learningRate;
    avgLosses.resize(numEpochs);
    build(batchSize, train.xShape());
    
    bool stopEpochs = false;
    for (size_t k = 0; k < numEpochs && !stopEpochs; k++) {
        cout << endl << "Epoch: " << k+1 << "/" << numEpochs << endl;

        float avgLoss = runEpoch(train, learningRate, batchSize, metric);
        stopEpochs = validateEpoch(val, metric, stop, k);

        avgLosses[k] = avgLoss;
        learningRate = initialLR/(1 + learningDecay*k);
    }

    tryBestWeights(stop);
    ConsoleUtils::printSepLine();
}

void NeuralNet::build(
    size_t batchSize, 
    const vector<size_t> &inShape, 
    bool isInference
) {
    vector<size_t> inShapeLocal = inShape;
    inShapeLocal[0] = batchSize;
    if (inShapeLocal == inputShape && !isInference)
        return;

    size_t numLayers = layers.size();
    maxBatchSize = batchSize;
    inputShape = inShapeLocal;

    for (size_t i = 0; i < numLayers; i++) {
        layers[i]->build(inShapeLocal, isInference);
        inShapeLocal = layers[i]->getBuildOutShape(inShapeLocal);
    }

    if (!isInference) {
        vector<size_t> lossShape = layers.back()->getOutput().getShape();
        lossShape[0] = maxBatchSize;
        dL = Tensor(lossShape);

    } else {
        dL.clear();
    }
}

float NeuralNet::runEpoch(
    const Dataset &train,
    float learningRate,
    size_t batchSize,
    ProgressMetric &metric
) {
    size_t N = train.sampleCount();
    metric.init(N);
    size_t numBatches = (N + batchSize - 1)/batchSize;
    vector<size_t> shuffledIndices = generateShuffledIndices(N);
    Batch batch(batchSize, train.xShape());

    for (size_t b = 0; b < numBatches; b++) {
        size_t start = b * batchSize;
        size_t end = min((b + 1) * batchSize, N);
        train.fillBatch(batch, start, end, shuffledIndices);
        
        fitBatch(batch, learningRate);
        float batchTotalLoss = loss->calculateTotalLoss(batch.getTargets(), layers.back()->getOutput());
        
        metric.update(batch, loss, layers.back()->getOutput(), batchTotalLoss);
        ConsoleUtils::printProgressBar(metric);
    }

    return metric.getTotalLoss()/N;
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
    if (batch.getBatchSize() != dL.getShape()[0]) {
        reShapeDL(batch.getBatchSize());
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

bool NeuralNet::validateEpoch(
    const Dataset &val,
    ProgressMetric &metric,
    EarlyStop *stop,
    size_t epoch
) {
    size_t N = val.sampleCount();
    if (N == 0)
        return false;

    metric.init(N);
    size_t numBatches = (N + maxBatchSize - 1)/maxBatchSize;

    vector<size_t> indices(N);
    iota(indices.begin(), indices.end(), 0);
    Batch batch(maxBatchSize, val.xShape());

    for (size_t b = 0; b < numBatches; b++) {
        size_t start = b * maxBatchSize;
        size_t end = min((b + 1) * maxBatchSize, N);
        val.fillBatch(batch, start, end, indices);

        forwardPassInference(batch.getData());
        float batchTotalLoss = loss->calculateTotalLoss(batch.getTargets(), layers.back()->getOutput());

        metric.update(batch, loss, layers.back()->getOutput(), batchTotalLoss);
    }

    ConsoleUtils::printValidationMetrics(metric);

    if (stop == nullptr)
        return false;

    return stop->shouldStop(metric.getAvgLoss(), epoch, *this);
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

Tensor NeuralNet::predictInternal(const Dataset &x) {
    build(INFERENCE_BATCH_SIZE, x.xShape(), true);

    size_t numSamples = x.sampleCount();
    size_t numBatches = (numSamples + INFERENCE_BATCH_SIZE - 1) / INFERENCE_BATCH_SIZE;
    vector<size_t> batchShape = x.xShape();
    batchShape[0] = INFERENCE_BATCH_SIZE;

    Tensor output;
    Tensor batch(batchShape);

    for (size_t i = 0; i < numBatches; i++) {
        size_t start = i * INFERENCE_BATCH_SIZE;
        size_t end = min((i + 1) * INFERENCE_BATCH_SIZE, numSamples);
        size_t batchSize = end - start;

        x.fillInferenceBatch(batch, start, batchSize);
        forwardPassInference(batch);
        cpyBatchToOutput(start, batchSize, i, numSamples, output);
    }

    return output;
}

Tensor NeuralNet::predict(const Tensor &x) {
    Dataset xMem(x);
    return predictInternal(xMem);
}

Tensor NeuralNet::predict(const BinLoader &x) {
    Dataset xBin(x);
    return predictInternal(x);
}

vector<size_t> NeuralNet::generateShuffledIndices(size_t sampleCount) const {
    if (sampleCount == 0) {
        return vector<size_t>();
    }

    vector<size_t> indices(sampleCount, 0);
    
    for (size_t i = 0; i < sampleCount; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

NeuralNet::~NeuralNet() {
    delete loss;
    deleteLayers();
}

void NeuralNet::printParamCount(const BinLoader &train) {
    Dataset trainData(train);
    build(INFERENCE_BATCH_SIZE, trainData.xShape());
    printParamCount();
}

void NeuralNet::printParamCount(const Tensor &train) {
    Dataset trainData(train);
    build(INFERENCE_BATCH_SIZE, trainData.xShape());
    printParamCount();
}

void NeuralNet::printParamCount() {
    size_t numLayers = layers.size();
    size_t params = 0;
    for (size_t i = 0; i < numLayers; i++) {
        params += layers[i]->paramCount();
    }

    cout << endl << "📊 Number of Model Parameters: " 
         << ConsoleUtils::integerWithCommas(params);

    if (params == 0) {
        cout << " (model not built yet)";
    }

    cout << endl;
    ConsoleUtils::printSepLine();
}

void NeuralNet::deleteLayers() {
    size_t numLayers = layers.size();
    for (size_t i = 0; i < numLayers; i++) {
        delete layers[i];
    }
    layers.clear();
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

void NeuralNet::saveBestWeights(ofstream &modelBin) const {
    size_t numActiveLayers = layers.size();
    for (uint32_t i = 0; i < numActiveLayers; i++) {
        layers[i]->writeBin(modelBin);
    }
}

void NeuralNet::tryBestWeights(EarlyStop *stop) {
    if (stop == nullptr || !stop->hasBestWeights())
        return;

    BinUtils::loadBestWeights(stop->getBestWeightPath(), *this);
    stop->deleteBestWeights();
}

void NeuralNet::loadBestWeights(ifstream &modelBin) {
    size_t numLayers = layers.size();
    vector<Layer*> temp;
    temp.reserve(numLayers);
    
    for (uint32_t i = 0; i < numLayers; i++) {
        Layer *layer = loadLayer(modelBin);

        if (layer == nullptr) {
            ConsoleUtils::printError(
                "Best weights file invalid/truncated at layer " + std::to_string(i)
                + ". Returning to previous weights."
            );

            for (Layer *layer : temp) {
                delete layer;
            }
            return;
        }

        temp.push_back(layer);
    }

    deleteLayers();
    layers.swap(temp);
}

Layer* NeuralNet::loadLayer(ifstream &modelBin) {
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
    } else if (layerEncoding == Layer::Encodings::Dropout) {
        layer = new Dropout();
    } else if (layerEncoding == Layer::Encodings::GlobalAveragePooling2D) {
        layer = new GlobalAveragePooling2D();
    } 

    if (layer) {
        layer->loadFromBin(modelBin);
    }

    return layer;
}

void NeuralNet::loadFromBin(ifstream &modelBin) {
    loadLoss(modelBin);
    uint32_t numActiveLayers;
    modelBin.read((char*) &numActiveLayers, sizeof(uint32_t));

    layers.reserve(numActiveLayers);
    for (uint32_t i = 0; i < numActiveLayers; i++) {
        Layer *layer = loadLayer(modelBin);

        if (layer == nullptr) ConsoleUtils::fatalError("Failed to load layer.");

        layers.push_back(layer);
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

