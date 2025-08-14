#include "utils/DataSplitter.h"
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>

Split DataSplitter::stratifiedSplit(
    const Tensor &x,
    const Tensor &y,
    float valRatio
) {
    unordered_map<size_t, vector<size_t>> indicesMap;

    size_t size = y.getSize();

    const vector<float> &yFlat = y.getFlat();
    const vector<float> &xFlat = x.getFlat();
    size_t sampleFloats = x.getSize() / x.getShape()[0];

    for (size_t i = 0; i < size; i++) {
        size_t label = (size_t) yFlat[i];
        indicesMap[label].push_back(i);
    }

    random_device rd;
    mt19937 gen(rd());

    size_t nVal = 0;
    size_t nTrain = 0;
    for (pair<const size_t, vector<size_t>> &keyVal : indicesMap) {
        const vector<size_t> &indices = keyVal.second;
        size_t valToAdd = (size_t)(valRatio * indices.size());
        nVal += valToAdd;
        nTrain += indices.size() - valToAdd;
    }

    Split split;
    size_t trainFloats = sampleFloats * nTrain;
    size_t valFloats = sampleFloats * nVal;

    vector<float> &xTrain = split.xTrain.getFlat();
    vector<float> &xVal = split.xVal.getFlat();
    vector<float> &yTrain = split.yTrain.getFlat();
    vector<float> &yVal = split.yVal.getFlat();

    xTrain.reserve(trainFloats);
    xVal.reserve(valFloats);
    yTrain.reserve(nTrain);
    yVal.reserve(nVal);

    vector<size_t> xTrainShape = x.getShape();
    vector<size_t> xValShape = x.getShape();
    vector<size_t> yTrainShape = y.getShape();
    vector<size_t> yValShape = y.getShape();

    xTrainShape[0] = nTrain;
    yTrainShape[0] = nTrain;
    xValShape[0] = nVal;
    yValShape[0] = nVal;

    split.xTrain.reShapeInPlace(xTrainShape);
    split.yTrain.reShapeInPlace(yTrainShape);
    split.xVal.reShapeInPlace(xValShape);
    split.yVal.reShapeInPlace(yValShape);

    for (pair<const size_t, vector<size_t>> &keyVal : indicesMap) {
        vector<size_t> &indices = keyVal.second;

        shuffle(indices.begin(), indices.end(), gen);

        size_t numIndices = indices.size();
        size_t valIndicesEnd = (size_t)(valRatio * numIndices);

        for (size_t i = 0; i < valIndicesEnd; i++) {
            size_t idx = indices[i] * sampleFloats;
            xVal.insert(xVal.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
            yVal.push_back(yFlat[indices[i]]);
        }

        for (size_t i = valIndicesEnd; i < numIndices; i++) {
            size_t idx = indices[i] * sampleFloats;
            xTrain.insert(xTrain.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
            yTrain.push_back(yFlat[indices[i]]);
        }
    }

    return split;
}

Split DataSplitter::randomSplit(
    const Tensor &x,
    const Tensor &y,
    float valRatio
) {

    size_t size = y.getSize();

    const vector<float> &yFlat = y.getFlat();
    const vector<float> &xFlat = x.getFlat();
    size_t sampleFloats = x.getSize() / x.getShape()[0];

    size_t nVal = (size_t) (valRatio * size);
    size_t nTrain = size - nVal;
    
    random_device rd;
    mt19937 gen(rd());

    vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0); 
    std::shuffle(indices.begin(), indices.end(), gen);

    Split split;
    size_t trainFloats = sampleFloats * nTrain;
    size_t valFloats = sampleFloats * nVal;

    vector<float> &xTrain = split.xTrain.getFlat();
    vector<float> &xVal = split.xVal.getFlat();
    vector<float> &yTrain = split.yTrain.getFlat();
    vector<float> &yVal = split.yVal.getFlat();

    xTrain.reserve(trainFloats);
    xVal.reserve(valFloats);
    yTrain.reserve(nTrain);
    yVal.reserve(nVal);

    vector<size_t> xTrainShape = x.getShape();
    vector<size_t> xValShape = x.getShape();
    vector<size_t> yTrainShape = y.getShape();
    vector<size_t> yValShape = y.getShape();

    xTrainShape[0] = nTrain;
    yTrainShape[0] = nTrain;
    xValShape[0] = nVal;
    yValShape[0] = nVal;

    split.xTrain.reShapeInPlace(xTrainShape);
    split.yTrain.reShapeInPlace(yTrainShape);
    split.xVal.reShapeInPlace(xValShape);
    split.yVal.reShapeInPlace(yValShape);

    for (size_t i = 0; i < nVal; i++) {
        size_t idx = indices[i] * sampleFloats;
        xVal.insert(xVal.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
        yVal.push_back(yFlat[indices[i]]);
    }

    for (size_t i = nVal; i < size; i++) {
        size_t idx = indices[i] * sampleFloats;
        xTrain.insert(xTrain.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
        yTrain.push_back(yFlat[indices[i]]);
    }

    return split;
}