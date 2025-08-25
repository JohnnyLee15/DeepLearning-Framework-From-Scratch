#include "utils/DataSplitter.h"
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "utils/ConsoleUtils.h"
#include <iostream>

float DataSplitter::clampRatio(float ratio) {
    return min(max(ratio, 0.0f), 0.999999f);
}

Split DataSplitter::prepareSplit(
    size_t sampleFloats,
    size_t nTrain,
    size_t nVal,
    const Tensor& x
) {
    Split split;
    size_t trainFloats = sampleFloats * nTrain;
    size_t valFloats = sampleFloats * nVal;

    vector<size_t> xTrainShape = x.getShape();
    vector<size_t> xValShape = x.getShape();

    xTrainShape[0] = nTrain;
    xValShape[0] = nVal;

    split.xTrain.reShapeInPlace(xTrainShape);
    split.xVal.reShapeInPlace(xValShape);

    split.xTrain.getFlat().reserve(trainFloats);
    split.xVal.getFlat().reserve(valFloats);
    split.yTrain.reserve(nTrain);
    split.yVal.reserve(nVal);

    return split;
}

Split DataSplitter::stratifiedSplit(
    const Tensor &x,
    const vector<float> &y,
    float valRatio
) {
    valRatio = clampRatio(valRatio);
    unordered_map<size_t, vector<size_t>> indicesMap;

    size_t size = y.size();

    const vector<float> &xFlat = x.getFlat();
    size_t sampleFloats = x.getSize() / x.getShape()[0];

    for (size_t i = 0; i < size; i++) {
        size_t label = (size_t) y[i];
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

    cout << endl << "✂️  Splitting " << size << " samples: "
         << nTrain << " | " << nVal << endl;
    ConsoleUtils::loadMessage("Splitting data with stratification.");

    Split split = prepareSplit(sampleFloats, nTrain, nVal, x);

    vector<float> &xTrain = split.xTrain.getFlat();
    vector<float> &xVal = split.xVal.getFlat();
    vector<float> &yTrain = split.yTrain;
    vector<float> &yVal = split.yVal;

    for (pair<const size_t, vector<size_t>> &keyVal : indicesMap) {
        vector<size_t> &indices = keyVal.second;

        shuffle(indices.begin(), indices.end(), gen);

        size_t numIndices = indices.size();
        size_t valIndicesEnd = (size_t)(valRatio * numIndices);

        for (size_t i = 0; i < valIndicesEnd; i++) {
            size_t idx = indices[i] * sampleFloats;
            xVal.insert(xVal.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
            yVal.push_back(y[indices[i]]);
        }

        for (size_t i = valIndicesEnd; i < numIndices; i++) {
            size_t idx = indices[i] * sampleFloats;
            xTrain.insert(xTrain.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
            yTrain.push_back(y[indices[i]]);
        }
    }

    ConsoleUtils::completeMessage();
    ConsoleUtils::printSepLine();
    
    return split;
}

Split DataSplitter::randomSplit(
    const Tensor &x,
    const vector <float> &y,
    float valRatio
) {
    

    valRatio = clampRatio(valRatio);
    size_t size = y.size();

    const vector<float> &xFlat = x.getFlat();
    size_t sampleFloats = x.getSize() / x.getShape()[0];

    size_t nVal = (size_t) (valRatio * size);
    size_t nTrain = size - nVal;

    cout << endl << "✂️  Splitting " << size << " samples: "
         << nTrain << " | " << nVal << endl;
    ConsoleUtils::loadMessage("Randomly splitting data.");

    random_device rd;
    mt19937 gen(rd());

    vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0); 
    std::shuffle(indices.begin(), indices.end(), gen);

    Split split = prepareSplit(sampleFloats, nTrain, nVal, x);

    vector<float> &xTrain = split.xTrain.getFlat();
    vector<float> &xVal = split.xVal.getFlat();
    vector<float> &yTrain = split.yTrain;
    vector<float> &yVal = split.yVal;

    for (size_t i = 0; i < nVal; i++) {
        size_t idx = indices[i] * sampleFloats;
        xVal.insert(xVal.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
        yVal.push_back(y[indices[i]]);
    }

    for (size_t i = nVal; i < size; i++) {
        size_t idx = indices[i] * sampleFloats;
        xTrain.insert(xTrain.end(), xFlat.begin() + idx, xFlat.begin() + idx + sampleFloats);
        yTrain.push_back(y[indices[i]]);
    }

    ConsoleUtils::completeMessage();
    ConsoleUtils::printSepLine();

    return split;
}

void Split::clear() {
    xTrain.clear();
    xVal.clear();
    vector<float>().swap(yTrain);
    vector<float>().swap(yVal);
}