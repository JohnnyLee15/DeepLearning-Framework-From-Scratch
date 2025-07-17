#include "core/data/ImageData2D.h"
#include "utils/TargetEncoder.h"

void ImageData2D::setTrainFeatures(const Tensor &rawTrainFeatures) {
    trainFeatures = rawTrainFeatures;
}

void ImageData2D::setTestFeatures(const Tensor &rawTestFeatures) {
    testFeatures = rawTestFeatures;
}

void ImageData2D::setTrainTargets(const vector<string> &rawTrainTargets) {
    if (labelMap.empty()) {
        labelMap = TargetEncoder::createLabelMap(rawTrainTargets);
    }

    trainTargets = TargetEncoder::getClassificationTarget(rawTrainTargets, labelMap);
}

void ImageData2D::setTestTargets(const vector<string> &rawTestTargets) {
    if (labelMap.empty()) {
        labelMap = TargetEncoder::createLabelMap(rawTestTargets);
    }

    testTargets = TargetEncoder::getClassificationTarget(rawTestTargets, labelMap);
}

const Tensor& ImageData2D::getTrainFeatures() const {
    return trainFeatures;
}

const Tensor& ImageData2D::getTestFeatures() const {
    return testFeatures;
}

const vector<float>& ImageData2D::getTrainTargets() const {
    return trainTargets;
}

const vector<float>& ImageData2D::getTestTargets() const {
    return testTargets;
}

size_t ImageData2D::getNumTrainSamples() const {
    return trainFeatures.getShape()[0];
}

uint32_t ImageData2D::getEncoding() const {
    return Data::Encodings::Image2D;
}

void ImageData2D::writeBin(ofstream &modelBin) const {
    uint32_t mapSize = labelMap.size();
    modelBin.write((char*) &mapSize, sizeof(uint32_t));
    for (const pair<const string, int > &pair : labelMap) {
        uint32_t keyLen = pair.first.size();
        modelBin.write((char*) &keyLen, sizeof(uint32_t));
        modelBin.write(pair.first.c_str(), keyLen);

        uint32_t mapVal = pair.second;
        modelBin.write((char*) &mapVal, sizeof(uint32_t));
    }
}

void ImageData2D::loadFromBin(ifstream &modelBin) {
    uint32_t mapSize;
    modelBin.read((char*) &mapSize, sizeof(uint32_t));

    for (uint32_t i = 0; i < mapSize; i++) {
        uint32_t keyLen;
        modelBin.read((char*) &keyLen, sizeof(uint32_t));

        string key(keyLen, '\0');
        modelBin.read(key.data(), keyLen);

        uint32_t value;
        modelBin.read((char*) &value, sizeof(uint32_t));

        labelMap[key] = value;
    }
}