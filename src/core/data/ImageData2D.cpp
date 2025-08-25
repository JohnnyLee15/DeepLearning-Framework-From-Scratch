#define STB_IMAGE_IMPLEMENTATION

#include <stb/stb_image.h>
#include <filesystem>
#include "core/data/ImageData2D.h"
#include "utils/TargetEncoder.h"
#include "utils/ConsoleUtils.h"
#include <iostream>
#include "utils/CsvUtils.h"

using namespace std;
namespace fs = filesystem;

ImageData2D::ImageData2D(size_t channels) : channels(channels) {}

ImageData2D::ImageData2D() : channels(0) {}

void ImageData2D::scanDirectory(
    vector<string> &paths,
    vector<string> &labels,
    const string &path
) const {
    ConsoleUtils::loadMessage("Scanning Image Directories.");
    for (const auto &labelDir : fs::directory_iterator(path)) {

        if (!labelDir.is_directory()) continue;

        string label = labelDir.path().filename().string();
        for (const auto &image : fs::directory_iterator(labelDir.path())) {
            paths.push_back(image.path().string());
            labels.push_back(label);
        }
    }
    ConsoleUtils::completeMessage();
}

void ImageData2D::extractImages(
    vector<RawImage> &features, 
    const vector<string> &paths
) const{
    ConsoleUtils::loadMessage("Extracting Images.");
    features.reserve(paths.size());
    for (const string &imgPath : paths) {
        int w, h, c;
        unsigned char *input = stbi_load(
            imgPath.c_str(),
            &w, &h, &c,
            channels
        );

        if (!input) {
            ConsoleUtils::fatalError("Could not load image: " + imgPath);
        }

        RawImage rawImage;
        rawImage.width = w;
        rawImage.height = h;
        rawImage.channels = channels;
        rawImage.pixels.assign(input, input + (w*h*channels));
        features.push_back(rawImage);

        stbi_image_free(input);
    }
    ConsoleUtils::completeMessage();
}

void ImageData2D::extractLabels(vector<float> &targets, const vector<string> &labels) {
    ConsoleUtils::loadMessage("Extracting Targets.");
    if (labelMap.empty()) {
        labelMap = TargetEncoder::createLabelMap(labels);
    }

    targets = TargetEncoder::getClassificationTarget(labels, labelMap);
    ConsoleUtils::completeMessage();
}

void ImageData2D::read(
    vector<RawImage> &features, 
    vector<float> &targets, 
    const string &path
) {
    vector<string> paths;
    vector<string> labels;
    scanDirectory(paths, labels, path);
    extractImages(features, paths);
    extractLabels(targets, labels);
    ConsoleUtils::printSepLine();
}

void ImageData2D::readTrain(const string &path) {
    cout << endl << "📥 Loading training data from: \"" << CsvUtils::trimFilePath(path) << "\"." << endl;
    read(trainFeatures, trainTargets, path);
}

void ImageData2D::readTest(const string &path) {
    cout << endl << "📥 Loading testing data from: \"" << CsvUtils::trimFilePath(path) << "\"." << endl;
    read(testFeatures, testTargets, path);
}

void ImageData2D::clearTrain() {
    trainFeatures.clear();
    vector<float>().swap(trainTargets);
}

void ImageData2D::clearTest() {
    testFeatures.clear();
    vector<float>().swap(testTargets);
}

const vector<RawImage>& ImageData2D::getTrainFeatures() const {
    return trainFeatures;
}

const vector<RawImage>& ImageData2D::getTestFeatures() const {
    return testFeatures;
}

const vector<float>& ImageData2D::getTrainTargets() const {
    return trainTargets;
}

const vector<float>& ImageData2D::getTestTargets() const {
    return testTargets;
}

size_t ImageData2D::getNumTrainSamples() const {
    return trainFeatures.size();
}

Data::Encodings ImageData2D::getEncoding() const {
    return Data::Encodings::Image2D;
}

void ImageData2D::writeBin(ofstream &modelBin) const {
    Data::writeBin(modelBin);

    uint32_t writeChannels = (uint32_t) channels;
    modelBin.write((char*) &writeChannels, sizeof(uint32_t));

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
    uint32_t readChannels;
    modelBin.read((char*) &readChannels, sizeof(uint32_t));
    channels = readChannels;

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

Data* ImageData2D::clone() const {
    return new ImageData2D(*this);
}