#define STB_IMAGE_IMPLEMENTATION

#include <stb/stb_image.h>
#include <filesystem>
#include "core/data/ImageData2D.h"
#include "utils/TargetEncoder.h"
#include "utils/ConsoleUtils.h"

using namespace std;
namespace fs = filesystem;

void ImageData2D::read(
    vector<RawImage> &features, 
    vector<float> &targets, 
    const string &path
) {
    vector<string> rawTargets;

    for (const auto &labelDir : fs::directory_iterator(path)) {
        string label = labelDir.path().filename().string();
        for (const auto &image : fs::directory_iterator(labelDir.path())) {
            string imgPath = image.path().string();

            int w, h, c;
            unsigned char *input = stbi_load(
                imgPath.c_str(),
                &w, &h, &c,
                0
            );

            if (!input) {
                ConsoleUtils::fatalError("Could not load image: " + imgPath);
            }

            RawImage rawImage;
            rawImage.width = w;
            rawImage.height = h;
            rawImage.channels = c;
            rawImage.pixels.assign(input, input + (w*h*c));

            features.push_back(rawImage);
            rawTargets.push_back(label);

            stbi_image_free(input);
        }
    }

    if (labelMap.empty()) {
        labelMap = TargetEncoder::createLabelMap(rawTargets);
    }

    targets = TargetEncoder::getClassificationTarget(rawTargets, labelMap);
}

void ImageData2D::readTrain(const string &path) {
    read(trainFeatures, trainTargets, path);
}

void ImageData2D::readTest(const string &path) {
    read(testFeatures, testTargets, path);
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