#include "core/data/BinLoader.h"

#include <sstream>
#include "utils/ConsoleUtils.h"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include "core/tensor/Tensor.h"
#include "core/data/Batch.h"
#include "utils/BinUtils.h"
#include <iostream>

namespace fs = std::filesystem;

const string BinLoader::DATA_EXTENSION = ".bin";

BinLoader::BinLoader(
    const string &path, 
    Tensor &features,
    vector<float> &targets
) : xShape(features.getShape()) {
    string pathToWrite = BinUtils::addExtension(path, DATA_EXTENSION);
    writeToBin(pathToWrite, features, targets);
    calculateOffsets();
    openData(pathToWrite);

    features.clear();
    vector<float>().swap(targets);
}

BinLoader::BinLoader(const string &path) {
    string pathToOpen= BinUtils::addExtension(path, DATA_EXTENSION);
    openData(pathToOpen);
    readShapeX();
    calculateOffsets();
}

BinLoader::BinLoader() {}

ofstream BinLoader::openOut(const string &path) const {
    if (fs::exists(path)) {
        ConsoleUtils::fatalError("File \"" + path + "\"" + " already exists.");
    }

    BinUtils::checkParentDirs(path);
    ofstream outBin(path, ios::out | ios::binary);

    if (!outBin) {
        ConsoleUtils::fatalError(
            "Could not open \"" + path + "\": " + strerror(errno) + "."
        );
    }

    return outBin;
}

void BinLoader::writeShape(ofstream &outBin, const Tensor &features) const {
    uint32_t rank = (uint32_t) features.getRank();
    vector<size_t> inShape = features.getShape();
    vector<uint32_t> shape(rank);
    for (size_t i = 0; i < rank; i++) {
        shape[i] = (uint32_t) inShape[i];
    }

    outBin.write((char*) &rank, sizeof(uint32_t));
    outBin.write((char*) shape.data(), sizeof(uint32_t) * rank);
}

void BinLoader::writeData(
    ofstream &outBin,
    const Tensor &features,
    const vector<float> &targets
) const {
    outBin.write((char*) features.getFlat().data(), features.getSize() * sizeof(float));
    outBin.write((char*) targets.data(), targets.size() * sizeof(float));
}

void BinLoader::closeOutBin(ofstream &outBin, const string &path) const {
    outBin.close();
    if (!outBin) {
        ConsoleUtils::fatalError(
            "Failed to write to \"" + path + "\". The file may be corrupted."
        );
    }
}

void BinLoader::writeToBin(
    const string &path, 
    const Tensor &features,
    const vector<float> &targets
) const {
    cout << endl << "🚚 Offloading " << targets.size() << " samples to " << path << "." << endl;
    ConsoleUtils::loadMessage("Serializing to binary for streaming and freeing RAM.");

    ofstream outBin = openOut(path);
    writeShape(outBin, features);
    writeData(outBin, features, targets);
    closeOutBin(outBin, path);

    ConsoleUtils::completeMessage();
    ConsoleUtils::printSepLine();
}

void BinLoader::openData(const string &path) {
    dataBin.open(path, ios::in | ios::binary);
    if (!dataBin) {
        ConsoleUtils::fatalError(
            "Could not open \"" + path + "\": " + strerror(errno) + "."
        );
    }
}

void BinLoader::readShapeX() {
    dataBin.clear();

    uint32_t rank;
    dataBin.read((char*) &rank, sizeof(uint32_t));

    vector<uint32_t> shape(rank);
    xShape = vector<size_t>(rank);

    dataBin.read((char*) shape.data(), sizeof(uint32_t) * rank);

    for (size_t i = 0; i < rank; i++) {
        xShape[i] = (uint32_t) shape[i];
    }
}

void BinLoader::calculateOffsets() {
    elementsPerSample = xSize() / xShape[0];
    rowBytes = sizeof(float) * elementsPerSample;
    featureOffBase = (xShape.size() + 1) * sizeof(uint32_t);
    targetOffBase = featureOffBase + rowBytes * xShape[0];
}

size_t BinLoader::sampleCount() const {
    if (xShape.size() == 0) {
        return 0;
    }
    
    return xShape[0];
}

const vector<size_t>& BinLoader::getShapeX() const {
    return xShape;
}

size_t BinLoader::xSize() const {
    if (xShape.size() == 0 || xShape[0] == 0) {
        return 0;
    }

    size_t dims = xShape.size();
    size_t size = 1;
    for (size_t i = 0; i < dims; i++) {
        size *= xShape[i];
    }

    return size;
}

void BinLoader::fillBatch(Batch &batch) const {
    size_t batchSize = batch.getBatchSize();

    vector<float> &xBatchFlat = batch.getData().getFlat();
    vector<float> &yBatchFlat = batch.getTargets().getFlat();
    const vector<size_t> &indices = batch.getIndices();

    for (size_t i = 0; i < batchSize; i++) {
        dataBin.clear();
        uint64_t rdIdx = (uint64_t) indices[i];

        uint64_t xOff = featureOffBase + rdIdx * rowBytes;
        dataBin.seekg(xOff, ios::beg);

        float *xBatch = xBatchFlat.data() + (i*elementsPerSample);
        dataBin.read((char*) xBatch, rowBytes);

        uint64_t yOff = targetOffBase + (rdIdx * sizeof(float));
        dataBin.seekg(yOff, ios::beg);

        float *yBatch = yBatchFlat.data() + i;
        dataBin.read((char*) yBatch, sizeof(float));
    }

    if (!dataBin) {
        ConsoleUtils::fatalError("Read error from data bin file.");
    }
}

void BinLoader::fillInfBatch(
    Tensor &batch,
    size_t start,
    size_t batchSize
) const {
    vector<float> &xBatchFlat = batch.getFlat();

    dataBin.clear();
    uint64_t xOff = featureOffBase + start * rowBytes;
    dataBin.seekg(xOff, ios::beg);
    dataBin.read((char*) xBatchFlat.data(), batchSize * rowBytes);

    if (!dataBin) {
        ConsoleUtils::fatalError("Read error from data bin file.");
    }
}

vector<float> BinLoader::loadTargets() const {
    dataBin.clear();

    const size_t N = sampleCount();
    vector<float> targets(N);

    dataBin.seekg(targetOffBase, ios::beg);
    dataBin.read((char*) targets.data(), N * sizeof(float));

    if (!dataBin) {
        ConsoleUtils::fatalError("BinLoader::loadTargets: short read / read error.");
    }
    return targets;
}