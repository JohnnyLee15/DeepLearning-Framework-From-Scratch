#include "core/Data.h"
#include "utils/VectorUtils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <omp.h>

random_device Data::rd;
mt19937 Data::generator(Data::rd());

Data::Data(): isDataLoaded(false) {}

const double Data::MAX_GREYSCALE_VALUE = 255.0;

void Data::readTrain(string filename, int targetIdx) {
    readData(filename, true, targetIdx);
}

void Data::readTest(string filename, int targetIdx) {
    readData(filename, false, targetIdx);
}

const Matrix& Data::getTrainFeatures() const {
    return trainFeatures;
}

const Matrix& Data::getTestFeatures() const {
    return testFeatures;
}

const vector<int>& Data::getTrainTarget() const {
    return trainTarget;
}

const vector<int>& Data::getTestTarget() const {
    return testTarget;
}

size_t Data::getTrainFeatureSize() const {
    return trainFeatures.getNumRows();
}

void Data::checkFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Failed to read " << filename << endl;
        exit(1);
    }
}

void Data::parseLine(
    const string &line, 
    vector<double> &sampleRow, 
    int &target, 
    int targetIdx
) {

    stringstream lineParser(line);
    string token;

    int i = 0;
    vector<double> sample;
    while(getline(lineParser, token, ',')) {
        if (i == targetIdx) {
            target = stoi(token); 
        } else {
            sample.push_back(stod(token));
        }
        i++;
    }

    sampleRow = sample;
}

void Data::collectLines(
    vector<string> &lines,
    string filename
) {
    ifstream file(filename);
    string line;

    // Read Header 
    getline(file, line);
    while(getline(file, line)) {
        lines.push_back(line);
    }
}

void Data::setData(
    const Matrix &features, 
    vector<int> &target,
    bool isTrainData
) {

    if (isTrainData) {
        trainFeatures = features;
        trainTarget = target;
    } else {
        testFeatures = features;
        testTarget = target;
    }
}

void Data::readData(string filename, bool isTrainData, int targetIdx) {
    checkFile(filename);

    vector<string> lines;
    collectLines(lines, filename);

    size_t numSamples = lines.size();
    vector<vector<double> > features(numSamples);
    vector<int> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        parseLine(lines[i], features[i], target[i], targetIdx);
    }

    setData(Matrix(features), target, isTrainData);
    isDataLoaded = true;
}

void Data::minmaxNormalizeColumn(
    Matrix &features, 
    double minVal, 
    double maxVal, 
    int colIdx
) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();
    vector<double> &featuresFlat = features.getFlat();
    double range = maxVal - minVal;
    
    if (range == 0.0) {
        range = 1.0;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        featuresFlat[i*numCols + colIdx] = (featuresFlat[i*numCols + colIdx] - minVal) / range;
    }
}

void Data::getMinMaxColumn(
    const Matrix &features, 
    double &minVal, 
    double &maxVal, 
    int colIdx
) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();
    const vector<double> &featuresFlat = features.getFlat();
    minVal = VectorUtils::INF;
    maxVal = -VectorUtils::INF;
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < numRows; i++) {
        double val = featuresFlat[i*numCols + colIdx];
        if (val < minVal) {
            minVal = val;
        }
        if (val > maxVal) {
            maxVal = val;
        }
    }
}

void Data::minmaxData() {
    size_t numCols = trainFeatures.getNumCols();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        double minVal, maxVal;
        getMinMaxColumn(trainFeatures, minVal, maxVal, j);
        minmaxNormalizeColumn(trainFeatures, minVal, maxVal, j);
        minmaxNormalizeColumn(testFeatures, minVal, maxVal, j);
    }
}

void Data::minmax() {
    if (isDataLoaded) {
        minmaxData();
    }
}

void Data::normalizeGreyScale(Matrix &features) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();

    vector<double> &featuresFlat = features.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            featuresFlat[i*numCols + j] /= MAX_GREYSCALE_VALUE;
        }
    }
}

void Data::minmaxGreyScale() {
    normalizeGreyScale(trainFeatures);
    normalizeGreyScale(testFeatures);
}

vector<int> Data::generateShuffledIndices() const {
    size_t size = trainFeatures.getNumRows();
    vector<int> indices(size, -1);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

