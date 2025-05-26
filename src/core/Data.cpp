#include "core/Data.h"
#include "utils/MatrixUtils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

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

const vector<vector<double> >& Data::getTrainFeatures() const {
    return trainFeatures;
}

const vector<vector<double> >& Data::getTestFeatures() const {
    return testFeatures;
}

const vector<int>& Data::getTrainTarget() const {
    return trainTarget;
}

const vector<int>& Data::getTestTarget() const {
    return testTarget;
}

size_t Data::getTrainFeatureSize() const {
    return trainFeatures.size();
}

void Data::checkFile(string filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Failed to read " << filename << endl;
        exit(1);
    }
}

void Data::parseLine(
    string line, 
    vector<vector<double> > &features, 
    vector<int> &target, 
    int targetIdx
) {

    stringstream lineParser(line);
    string token;

    int i = 0;
    vector<double> sample;
    while(getline(lineParser, token, ',')) {
        if (i == targetIdx) {
            target.push_back(stoi(token)); 
        } else {
            sample.push_back(stod(token));
        }
        i++;
    }

    features.push_back(sample);
}

void Data::setData(
    vector<vector<double> > &features, 
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

    ifstream file(filename);
    string line;
    vector<vector<double> > features;
    vector<int> target;

    // Read Header
    getline(file, line);
    
    while(getline(file, line)) {
        parseLine(line, features, target, targetIdx);
    }
    setData(features, target, isTrainData);
    isDataLoaded = true;
}

void Data::minmaxNormalizeColumn(
    vector<vector<double> > &features, 
    double minVal, 
    double maxVal, 
    int colIdx
) {
    size_t size = features.size();
    double range = maxVal - minVal;
    
    if (range == 0) {
        range = 1.0;
    }
    for (size_t i = 0; i < size; i++) {
        features[i][colIdx] = (features[i][colIdx] - minVal) / (range);
    }
}

void Data::getMinMaxColumn(
    vector<vector<double> > &features, 
    double &minVal, 
    double &maxVal, 
    int colIdx
) {
    size_t size = features.size();
    minVal = MatrixUtils::INF;
    maxVal = -MatrixUtils::INF;
    
    for (size_t i = 0; i < size; i++) {
        if (features[i][colIdx] < minVal) {
            minVal = features[i][colIdx];
        }
        if (features[i][colIdx] > maxVal) {
            maxVal = features[i][colIdx];
        }
    }
}

void Data::minmaxData() {
    double minVal, maxVal;
    size_t numCols = trainFeatures[0].size();

    for (size_t j = 0; j < numCols; j++) {
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

void Data::normalizeGreyScale(vector<vector<double> > &features) {
    size_t numRows = features.size();
    for (size_t i = 0; i < numRows; i++) {

        size_t numCols = features[i].size();
        for (size_t j = 0; j < numCols; j++) {
            features[i][j] /= MAX_GREYSCALE_VALUE;
        }
    }
}

void Data::minmaxGreyScale() {
    normalizeGreyScale(trainFeatures);
    normalizeGreyScale(testFeatures);
}

vector<int> Data::generateShuffledIndices() const {
    size_t size = trainFeatures.size();
    vector<int> indices(size, -1);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

