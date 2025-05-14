#include "Data.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

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

const vector<double>& Data::getTrainTarget() const {
    return trainTarget;
}

const vector<double>& Data::getTestTarget() const {
    return testTarget;
}

int Data::getTrainFeatureSize() const {
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
    vector<double> &target, 
    int targetIdx
) {

    stringstream lineParser(line);
    string token;

    int i = 0;
    vector<double> sample;
    while(getline(lineParser, token, ',')) {
        double value = stod(token);
        if (i == targetIdx) {
            target.push_back(value); 
        } else {
            sample.push_back(value);
        }
        i++;
    }

    features.push_back(sample);
}

void Data::setData(
    vector<vector<double> > &features, 
    vector<double> &target,
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
    vector<double> target;

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
    double range = maxVal - minVal;
    if (range == 0) {
        range = 1.0;
    }
    for (int i = 0; i < features.size(); i++) {
        features[i][colIdx] = (features[i][colIdx] - minVal) / (range);
    }
}

void Data::getMinMaxColumn(
    vector<vector<double> > &features, 
    double &minVal, 
    double &maxVal, 
    int colIdx
) {
    minVal = INFINITY;
    maxVal = -INFINITY;
    for (int i = 0; i < features.size(); i++) {
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
    for (int j = 0; j < trainFeatures[0].size(); j++) {
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
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].size(); j++) {
            features[i][j] /= MAX_GREYSCALE_VALUE;
        }
    }
}

void Data::minmaxGreyScale() {
    normalizeGreyScale(trainFeatures);
    normalizeGreyScale(testFeatures);
}

vector<int> Data::generateShuffledIndices() const {
    vector<int> indices(trainFeatures.size(), -1);

    for (int i = 0; i < trainFeatures.size(); i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

