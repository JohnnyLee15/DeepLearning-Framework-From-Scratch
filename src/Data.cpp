#include "Data.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

Data::Data(): isDataLoaded(false) {}

void Data::readTrain(string filename, int targetIdx) {
    readData(filename, true, targetIdx);
}

void Data::readTest(string filename, int targetIdx) {
    readData(filename, false, targetIdx);
}

vector<vector<double> > Data::getTrainFeatures() const {
    return trainFeatures;
}

vector<vector<double> > Data::getTestFeatures() const {
    return testFeatures;
}

vector<double> Data::getTrainTarget() const {
    return trainTarget;
}

vector<double> Data::getTestTarget() const {
    return testTarget;
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
            sample.push_back(value/255.0);
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

void Data::getMinAndMax(double &minVal, double &maxVal) const {
    for (int i = 0; i < trainFeatures.size(); i++) {
        for (int j = 0; j < trainFeatures[i].size(); j++) {
            if (trainFeatures[i][j] < minVal) {
                minVal = trainFeatures[i][j];
            }
            if (trainFeatures[i][j] > maxVal) {
                maxVal = trainFeatures[i][j];
            }
        }
    }
}

void Data::minmaxNormalize(double minVal, double maxVal) {
    for (int i = 0; i < trainFeatures.size(); i++) {
        for (int j = 0; j < trainFeatures[i].size(); j++) {
            trainFeatures[i][j] = (trainFeatures[i][j] - minVal) / (maxVal - minVal);
        }
    }
}

void Data::minmax() {
    if (isDataLoaded) {
        double minVal, maxVal;
        getMinAndMax(minVal, maxVal);
        minmaxNormalize(minVal, maxVal);
    }
}

