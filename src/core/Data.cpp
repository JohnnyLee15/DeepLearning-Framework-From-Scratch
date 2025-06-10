#include "core/Data.h"
#include "utils/VectorUtils.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include "utils/FeatureEncoder.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "utils/TrainingUtils.h"

random_device Data::rd;
mt19937 Data::generator(Data::rd());

const string Data::NO_TARGET_COL = "";
const int Data::NO_TARGET_IDX = -1;
const size_t Data::MAX_DISPLAY_COLS = 15;

Data::Data() : task(nullptr), isDataLoaded(false) {}

void Data::readTrain(string filename, int targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading training data from: " << filename << endl;
    readCsv(filename, true, targetIdx, NO_TARGET_COL, hasHeader);
}

void Data::setTask(Task *taskType) {
    if (task) {
        delete task;
    }
    task = taskType;
}

void Data::readTrain(string filename, const string &colname) {
    cout << endl << "📥 Loading training data from: " << filename << endl;
    readCsv(filename, true, NO_TARGET_IDX, colname, true);
}

void Data::readTest(string filename, int targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading testing data from: " << filename << endl;
    readCsv(filename, false, targetIdx, NO_TARGET_COL, hasHeader);
}

void Data::readTest(string filename, const string &colname) {
    cout << endl << "📥 Loading testing data from: " << filename << endl;
    readCsv(filename, false, NO_TARGET_IDX, colname, true);
}

const Matrix& Data::getTrainFeatures() const {
    checkDataLoaded();
    return trainFeatures;
}

const Matrix& Data::getTestFeatures() const {
    checkDataLoaded();
    return testFeatures;
}

const vector<double>& Data::getTrainTargets() const {
    checkDataLoaded();
    return trainTargets;
}

const vector<double>& Data::getTestTargets() const {
    checkDataLoaded();
    return testTargets;
}

size_t Data::getNumTrainSamples() const {
    checkDataLoaded();
    return trainFeatures.getNumRows();
}

const Task* Data::getTask() const {
    checkDataLoaded();
    return task;
}

void Data::headTrain(size_t numRows) const {
    head(numRows, trainFeatures);
}

void Data::headTest(size_t numRows) const {
    head(numRows, testFeatures);
}

void Data::resetToRaw() {
    trainFeatures = rawTrainFeatures;
    trainTargets = rawTrainTargets;
    testFeatures = rawTestFeatures;
    testTargets = rawTestTargets;
}

void Data::setScalars(Scalar *featureScalar, Scalar *targetScalar) {
    task->setFeatureScalar(featureScalar);
    if (targetScalar) {
        task->setTargetScalar(targetScalar);
    }
}

void Data::fitScalars() {
    task->fitScalars(trainFeatures, trainTargets, testFeatures, testTargets);
}

Data::~Data() {
    delete task;
}

void Data::setData(const Matrix &features, vector<double> &target, bool isTrainData) {
    if (isTrainData) {
        rawTrainFeatures = features;
        rawTrainTargets = target;
        trainFeatures = features;
        trainTargets = target;
    } else {
        rawTestFeatures = features;
        rawTestTargets = target;
        testFeatures = features;
        testTargets = target;
    }
}

int Data::getColIdx(const string &colname) const {
    string cleanCol = CsvUtils::toLowerCase(CsvUtils::trim(colname));
    int idx = -1;
    size_t numCols = header.size();

    for (size_t i = 0; i < numCols && idx == -1; i++) {
        if (header[i] == cleanCol) {
            return (int) i;
        }
    }

    cout << "Error: Column name " << colname << " not found." << endl;
    exit(1);

    return idx;
}

void Data::checkDataLoaded() const {
    if (!isDataLoaded) {
        cout << "Error: Data must be loaded before using this method." << endl;
        exit(1);
    }
}

void Data::head(
    size_t numRows,
    const Matrix &mat
) const {
    checkDataLoaded();

    const vector<double> &matFlat = mat.getFlat();
    size_t matRows = mat.getNumRows();
    size_t matCols = mat.getNumCols();
    size_t displayCols = min(matCols, MAX_DISPLAY_COLS);

    cout << "Data shape: " << matRows << " x " << matCols << endl;
    cout << "Displaying first " << numRows << " rows and " << displayCols << " columns." << endl << endl;

    for (size_t i = 0; i < numRows; i++) {
        cout << "[" << (i+1) << "] ";
        for (size_t j = 0; j < displayCols; j++) {
            cout << matFlat[i * matCols + j];

            if (j < displayCols - 1) {
                cout << ", ";
            }
        }

        if (matCols > displayCols) {
            cout << ", ...";
        }
        cout << endl;
    }
}

void Data::readCsv(
    string filename, 
    bool isTrainData, 
    int targetIdx, 
    const string& colname,
    bool hasHeader
) {
    CsvUtils::checkFile(filename);

    if (hasHeader) header = CsvUtils::readHeader(filename);
    if (colname != NO_TARGET_COL) targetIdx = getColIdx(colname);

    vector<string> lines = CsvUtils::collectLines(filename, hasHeader);
    
    size_t numSamples = lines.size();
    assert(numSamples > 0);
    size_t numCols = CsvUtils::countFirstCol(lines[0]);

    vector<vector<string> > featuresRaw(numSamples, vector<string>(numCols - 1));
    vector<string> targetRaw(numSamples);

    CsvUtils::parseLines(lines, featuresRaw, targetRaw, targetIdx);
    Matrix features = FeatureEncoder::getFeatures(featuresRaw);
    vector<double> target = task->getTarget(targetRaw);

    setData(features, target, isTrainData);
    isDataLoaded = true;

    ConsoleUtils::printSepLine();
}

vector<int> Data::generateShuffledIndices() const {
    checkDataLoaded();
    size_t size = trainFeatures.getNumRows();
    vector<int> indices(size, -1);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

