#include "core/Data.h"
#include "utils/VectorUtils.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include "utils/FeatureEncoder.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include "utils/TrainingUtils.h"

random_device Data::rd;
mt19937 Data::generator(Data::rd());

const string Data::NO_TARGET_COL = "";
const size_t Data::NO_TARGET_IDX = numeric_limits<size_t>::max();
const size_t Data::MAX_DISPLAY_COLS = 15;

Data::Data() : task(nullptr), isDataLoaded(false) {}

void Data::readTrain(string filename, size_t targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading training data from: " << filename << endl;
    readCsv(filename, true, targetIdx, NO_TARGET_COL, hasHeader);
}

void Data::checkDataLoaded() const {
    if (!isDataLoaded) {
        cerr << "Fatal Error: Attempted to access data before loading.\n"
             << "Please ensure readTrain() is called first." << endl;
        exit(1);
    }
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

void Data::readTest(string filename, size_t targetIdx, bool hasHeader) {
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

void Data::fitScalars() {
    task->fitScalars(trainFeatures, trainTargets, testFeatures, testTargets);
}

Data::~Data() {
    delete task;
}

void Data::resetToRaw() {
    if (!isDataLoaded) {
        cerr << "Warning: Attempted to reset data before loading. No action taken." << endl;
        return;
    }

    trainFeatures = rawTrainFeatures;
    trainTargets = rawTrainTargets;
    testFeatures = rawTestFeatures;
    testTargets = rawTestTargets;
}

void Data::setScalars(Scalar *featureScalar, Scalar *targetScalar) {
    if (!task) {
        cerr << "Fatal Error: Task must be set before setting scalars.\n"
             << "Call setTask() before setScalars()." << endl;
        exit(1);
    }

    task->setFeatureScalar(featureScalar);
    if (targetScalar) {
        task->setTargetScalar(targetScalar);
    }
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

size_t Data::getColIdx(const string &colname) const {
    string cleanCol = CsvUtils::toLowerCase(CsvUtils::trim(colname));
    size_t numCols = header.size();

    for (size_t i = 0; i < numCols; i++) {
        if (header[i] == cleanCol) {
            return i;
        }
    }

    cerr << "Fatal Error: Column name \"" << colname << "\" not found in the dataset.\n"
        << "Please verify that the specified column exists in the CSV header." << endl;

    exit(1);

    return numeric_limits<size_t>::max();
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

vector<string> Data::validateAndLoadCsv(const string &filename, bool hasHeader) {
    CsvUtils::checkFile(filename);

    if (hasHeader) {
        header = CsvUtils::readHeader(filename);
    }

    return CsvUtils::collectLines(filename, hasHeader);
}

void Data::parseRawData(
    vector<vector<string> > &featuresRaw,
    vector<string> &targetsRaw,
    const vector<string> &lines,
    size_t targetIdx
) {
    size_t numSamples = lines.size();
    size_t numCols = CsvUtils::countFirstCol(lines[0]);

    featuresRaw.resize(numSamples, vector<string>(numCols - 1));
    targetsRaw.resize(numSamples);

    CsvUtils::parseLines(lines, featuresRaw, targetsRaw, targetIdx);
}

void Data::readCsv(
    string filename, 
    bool isTrainData, 
    size_t targetIdx, 
    const string& colname,
    bool hasHeader
) {
    vector<string> lines = validateAndLoadCsv(filename, hasHeader);

    if (colname != NO_TARGET_COL) {
        targetIdx = getColIdx(colname); 
    }

    vector<vector<string> > featuresRaw;
    vector<string> targetsRaw;
    parseRawData(featuresRaw, targetsRaw, lines, targetIdx);

    Matrix features = FeatureEncoder::getFeatures(featuresRaw);
    vector<double> target = task->getTarget(targetsRaw);

    setData(features, target, isTrainData);
    isDataLoaded = true;

    ConsoleUtils::printSepLine();
}

vector<size_t> Data::generateShuffledIndices() const {
    checkDataLoaded();
    size_t size = trainFeatures.getNumRows();
    vector<size_t> indices(size, -1);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

