#include "core/Data.h"
#include "utils/VectorUtils.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include "utils/FeatureEncoder.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "utils/TransformUtils.h"

random_device Data::rd;
mt19937 Data::generator(Data::rd());
const string Data::NO_TARGET_COL = "";
const int Data::NO_TARGET_IDX = -1;
const size_t Data::MAX_DISPLAY_COLS = 15;

Data::Data(): isDataLoaded(false) {}

void Data::readTrain(string filename, int targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading training data from: " << filename << endl;
    readCsv(filename, true, targetIdx, NO_TARGET_COL, hasHeader);
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

void Data::createLabelMap(
    const vector<string> &targetRaw
) {
    size_t numSamples = targetRaw.size();
    int nextIdx = 0;

    for (size_t i = 0; i < numSamples; i++) {
        const string &val = targetRaw[i];
        if (labelMap.find(val) == labelMap.end()) {
            labelMap[val] = nextIdx++;
        }
    }
}

vector<int> Data::getTarget(
    const vector<string> &targetRaw
) {
    ConsoleUtils::loadMessage("Extracting Targets.");
    createLabelMap(targetRaw);
    size_t numSamples = targetRaw.size();
    vector<int> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        target[i] = labelMap[targetRaw[i]];
    }
    ConsoleUtils::completeMessage();

    return target;
}

void Data::setData(const Matrix &features, vector<int> &target, bool isTrainData) {
    if (isTrainData) {
        trainFeatures = features;
        trainTarget = target;
    } else {
        testFeatures = features;
        testTarget = target;
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

void Data::head(
    size_t numRows,
    const Matrix &mat
) const {
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

void Data::headTrain(size_t numRows) const {
    head(numRows, trainFeatures);
}

void Data::headTest(size_t numRows) const {
    head(numRows, testFeatures);
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
    size_t numCols = CsvUtils::countFirstCol(lines[0]);

    vector<vector<string> > featuresRaw(numSamples, vector<string>(numCols - 1));
    vector<string> targetRaw(numSamples);

    CsvUtils::parseLines(lines, featuresRaw, targetRaw, targetIdx);
    Matrix features = FeatureEncoder::getFeatures(featuresRaw);
    vector<int> target = getTarget(targetRaw);

    setData(features, target, isTrainData);
    isDataLoaded = true;

    ConsoleUtils::printSepLine();
}


void Data::minmax() {
    if (isDataLoaded) {
        TransformUtils::minmaxData(trainFeatures, testFeatures);
    }
}

void Data::minmaxGreyScale() {
    TransformUtils::normalizeGreyScale(trainFeatures);
    TransformUtils::normalizeGreyScale(testFeatures);
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