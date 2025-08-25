#include "core/data/TabularData.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include "utils/FeatureEncoder.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include "core/tensor/Matrix.h"
#include "utils/TrainingUtils.h"
#include <cstdint>
#include "utils/TargetEncoder.h"

const string TabularData::NO_TARGET_COL = "";
const size_t TabularData::NO_TARGET_IDX = numeric_limits<size_t>::max();
const size_t TabularData::MAX_DISPLAY_COLS = 25;
const string TabularData::REGRESSION_TASK = "regression";
const string TabularData::CLASSIFICATION_TASK = "classification";

TabularData::TabularData(const string &taskType) : 
    isTrainLoaded(false), isTestLoaded(false), isLoadedFromModel(false) {
    string taskFormatted = CsvUtils::toLowerCase(CsvUtils::trim(taskType));

    if (taskFormatted != REGRESSION_TASK && taskFormatted != CLASSIFICATION_TASK) {
        ConsoleUtils::fatalError(
            "Invalid task type: \"" + taskType + "\".\n"
            "Must be \"" + REGRESSION_TASK + "\" or \"" + CLASSIFICATION_TASK + "\"."
        );
    }

    task = taskFormatted;
}

TabularData::TabularData() {}

void TabularData::checkTrainLoaded() const {
    if (!isTrainLoaded) {
        ConsoleUtils::fatalError(
            "Attempted to access data before loading.\n" 
            "Please ensure readTrain() is called first."
        );
    }
}

void TabularData::checkTestLoaded() const {
    if (!isTestLoaded) {
        ConsoleUtils::fatalError(
            "Attempted to access data before loading.\n" 
            "Please ensure readTest() is called first."
        );
    }
}

void TabularData::readTrain(const string &filename, size_t targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading training data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, true, targetIdx, NO_TARGET_COL, hasHeader);
    isTrainLoaded = true;
}

void TabularData::readTrain(const string &filename, const string &colname) {
    cout << endl << "📥 Loading training data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, true, NO_TARGET_IDX, colname, true);
    isTrainLoaded = true;
}

void TabularData::readTest(const string &filename, size_t targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading testing data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, false, targetIdx, NO_TARGET_COL, hasHeader);
    isTestLoaded = true;
}

void TabularData::readTest(const string &filename, const string &colname) {
    cout << endl << "📥 Loading testing data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, false, NO_TARGET_IDX, colname, true);
    isTestLoaded = true;
}

const Tensor& TabularData::getTrainFeatures() const {
    checkTrainLoaded();
    return trainFeatures;
}

const Tensor& TabularData::getTestFeatures() const {
    checkTestLoaded();
    return testFeatures;
}

const vector<float>& TabularData::getTrainTargets() const {
    checkTrainLoaded();
    return trainTargets;
}

const vector<float>& TabularData::getTestTargets() const {
    checkTestLoaded();
    return testTargets;
}

size_t TabularData::getNumTrainSamples() const {
    checkTrainLoaded();
    return trainFeatures.M().getNumRows();
}

Data::Encodings TabularData::getEncoding() const {
    return Data::Encodings::Tabular;
}

void TabularData::headTrain(size_t numRows) const {
    checkTrainLoaded();
    head(numRows, trainFeatures);
}

void TabularData::headTest(size_t numRows) const {
    checkTestLoaded();
    head(numRows, testFeatures);
}

void TabularData::clearTrain() {
    trainFeatures.clear();
    vector<float>().swap(trainTargets);
}

void TabularData::clearTest() {
    testFeatures.clear();
    vector<float>().swap(testTargets);
}

void TabularData::head(
    size_t numRows,
    const Tensor &mat
) const {
    Matrix newMat = mat.M();
    const vector<float> &matFlat = mat.getFlat();
    size_t matRows = newMat.getNumRows();
    size_t matCols = newMat.getNumCols();
    size_t displayCols = min(matCols, MAX_DISPLAY_COLS);

    cout << "Data shape: " << matRows << " x " << matCols << endl;
    cout << "Displaying first " << numRows << " rows and " << displayCols << " columns." << endl << endl;

    numRows = numRows < matRows ? numRows : matRows;

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

void TabularData::setData(const Tensor &features, vector<float> &target, bool isTrainData) {
    if (isTrainData) {
        trainFeatures = features;
        trainTargets = target;
    } else {
        testFeatures = features;
        testTargets = target;
    }
}

size_t TabularData::getColIdx(const string &colname) const {
    string cleanCol = CsvUtils::toLowerCase(CsvUtils::trim(colname));
    size_t numCols = header.size();

    for (size_t i = 0; i < numCols; i++) {
        if (header[i] == cleanCol) {
            return i;
        }
    }
    
    ConsoleUtils::fatalError(
        "Column name \"" + colname + "\" not found in the dataset.\n"
        "Please verify that the specified column exists in the CSV header."
    );
    
    return numeric_limits<size_t>::max();
}

vector<string> TabularData::validateAndLoadCsv(const string &filename, bool hasHeader) {
    CsvUtils::checkFile(filename);

    if (hasHeader && header.empty()) {
        header = CsvUtils::readHeader(filename);
    }

    return CsvUtils::collectLines(filename, hasHeader);
}

void TabularData::parseRawData(
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

void TabularData::readCsv(
    const string &filename, 
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

    Tensor features = readFeatures(featuresRaw);
    vector<float> target = readTargets(targetsRaw);

    setData(features, target, isTrainData);
    ConsoleUtils::printSepLine();
}

vector<float> TabularData::readTargets(const vector<string> &targetsRaw) {
    ConsoleUtils::loadMessage("Extracting Targets.");
    vector<float> targets;
    if (task == REGRESSION_TASK) {
        targets = TargetEncoder::getRegressionTarget(targetsRaw);
    } else {
        if (labelMap.empty()) {
            labelMap = TargetEncoder::createLabelMap(targetsRaw);
        }

        targets = TargetEncoder::getClassificationTarget(targetsRaw, labelMap);
    }
    
    ConsoleUtils::completeMessage();
    return targets;
}

Tensor TabularData::readFeatures(const vector<vector<string> > &featuresRaw) {
    ConsoleUtils::loadMessage("Extracting Features.");
    if (isCategorical.empty()) {
        isCategorical = FeatureEncoder::getCategoricalCols(featuresRaw);
    }

    if (featureEncodings.empty()) {
        featureEncodings = FeatureEncoder::encodeFeatures(featuresRaw, isCategorical);
    }

    Tensor features = FeatureEncoder::getFeatures(isCategorical, featureEncodings, featuresRaw);
    ConsoleUtils::completeMessage();
    return features;
}

void TabularData::writeBin(ofstream &modelBin) const {
    Data::writeBin(modelBin);

    uint32_t taskLen = task.size();
    modelBin.write((char*) &taskLen, sizeof(uint32_t));
    modelBin.write(task.c_str(), taskLen);

    uint32_t headerSize =  header.size();
    modelBin.write((char*) &headerSize, sizeof(uint32_t));
    for (uint32_t i = 0; i < headerSize; i++) {
        uint32_t colNameLen = header[i].size();
        modelBin.write((char*) &colNameLen, sizeof(uint32_t));
        modelBin.write(header[i].c_str(), colNameLen);
    }

    uint32_t isCategoricalSize = isCategorical.size();
    modelBin.write((char*) &isCategoricalSize, sizeof(uint32_t));
    for (uint32_t i = 0; i < isCategoricalSize; i++) {
        uint8_t val = 0;
        if (isCategorical[i]) {
            val = 1;
        }
        modelBin.write((char*) &val, sizeof(uint8_t));
    }

    uint32_t numCatCols = featureEncodings.size();
    modelBin.write((char*) &numCatCols, sizeof(uint32_t));
    for (uint32_t i = 0; i < numCatCols; i++) {
        uint32_t mapSize = featureEncodings[i].size();
        modelBin.write((char*) &mapSize, sizeof(uint32_t));
        for (const pair<const string, float> &pair : featureEncodings[i]) {
            uint32_t keyLen = pair.first.size();
            modelBin.write((char*) &keyLen, sizeof(uint32_t));
            modelBin.write(pair.first.c_str(), keyLen);
            modelBin.write((char*) &pair.second, sizeof(float));
        }
    }

    if (task == CLASSIFICATION_TASK) {
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
}

void TabularData::loadFromBin(ifstream &modelBin) {
    isLoadedFromModel = true;
    
    uint32_t taskLen;
    modelBin.read((char*) &taskLen, sizeof(uint32_t));
    task = string(taskLen, '\0');
    modelBin.read(task.data(), taskLen);

    uint32_t headerSize;
    modelBin.read((char*) &headerSize, sizeof(uint32_t));
    for (uint32_t i = 0; i < headerSize; i++) {
        uint32_t colNameLen;
        modelBin.read((char*) &colNameLen, sizeof(uint32_t));

        string colName(colNameLen, '\0');
        modelBin.read(colName.data(), colNameLen);

        header.push_back(colName);
    }

    uint32_t isCategoricalSize;
    modelBin.read((char*) &isCategoricalSize, sizeof(uint32_t));
    for (uint32_t i = 0; i < isCategoricalSize; i++) {
        uint8_t val;
        modelBin.read((char*) &val, sizeof(uint8_t));
        isCategorical.push_back(val != 0);
    }

    uint32_t numCatCols;
    modelBin.read((char*) &numCatCols, sizeof(uint32_t));
    for (uint32_t i = 0; i < numCatCols; i++) {
        featureEncodings.push_back(unordered_map<string, float>());
        uint32_t mapSize;
        modelBin.read((char*) &mapSize, sizeof(uint32_t));

        for (uint32_t j = 0; j < mapSize; j++) {
            uint32_t keyLen;
            modelBin.read((char*) &keyLen, sizeof(uint32_t));

            string key(keyLen, '\0');
            modelBin.read(key.data(), keyLen);

            float value;
            modelBin.read((char*) &value, sizeof(float));

            featureEncodings[i][key] = value;
        }
    }

    if (task == CLASSIFICATION_TASK) {
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
}

Data* TabularData::clone() const {
    return new TabularData(*this);
}