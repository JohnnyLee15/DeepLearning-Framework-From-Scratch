#include "core/TabularData.h"
#include "utils/VectorUtils.h"
#include "utils/CsvUtils.h"
#include "utils/ConsoleUtils.h"
#include "utils/FeatureEncoder.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include "core/Matrix.h"
#include "utils/TrainingUtils.h"
#include <cstdint>
#include "core/RegressionTask.h"
#include "core/ClassificationTask.h"

random_device TabularData::rd;
mt19937 TabularData::generator(TabularData::rd());

const string TabularData::NO_TARGET_COL = "";
const size_t TabularData::NO_TARGET_IDX = numeric_limits<size_t>::max();
const size_t TabularData::MAX_DISPLAY_COLS = 250;

TabularData::TabularData() : task(nullptr), isDataLoaded(false), isLoadedFromModel(false) {}

TabularData::~TabularData() {
    delete task;
}

void TabularData::checkDataLoaded() const {
    if (!isDataLoaded) {
        ConsoleUtils::fatalError(
            "Attempted to access data before loading.\n" 
            "Please ensure readTrain() is called first."
        );
    }
}

void TabularData::checkTask(const string &context) const {
    if (!task) {
        ConsoleUtils::fatalError(
            "Task must be set before " + context + ".\n"
            "Call setTask() before this operation."
        );
    }
}

void TabularData::setTask(Task *taskType) {
    if (task) {
        delete task;
    }
    task = taskType;
}

void TabularData::readTrain(string filename, size_t targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading training data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, true, targetIdx, NO_TARGET_COL, hasHeader);
}

void TabularData::readTrain(string filename, const string &colname) {
    cout << endl << "📥 Loading training data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, true, NO_TARGET_IDX, colname, true);
}

void TabularData::readTest(string filename, size_t targetIdx, bool hasHeader) {
    cout << endl << "📥 Loading testing data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, false, targetIdx, NO_TARGET_COL, hasHeader);
}

void TabularData::readTest(string filename, const string &colname) {
    cout << endl << "📥 Loading testing data from: \"" << CsvUtils::trimFilePath(filename) << "\"." << endl;
    readCsv(filename, false, NO_TARGET_IDX, colname, true);
}

const Tensor& TabularData::getTrainFeatures() const {
    checkDataLoaded();
    return trainFeatures;
}

const Tensor& TabularData::getTestFeatures() const {
    checkDataLoaded();
    return testFeatures;
}

const vector<double>& TabularData::getTrainTargets() const {
    checkDataLoaded();
    return trainTargets;
}

const vector<double>& TabularData::getTestTargets() const {
    checkDataLoaded();
    return testTargets;
}

size_t TabularData::getNumTrainSamples() const {
    checkDataLoaded();
    return trainFeatures.M().getNumRows();
}

const Task* TabularData::getTask() const {
    checkDataLoaded();
    return task;
}

void TabularData::headTrain(size_t numRows) const {
    head(numRows, trainFeatures);
}

void TabularData::headTest(size_t numRows) const {
    head(numRows, testFeatures);
}

void TabularData::setScalars(Scalar *featureScalar, Scalar *targetScalar) {
    checkTask("setting scalars");
    if (isLoadedFromModel) {
        ConsoleUtils::fatalError(
            "setScalars() is not allowed after loading a saved model. "
            "Use transformTrain() and transformTest() instead."
        );
    }

    task->setFeatureScalar(featureScalar);
    if (targetScalar) {
        task->setTargetScalar(targetScalar);
    }
}

void TabularData::fitScalars() {
    checkDataLoaded();
    if (isLoadedFromModel) {
        ConsoleUtils::fatalError(
            "fitScalars() is not allowed after loading a saved model. "
            "Use transformTrain() and transformTest() instead."
        );
    }
    task->fitScalars(trainFeatures, trainTargets);
}

void TabularData::transformTrain() {
    checkTask("setting scalars");
    task->transformScalars(trainFeatures, trainTargets);
}

void TabularData::transformTest() {
    checkTask("setting scalars");
    task->transformScalars(testFeatures, testTargets);
}

void TabularData::reverseTransformTrain() {
    checkTask("setting scalars");
    task->reverseTransformScalars(trainFeatures, trainTargets);
}

void TabularData::reverseTransformTest() {
    checkTask("setting scalars");
    task->reverseTransformScalars(testFeatures, testTargets);
}

void TabularData::resetToRaw() {
    if (!isDataLoaded) {
        cerr << "Warning: Attempted to reset data before loading. No action taken." << endl;
        return;
    }

    trainFeatures = rawTrainFeatures;
    trainTargets = rawTrainTargets;
    testFeatures = rawTestFeatures;
    testTargets = rawTestTargets;

    if (task) {
        task->resetToRaw();
    }
}

void TabularData::setData(const Tensor &features, vector<double> &target, bool isTrainData) {
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

void TabularData::head(
    size_t numRows,
    const Tensor &mat
) const {
    checkDataLoaded();
    Matrix newMat = mat.M();
    const vector<double> &matFlat = mat.getFlat();
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
    string filename, 
    bool isTrainData, 
    size_t targetIdx, 
    const string& colname,
    bool hasHeader
) {
    checkTask("reading CSV data");
    vector<string> lines = validateAndLoadCsv(filename, hasHeader);

    if (colname != NO_TARGET_COL) {
        targetIdx = getColIdx(colname); 
    }

    vector<vector<string> > featuresRaw;
    vector<string> targetsRaw;
    parseRawData(featuresRaw, targetsRaw, lines, targetIdx);

    Tensor features = readFeatures(featuresRaw);
    vector<double> target = task->getTarget(targetsRaw);

    setData(features, target, isTrainData);
    isDataLoaded = true;

    ConsoleUtils::printSepLine();
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

vector<size_t> TabularData::generateShuffledIndices() const {
    checkDataLoaded();
    size_t size = trainFeatures.M().getNumRows();
    vector<size_t> indices(size, 0);
    
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

void TabularData::writeBin(ofstream &modelBin) const {
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
        for (const pair<const string, double> &pair : featureEncodings[i]) {
            uint32_t keyLen = pair.first.size();
            modelBin.write((char*) &keyLen, sizeof(uint32_t));
            modelBin.write(pair.first.c_str(), keyLen);
            modelBin.write((char*) &pair.second, sizeof(double));
        }
    }

    task->writeBin(modelBin);
}

void TabularData::loadFromBin(ifstream &modelBin) {
    isLoadedFromModel = true;

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
        featureEncodings.push_back(unordered_map<string, double>());
        uint32_t mapSize;
        modelBin.read((char*) &mapSize, sizeof(uint32_t));

        for (uint32_t j = 0; j < mapSize; j++) {
            uint32_t keyLen;
            modelBin.read((char*) &keyLen, sizeof(uint32_t));

            string key(keyLen, '\0');
            modelBin.read(key.data(), keyLen);

            double value;
            modelBin.read((char*) &value, sizeof(double));

            featureEncodings[i][key] = value;
        }
    }

    uint32_t taskEncoding;
    modelBin.read((char*) &taskEncoding, sizeof(uint32_t));
    if (taskEncoding == Task::Encodings::Regression) {
        task = new RegressionTask();
    } else if (taskEncoding == Task::Encodings::Classification) {
        task = new ClassificationTask();
    } else {
        ConsoleUtils::fatalError(
            "Unsupported task encoding \"" + to_string(taskEncoding) + "\"."
        );
    }

    task->loadFromBin(modelBin);
}