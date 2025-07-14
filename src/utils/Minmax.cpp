#include "utils/Minmax.h"
#include "core/Matrix.h"
#include "core/Tensor.h"
#include <omp.h>
#include "utils/ConsoleUtils.h"
#include <limits>

void Minmax::checkRank(const Tensor &data) const {
    if (data.getRank() != 2) {
        ConsoleUtils::fatalError(
            "Minmax scaling only supports rank-2 tensors (matrices).\n"
            "Received tensor with rank: " + to_string(data.getRank()) + "."
        );
    }
}

void Minmax::checkDims(size_t toFitDim) const {
    if (minVals.size() != toFitDim) {
        ConsoleUtils::fatalError(
            "Minmax scaling dimension mismatch:\n"
            "Expected numCols = " + to_string(minVals.size()) +
            ", but got " + to_string(toFitDim) + "."
        );
    }
}

void Minmax::fit(const Tensor &data) {
    checkRank(data);
    Scalar::fit(data);
    
    Matrix dataMat = data.M();
    size_t numCols = dataMat.getNumCols();
    size_t numRows = dataMat.getNumRows();
    const vector<float> &dataFlat = data.getFlat();

    minVals = vector<float>(numCols, numeric_limits<float>::infinity());
    maxVals = vector<float>(numCols, -numeric_limits<float>::infinity());

    #pragma omp parallel
    {
        vector<float> threadMinVals(numCols, numeric_limits<float>::infinity());
        vector<float> threadMaxVals(numCols, -numeric_limits<float>::infinity());

        #pragma omp for
        for (size_t j = 0; j < numCols; j++) {
            for (size_t i = 0; i < numRows; i++) {
                float val = dataFlat[i * numCols + j];
                if (val < threadMinVals[j]) threadMinVals[j] = val;
                if (val > threadMaxVals[j]) threadMaxVals[j] = val;
            }
        }

        #pragma omp critical
        {
            for (size_t j = 0; j < numCols; j++) {
                if (threadMinVals[j] < minVals[j]) minVals[j] = threadMinVals[j];
                if (threadMaxVals[j] > maxVals[j]) maxVals[j] = threadMaxVals[j];
            }
        }
    }
}

void Minmax::fit(const vector<float> &data) {
    Scalar::fit(data);

    size_t size = data.size();
    float minVal = numeric_limits<float>::infinity();
    float maxVal = -numeric_limits<float>::infinity();

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < size; i++) {
        float val = data[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    minVals = vector<float>(1, minVal);
    maxVals = vector<float>(1, maxVal);
}

Tensor Minmax::transform(const Tensor &data) const {
    checkFitted();
    checkRank(data);
    checkDims(data.getShape()[1]);

    Tensor transformed(data.getShape());
    vector<float> &transformedFlat = transformed.getFlat();

    Matrix dataMat = data.M();
    size_t numCols = dataMat.getNumCols();
    size_t numRows = dataMat.getNumRows();
    const vector<float> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        float scaleFactor = maxVals[j] - minVals[j];
        if (scaleFactor == 0.0) {
            scaleFactor = 1.0;
        }

        for (size_t i = 0; i < numRows; i++) {
            transformedFlat[i * numCols + j] = (dataFlat[i * numCols + j] - minVals[j])/scaleFactor;
        }
    }

    return transformed;
}

vector<float> Minmax::transform(const vector<float> &data) const {
    checkFitted();
    checkDims(1);

    size_t size = data.size();
    vector<float> transformed(size);

    float scaleFactor = maxVals[0] - minVals[0];
    if (scaleFactor == 0.0) {
        scaleFactor = 1.0;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        transformed[i] = (data[i] - minVals[0])/scaleFactor;
    }

    return transformed;
}

Tensor Minmax::reverseTransform(const Tensor &data) const {
    checkFitted();
    checkRank(data);
    checkDims(data.getShape()[1]);

    Tensor transformed(data.getShape());
    vector<float> &transformedFlat = transformed.getFlat();

    Matrix dataMat = data.M();
    size_t numCols = dataMat.getNumCols();
    size_t numRows = dataMat.getNumRows();
    const vector<float> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        float scaleFactor = maxVals[j] - minVals[j];
        if (scaleFactor == 0.0) {
            scaleFactor = 1.0;
        }

        for (size_t i = 0; i < numRows; i++) {
            transformedFlat[i * numCols + j] = dataFlat[i * numCols + j] * scaleFactor + minVals[j];
        }
    }

    return transformed;
}

vector<float> Minmax::reverseTransform(const vector<float> &data) const {
    checkFitted();
    checkDims(1);
    
    size_t size = data.size();
    vector<float> transformed(size);

    float scaleFactor = maxVals[0] - minVals[0];
    if (scaleFactor == 0.0) {
        scaleFactor = 1.0;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        transformed[i] = data[i] * scaleFactor + minVals[0];
    }

    return transformed;
}

void Minmax::reset() {
    minVals.clear();
    maxVals.clear();
}

void Minmax::writeBin(ofstream &modelBin) const {
    Scalar::writeBin(modelBin);

    uint32_t minSize = minVals.size();
    uint32_t maxSize = maxVals.size();

    modelBin.write((char*) &minSize, sizeof(uint32_t));
    modelBin.write((char*) minVals.data(), minSize * sizeof(float));

    modelBin.write((char*) &maxSize, sizeof(uint32_t));
    modelBin.write((char*) maxVals.data(), maxSize * sizeof(float));
}

void Minmax::loadFromBin(ifstream &modelBin) {
    Scalar::loadFromBin(modelBin);

    uint32_t minSize;
    modelBin.read((char*) &minSize, sizeof(uint32_t));
    minVals.resize(minSize);
    modelBin.read((char*) minVals.data(), minSize * sizeof(float));

    uint32_t maxSize;
    modelBin.read((char*) &maxSize, sizeof(uint32_t));
    maxVals.resize(maxSize);
    modelBin.read((char*) maxVals.data(), maxSize * sizeof(float));
}

uint32_t Minmax::getEncoding() const {
    return Scalar::Encodings::Minmax;
}