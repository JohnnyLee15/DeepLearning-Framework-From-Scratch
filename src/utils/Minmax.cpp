#include "utils/Minmax.h"
#include "core/Matrix.h"
#include "utils/VectorUtils.h"
#include <omp.h>

void Minmax::fit(const Matrix &data) {
    Scalar::fit(data);

    size_t numCols = data.getNumCols();
    size_t numRows = data.getNumRows();
    const vector<double> &dataFlat = data.getFlat();
    minVals = vector<double>(numCols, VectorUtils::INF);
    maxVals = vector<double>(numCols, -VectorUtils::INF);

    #pragma omp parallel
    {
        vector<double> threadMinVals(numCols, VectorUtils::INF);
        vector<double> threadMaxVals(numCols, -VectorUtils::INF);

        #pragma omp for
        for (size_t j = 0; j < numCols; j++) {
            for (size_t i = 0; i < numRows; i++) {
                double val = dataFlat[i * numCols + j];
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

void Minmax::fit(const vector<double> &data) {
    Scalar::fit(data);

    size_t size = data.size();
    double minVal = VectorUtils::INF;
    double maxVal = -VectorUtils::INF;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < size; i++) {
        double val = data[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    minVals = vector<double>(1, minVal);
    maxVals = vector<double>(1, maxVal);
}

void Minmax::transform(Matrix &data) {
    Scalar::transform(data);

    size_t numCols = data.getNumCols();
    size_t numRows = data.getNumRows();
    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        double scaleFactor = maxVals[j] - minVals[j];
        if (scaleFactor == 0.0) {
            scaleFactor = 1.0;
        }

        for (size_t i = 0; i < numRows; i++) {
            dataFlat[i * numCols + j] = (dataFlat[i * numCols + j] - minVals[j])/scaleFactor;
        }
    }
}

void Minmax::transform(vector<double> &data) {
    Scalar::transform(data);

    size_t size = data.size();

    double scaleFactor = maxVals[0] - minVals[0];
    if (scaleFactor == 0.0) {
        scaleFactor = 1.0;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] - minVals[0])/scaleFactor;
    }
}

void Minmax::reverseTransform(Matrix &data) const {
    Scalar::reverseTransform(data);

    size_t numCols = data.getNumCols();
    size_t numRows = data.getNumRows();
    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        double scaleFactor = maxVals[j] - minVals[j];
        if (scaleFactor == 0.0) {
            scaleFactor = 1.0;
        }

        for (size_t i = 0; i < numRows; i++) {
            dataFlat[i * numCols + j] = dataFlat[i * numCols + j] * scaleFactor + minVals[j];
        }
    }
}

void Minmax::reverseTransform(vector<double> &data) const {
    Scalar::reverseTransform(data);

    size_t size = data.size();

    double scaleFactor = maxVals[0] - minVals[0];
    if (scaleFactor == 0.0) {
        scaleFactor = 1.0;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] = data[i] * scaleFactor + minVals[0];
    }
}