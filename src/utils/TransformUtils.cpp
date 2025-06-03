#include "utils/TransformUtils.h"
#include "utils/VectorUtils.h"
#include "core/Matrix.h"

const double TransformUtils::MAX_GREYSCALE_VALUE = 255.0;

void TransformUtils::minmaxNormalizeColumn(
    Matrix &features, 
    double minVal, 
    double maxVal, 
    int colIdx
) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();
    vector<double> &featuresFlat = features.getFlat();
    double range = maxVal - minVal;
    
    if (range == 0.0) {
        range = 1.0;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        featuresFlat[i*numCols + colIdx] = (featuresFlat[i*numCols + colIdx] - minVal) / range;
    }
}

void TransformUtils::getMinMaxColumn(
    const Matrix &features, 
    double &minVal, 
    double &maxVal, 
    int colIdx
) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();
    const vector<double> &featuresFlat = features.getFlat();
    minVal = VectorUtils::INF;
    maxVal = -VectorUtils::INF;
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < numRows; i++) {
        double val = featuresFlat[i*numCols + colIdx];
        if (val < minVal) {
            minVal = val;
        }
        if (val > maxVal) {
            maxVal = val;
        }
    }
}

void TransformUtils::minmaxData(
    Matrix &trainFeatures,
    Matrix &testFeatures
) {
    size_t numCols = trainFeatures.getNumCols();

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++) {
        double minVal, maxVal;
        getMinMaxColumn(trainFeatures, minVal, maxVal, j);
        minmaxNormalizeColumn(trainFeatures, minVal, maxVal, j);
        minmaxNormalizeColumn(testFeatures, minVal, maxVal, j);
    }
}

void TransformUtils::normalizeGreyScale(Matrix &features) {
    size_t numRows = features.getNumRows();
    size_t numCols = features.getNumCols();

    vector<double> &featuresFlat = features.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            featuresFlat[i*numCols + j] /= MAX_GREYSCALE_VALUE;
        }
    }
}