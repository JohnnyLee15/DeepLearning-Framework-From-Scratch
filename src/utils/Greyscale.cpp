#include "utils/Greyscale.h"
#include "core/Matrix.h"
#include <iostream>
#include "utils/ConsoleUtils.h"
#include <omp.h>

const double Greyscale::MAX_GREYSCALE_VALUE = 255.0;

void Greyscale::transform(Matrix &data) {
    Scalar::transform(data);

    size_t numRows = data.getNumRows();
    size_t numCols = data.getNumCols();

    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            dataFlat[i*numCols + j] /= MAX_GREYSCALE_VALUE;
        }
    }
}

void Greyscale::reverseTransform(Matrix &data) const {
    Scalar::reverseTransform(data);

    size_t numRows = data.getNumRows();
    size_t numCols = data.getNumCols();

    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            dataFlat[i*numCols + j] *= MAX_GREYSCALE_VALUE;
        }
    }
}

void Greyscale::throwDataFormatError() const {
    ConsoleUtils::fatalError(
        "Greyscale only supports Matrix input.\n"
        "Vector input is not supported for this scalar."
    );
}

void Greyscale::fit(const vector<double> &data) {
    throwDataFormatError();
}

void Greyscale::transform(vector<double> &data) {
    throwDataFormatError();
}

void Greyscale::reverseTransform(vector<double> &data) const {
    throwDataFormatError();
}