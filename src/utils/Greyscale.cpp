#include "utils/Greyscale.h"
#include "core/Matrix.h"
#include <iostream>
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

void Greyscale::fit(const vector<double> &data) {
    cout << "Error: GreyscaleScaler only supports Matrix input." << endl;
    exit(1);
}

void Greyscale::transform(vector<double> &data) {
    cout << "Error: GreyscaleScaler only supports Matrix input." << endl;
    exit(1);
}

void Greyscale::reverseTransform(vector<double> &data) const {
    cout << "Error: GreyscaleScaler only supports Matrix input." << endl;
    exit(1);
}