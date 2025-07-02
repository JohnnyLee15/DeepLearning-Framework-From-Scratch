#include "core/Tensor.h"
#include "utils/ConsoleUtils.h"
#include "core/Matrix.h"

Tensor::Tensor(const vector<size_t> &shape) : shape(shape) {
    size_t size = 1;
    size_t dims = shape.size();
    for (size_t i = 0; i < dims; i++) {
        size *= shape[i];
    }

    data = vector<double>(size);
}

Tensor::Tensor(const vector<vector<double> > &mat) {
    size_t numRows = mat.size();
    size_t numCols = 0;
    if (numRows > 0) {
        numCols = mat[0].size();
    }

    shape = {numRows, numCols};
    data = vector<double>(numRows * numCols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            data[i * numCols + j] = mat[i][j];
        }
    }
}

Tensor::Tensor() {}

const vector<double>& Tensor::getFlat() const {
    return data;
}

vector<double>& Tensor::getFlat() {
    return data;
}

const vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::getSize() const {
    if (shape.empty()) {
        return 0;
    }
    size_t size = 1;
    size_t dims = shape.size();
    for (size_t i = 0; i < dims; i++) {
        size *= shape[i];
    }

    return size;
}

size_t Tensor::getRank() const {
    return shape.size();
}

Matrix Tensor::M() const {
    if (getRank() != 2) {
        ConsoleUtils::fatalError(
            string("Matrix view requires a rank-2 tensor.\n") +
            "Received tensor with rank: " + to_string(getRank()) + "."
        );
    }

    return Matrix(const_cast<Tensor&>(*this));
}
