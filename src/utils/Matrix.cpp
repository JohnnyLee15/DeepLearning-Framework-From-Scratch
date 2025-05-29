#include "utils/Matrix.h"
#include "utils/MatrixT.h"
#include <cassert>

Matrix::Matrix(size_t numRows, size_t numCols) : 
    matrix(numRows * numCols), numRows(numRows), numCols(numCols) {}

Matrix::Matrix() :
    numRows(0), numCols(0) {}

Matrix::Matrix(const vector<vector<double> > &mat) : numRows(mat.size()) {
    numCols = 0;
    if (numRows > 0) {
        numCols = mat[0].size();
    }

    matrix = vector<double>(numRows * numCols);

    #pragma omp parallel for
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = mat[i][j];
        }
    }
}

size_t Matrix::getNumCols() const {
    return numCols;
}

size_t Matrix::getNumRows() const {
    return numRows;
}

const vector<double>& Matrix::getFlat() const {
    return matrix;
}

vector<double>& Matrix::getFlat() {
    return matrix;
}

Matrix Matrix::operator *(const Matrix &mat2) const {
    assert(numCols == mat2.numRows);

    size_t mat2Rows = mat2.numRows;
    size_t mat2Cols = mat2.numCols;

    Matrix product(numRows, mat2Cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matrix[i * numCols + k]* mat2.matrix[k * mat2Cols + j];
            }
            product.matrix[i * mat2Cols + j] = value;
        }
    }

    return product;
}

Matrix Matrix::operator *(const MatrixT &mat2) const {
    assert(numCols == mat2.getNumRows());

    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    Matrix product(numRows, mat2Cols);

    const vector<double> &mat2Flat = mat2.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matrix[i * numCols + k] * mat2Flat[j * mat2Rows+ k];
            }
            product.matrix[i * mat2Cols + j] = value;
        }
    }

    return product;
}

vector<double> Matrix::operator *(const vector<double> &vec) const {
    assert(numCols == vec.size());

    vector<double> product(numRows, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        double value = 0.0;
        for (size_t j = 0; j < numCols; j++) {
            value += matrix[i * numCols + j] * vec[j];
        }
        product[i] = value;
    }

    return product;
}

vector<double> Matrix::colSums() const {
    vector<double> colSumsVec(numCols, 0.0);

    #pragma omp parallel 
    {
        vector<double> threadColSums(numCols, 0.0);

        #pragma omp for
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                threadColSums[j] += matrix[i * numCols + j];
            }
        }
        
        #pragma omp critical 
        {
            for (size_t j = 0; j < numCols; j++) {
                colSumsVec[j] += threadColSums[j];
            }
        }
    }

    return colSumsVec;
}

Matrix& Matrix::operator *=(const Matrix &mat2) {
    assert(numRows ==  mat2.getNumRows());
    assert(numCols == mat2.getNumCols());
    size_t size = numRows * numCols;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        matrix[i] *= mat2.matrix[i];
    }

    return *this;
}

Matrix& Matrix::operator *=(double scaleFactor){
    size_t size = numCols * numRows;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        matrix[i] *= scaleFactor;
    }

    return *this;
}

Matrix& Matrix::operator +=(const Matrix &mat2) {
    assert(numRows ==  mat2.getNumRows());
    assert(numCols == mat2.getNumCols());
    size_t size = numRows * numCols;


    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        matrix[i] += mat2.matrix[i];
    }

    return *this;
}

void Matrix::addToRows(const vector<double> &vec) {
    assert(vec.size() == numCols);

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            matrix[i * numCols + j] += vec[j];
        }
    }
}


MatrixT Matrix::T() const {
    return MatrixT(*this);
}
