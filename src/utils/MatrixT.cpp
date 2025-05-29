#include "utils/MatrixT.h"
#include <cassert>

MatrixT::MatrixT(const Matrix &matrix) :
    numRows(matrix.getNumCols()), numCols(matrix.getNumRows()), matrix(matrix) {}

size_t MatrixT::getNumRows() const {
    return numRows;
}

size_t MatrixT::getNumCols() const {
    return numCols;
}

double MatrixT::getValue(size_t row, size_t col) const {
    return matrix.getFlat()[col * matrix.getNumCols() + row];
}

Matrix MatrixT::operator *(const Matrix &mat2) const {
    assert(numCols == mat2.getNumRows());

    if (numCols*numRows > Matrix::L2_CACHE_DOUBLES) {
        return (*this).transpose() * mat2;
    }

    return multMatTMat(mat2);
}

Matrix MatrixT::operator *(const MatrixT &mat2) const {
    assert(numCols == mat2.numRows);

    size_t size1 = numCols * numRows;
    size_t size2 = mat2.numRows * mat2.numCols;

    if (size1 > Matrix::L2_CACHE_DOUBLES && size2 > Matrix::L2_CACHE_DOUBLES) {
        return (*this).transpose() * mat2.transpose();
    } else if (size1 > Matrix::L2_CACHE_DOUBLES) {
        return (*this).transpose() * mat2;
    } else if (size2 > Matrix::L2_CACHE_DOUBLES) { 
        return multMatTMat(mat2.transpose());
    } 
        
    return multMatTMatT(mat2);

}

Matrix MatrixT::multMatTMat(const Matrix &mat2) const {
    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();

    Matrix product(numRows, mat2Cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += getValue(i, k) * mat2.getValue(k, j);
            }
            product.setValue(i, j, value);
        }
    }

    return product;
}

Matrix MatrixT::multMatTMatT(const MatrixT &mat2) const {
    size_t mat2Rows = mat2.numRows;
    size_t mat2Cols = mat2.numCols;

    Matrix product(numRows, mat2Cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += getValue(i, k) * mat2.getValue(k, j);
            }
            product.setValue(i, j, value);
        }
    }

    return product;
}

Matrix MatrixT::transpose() const {
    const vector<double> &mat = matrix.getFlat();
    Matrix matT(numRows, numCols);

    size_t matRows = matrix.getNumRows();
    size_t matCols = matrix.getNumCols();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < matRows; i++) {
        for (size_t j = 0; j < matCols; j++) {
            matT.setValue(j, i, mat[i*matCols + j]);
        }
    }

    return matT;
}
