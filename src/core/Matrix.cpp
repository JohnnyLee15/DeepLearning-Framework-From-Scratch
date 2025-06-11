#include "core/Matrix.h"
#include "core/MatrixT.h"
#include <iostream>

Matrix::Matrix(size_t numRows, size_t numCols) : 
    matrix(numRows * numCols, 0), numRows(numRows), numCols(numCols) {}

Matrix::Matrix() :
    numRows(0), numCols(0) {}

Matrix::Matrix(const vector<vector<double> > &mat) : numRows(mat.size()) {
    numCols = 0;
    if (numRows > 0) {
        numCols = mat[0].size();
    }

    matrix = vector<double>(numRows * numCols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
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

void Matrix::checkSizeMatch(size_t mat1Cols, size_t mat2Rows) {
    if (mat1Cols != mat2Rows) {
        cerr << "Fatal Error: Matrix multiplication size mismatch.\n"
            << "Left matrix columns: " << mat1Cols
            << ", Right matrix rows: " << mat2Rows << "." << endl;
        exit(1);
    }
}

void Matrix::checkSameShape(
    size_t mat1Rows,
    size_t mat1Cols,
    size_t mat2Rows,
    size_t mat2Cols,
    const string &operation
) {
    if (mat1Rows!= mat2Rows || mat1Cols != mat2Cols) {
        cerr << "Fatal Error: Hadamard product requires matrices of the same shape.\n"
             << "Left matrix shape: (" << mat1Rows << ", " << mat1Cols << "), "
             << "Right matrix shape: (" << mat2Rows  << ", " << mat2Cols << ")." << endl;
        exit(1);
    }
}

Matrix Matrix::operator *(const Matrix &mat2) const {
    checkSizeMatch(numCols, mat2.numRows);

    size_t mat2Rows = mat2.numRows;
    size_t mat2Cols = mat2.numCols;

    Matrix product(numRows, mat2Cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        size_t offsetThis = i * numCols;
        size_t offsetProd = i * mat2Cols;
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matrix[offsetThis + k]* mat2.matrix[k * mat2Cols + j];
            }
            product.matrix[offsetProd + j] = value;
        }
    }

    return product;
}

Matrix Matrix::operator *(const MatrixT &mat2) const {
    checkSizeMatch(numCols,mat2.getNumRows());

    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    Matrix product(numRows, mat2Cols);

    const vector<double> &mat2Flat = mat2.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        size_t offsetThis = i * numCols;
        size_t offsetProd = i * mat2Cols;
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matrix[offsetThis + k] * mat2Flat[j * mat2Rows+ k];
            }
            product.matrix[offsetProd + j] = value;
        }
    }

    return product;
}

vector<double> Matrix::operator *(const vector<double> &vec) const {
    checkSizeMatch(numCols, vec.size());

    vector<double> product(numRows, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        double value = 0.0;
        size_t offset = i * numCols;
        for (size_t j = 0; j < numCols; j++) {
            value += matrix[offset + j] * vec[j];
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
    checkSameShape(numRows, numCols, mat2.numRows, mat2.numCols, "Hadamard Product");
    
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
    checkSameShape(numRows, numCols, mat2.numRows, mat2.numCols, "Matrix Addition");
    size_t size = numRows * numCols;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        matrix[i] += mat2.matrix[i];
    }

    return *this;
}

void Matrix::addToRows(const vector<double> &vec) {
    if (vec.size() != numCols) {
        cerr << "Fatal Error: Cannot broadcast vector to matrix rows.\n"
             << "Vector size: " << vec.size()
             << ", Matrix columns: " << numCols << "." << endl;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        size_t offset = i * numCols;
        for (size_t j = 0; j < numCols; j++) {
            matrix[offset+ j] += vec[j];
        }
    }
}


MatrixT Matrix::T() const {
    return MatrixT(*this);
}
