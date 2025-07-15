#include "core/Matrix.h"
#include "core/MatrixT.h"
#include "utils/ConsoleUtils.h"

Matrix::Matrix(Tensor &tensor) : tensor(tensor) {}

size_t Matrix::getNumCols() const {
    return tensor.getShape()[1];
}

size_t Matrix::getNumRows() const {
    return tensor.getShape()[0];
}

const vector<float>& Matrix::getFlat() const {
    return tensor.getFlat();
}

void Matrix::checkSizeMatch(size_t mat1Cols, size_t mat2Rows) {
    if (mat1Cols != mat2Rows) {
        ConsoleUtils::fatalError(
            string("Matrix multiplication size mismatch.\n") +
            "Left matrix columns: " + to_string(mat1Cols) + 
            ", Right matrix rows: " + to_string(mat2Rows) + "."
        );
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
        ConsoleUtils::fatalError(
            operation + " requires matrices of the same shape.\n" +
            "Left matrix shape: (" + to_string(mat1Rows) + ", " + to_string(mat1Cols) + "), " +
            "Right matrix shape: (" + to_string(mat2Rows) + ", " + to_string(mat2Cols) + ")."
        );
    }
}

void Matrix::mm(const Matrix &mat2, Tensor &prod) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    checkSizeMatch(numCols, mat2Rows);

    vector<float> &prodFlat = prod.getFlat();
    const vector<float> &matFlat = tensor.getFlat();
    const vector<float> &mat2Flat = mat2.tensor.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            size_t offsetThis = i * numCols;
            size_t offsetProd = i * mat2Cols;
            float value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matFlat[offsetThis + k]* mat2Flat[k * mat2Cols + j];
            }
            prodFlat[offsetProd + j] = value;
        }
    }
    
}

void Matrix::mmT(const MatrixT &mat2, Tensor &prod) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    checkSizeMatch(numCols,mat2Rows);

    vector<float> &prodFlat = prod.getFlat();
    const vector<float> &matFlat = tensor.getFlat();
    const vector<float> &mat2Flat = mat2.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            float value = 0.0;
            size_t offsetThis = i * numCols;
            size_t offsetProd = i * mat2Cols;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matFlat[offsetThis + k] * mat2Flat[j * mat2Rows+ k];
            }
            prodFlat[offsetProd + j] = value;
        }
    }
    
}

void Matrix::colSums(Tensor &vec) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    vector<float> &vecFlat = vec.getFlat();
    fill(vecFlat.begin(), vecFlat.end(), 0.0f);
    const vector<float> &matFlat = tensor.getFlat();

    #pragma omp parallel 
    {
        vector<float> threadColSums(numCols, 0.0);

        #pragma omp for
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                threadColSums[j] += matFlat[i * numCols + j];
            }
        }
        
        #pragma omp critical 
        {
            for (size_t j = 0; j < numCols; j++) {
                vecFlat[j] += threadColSums[j];
            }
        }
    }
    
}

void Matrix::addToRows(const Tensor &vec) {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    const vector<float> &vecFlat = vec.getFlat();
    if (vec.getSize() != numCols) {
        ConsoleUtils::fatalError(
            string("Cannot broadcast vector to matrix rows.\n") +
            "Vector size: " + to_string(vec.getSize()) +
            ", Matrix columns: " + to_string(numCols) + "."
        );
    }

    vector<float> &matFlat = tensor.getFlat();
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            matFlat[i * numCols + j] += vecFlat[j];
        }
    }
    
}

MatrixT Matrix::T() const {
    return MatrixT(*this);
}