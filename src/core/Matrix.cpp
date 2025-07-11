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

const vector<double>& Matrix::getFlat() const {
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

Tensor Matrix::operator *(const Matrix &mat2) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    checkSizeMatch(numCols, mat2Rows);

    Tensor product({numRows, mat2Cols});
    vector<double> &prodFlat = product.getFlat();
    const vector<double> &matFlat = tensor.getFlat();
    const vector<double> &mat2Flat = mat2.tensor.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            size_t offsetThis = i * numCols;
            size_t offsetProd = i * mat2Cols;
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matFlat[offsetThis + k]* mat2Flat[k * mat2Cols + j];
            }
            prodFlat[offsetProd + j] = value;
        }
    }

    return product;
}

Tensor Matrix::operator *(const MatrixT &mat2) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();
    checkSizeMatch(numCols,mat2Rows);

    Tensor product({numRows, mat2Cols});
    vector<double> &prodFlat = product.getFlat();
    const vector<double> &matFlat = tensor.getFlat();
    const vector<double> &mat2Flat = mat2.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double value = 0.0;
            size_t offsetThis = i * numCols;
            size_t offsetProd = i * mat2Cols;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += matFlat[offsetThis + k] * mat2Flat[j * mat2Rows+ k];
            }
            prodFlat[offsetProd + j] = value;
        }
    }

    return product;
}

vector<double> Matrix::operator *(const vector<double> &vec) const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    checkSizeMatch(numCols, vec.size());

    vector<double> product(numRows, 0.0);
    const vector<double> &matFlat = tensor.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++) {
        double value = 0.0;
        size_t offset = i * numCols;
        for (size_t j = 0; j < numCols; j++) {
            value += matFlat[offset + j] * vec[j];
        }
        product[i] = value;
    }

    return product;
}

vector<double> Matrix::colSums() const {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    vector<double> colSumsVec(numCols, 0.0);
    const vector<double> &matFlat = tensor.getFlat();

    #pragma omp parallel 
    {
        vector<double> threadColSums(numCols, 0.0);

        #pragma omp for
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                threadColSums[j] += matFlat[i * numCols + j];
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

void Matrix::addToRows(const vector<double> &vec) {
    size_t numRows = getNumRows();
    size_t numCols = getNumCols();
    if (vec.size() != numCols) {
        ConsoleUtils::fatalError(
            string("Cannot broadcast vector to matrix rows.\n") +
            "Vector size: " + to_string(vec.size()) +
            ", Matrix columns: " + to_string(numCols) + "."
        );
    }

    vector<double> &matFlat = tensor.getFlat();
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            matFlat[i * numCols + j] += vec[j];
        }
    }
}

MatrixT Matrix::T() const {
    return MatrixT(*this);
}
