#include "core/MatrixT.h"
#include "core/Matrix.h"

MatrixT::MatrixT(const Matrix &matrix) :
    numRows(matrix.getNumCols()), numCols(matrix.getNumRows()), matrix(matrix) {}

size_t MatrixT::getNumRows() const {
    return numRows;
}

size_t MatrixT::getNumCols() const {
    return numCols;
}

const vector<float>& MatrixT::getFlat() const {
    return matrix.getFlat();
}

void MatrixT::mTm(const Matrix &mat2, Tensor &prod) const {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __OBJC__
            mTmGpu(mat2, prod);
        #endif
    } else {
        Matrix::checkSizeMatch(numCols, mat2.getNumRows());

        size_t mat2Rows = mat2.getNumRows();
        size_t mat2Cols = mat2.getNumCols();

        vector<float> &productFlat = prod.getFlat();
        const vector<float> &mat2Flat = mat2.getFlat();
        const vector<float> &flat = matrix.getFlat();

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < mat2Cols; j++) {
                size_t offsetProd = i * mat2Cols;
                float value = 0.0;
                for (size_t k = 0; k < mat2Rows; k++) {
                    value += flat[k*  numRows + i] * mat2Flat[k*mat2Cols + j];
                }
                productFlat[offsetProd + j] = value;
            }
        }
    }
}

void MatrixT::mTmT(const MatrixT &mat2, Tensor &prod) const {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __OBJC__
            mTmTGpu(mat2, prod);
        #endif
    } else {
        Matrix::checkSizeMatch(numCols, mat2.numRows);

        size_t mat2Rows = mat2.numRows;
        size_t mat2Cols = mat2.numCols;

        vector<float> &productFlat = prod.getFlat();
        const vector<float> &mat2Flat = mat2.getFlat();
        const vector<float> &flat = matrix.getFlat();
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < mat2Cols; j++) {
                size_t offset = j*mat2Rows;
                size_t offsetProd = i * mat2Cols;
                float value = 0.0;
                for (size_t k = 0; k < mat2Rows; k++) {
                    value += flat[k*numRows + i] * mat2Flat[offset + k];
                }
                productFlat[offsetProd + j] = value;
            }
        }
    }
}