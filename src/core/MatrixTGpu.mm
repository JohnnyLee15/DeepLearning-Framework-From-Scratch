#include "core/MatrixT.h"
#include "core/Tensor.h"
#include "core/Matrix.h"

id<MTLBuffer> MatrixT::getGpuData() const {
    return matrix.getGpuData();
}

void MatrixT::mTmGpu(
    const Matrix &mat2,
    Tensor &prod
) {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    Matrix::matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatTMatPipe()
    );
}

void MatrixT::mTmTGpu(
    const MatrixT &mat2,
    Tensor &prod
) {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    Matrix::matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatTMatTPipe()
    );
}
