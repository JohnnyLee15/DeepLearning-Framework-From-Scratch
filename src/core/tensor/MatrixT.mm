#include "core/tensor/MatrixT.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/gpu/GpuEngine.h"

id<MTLBuffer> MatrixT::getGpuData() const {
    return matrix.getGpuData();
}

void MatrixT::mTmGpu(
    const Matrix &mat2,
    Tensor &prod,
    id<MTLCommandBuffer> cmdBuf
) const {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    Matrix::matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatTMatPipe(),
        cmdBuf
    );
}

void MatrixT::mTmTGpu(
    const MatrixT &mat2,
    Tensor &prod,
    id<MTLCommandBuffer> cmdBuf
) const {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    Matrix::matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatTMatTPipe(),
        cmdBuf
    );
}
