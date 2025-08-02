#include "core/tensor/MatrixT.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/gpu/GpuEngine.h"

#define MEDIUM_TILE 16

#define TILE_MAT1_ROWS 64
#define TILE_MAT2_COLS 64

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

void MatrixT::applyKernelGrads(
    const Matrix &mat2,
    Tensor &kernels,
    float scaleFactor,
    id<MTLCommandBuffer> cmdBuf
) const {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    uint32_t mat1RowsU = (uint32_t) getNumRows();
    uint32_t mat1ColsU = (uint32_t) getNumCols();
    uint32_t mat2ColsU = (uint32_t) mat2.getNumCols();
    uint32_t dims[3] = {mat1RowsU, mat1ColsU, mat2ColsU};

    id<MTLBuffer> mat1Buf = getGpuData();
    id<MTLBuffer> mat2Buf = mat2.getGpuData();
    id<MTLBuffer> kBuf = kernels.getGpuData();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:GpuEngine::getApplyKernelGradsPipe()];
    [encoder setBuffer:mat1Buf offset:0 atIndex:0];
    [encoder setBuffer:mat2Buf offset:0 atIndex:1];
    [encoder setBuffer:kBuf offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder setBytes:&scaleFactor length:sizeof(float) atIndex:4];

    MTLSize threadGroupSize = MTLSizeMake(MEDIUM_TILE, MEDIUM_TILE, 1);

    NSUInteger numTgRows = (mat1RowsU + TILE_MAT1_ROWS - 1)/TILE_MAT1_ROWS;
    NSUInteger numTgCols = (mat2ColsU + TILE_MAT2_COLS - 1)/TILE_MAT2_COLS;
    MTLSize numThreadGroups = MTLSizeMake(numTgCols, numTgRows, 1);

    [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}