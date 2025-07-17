#include "core/tensor/Matrix.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"
#include "core/tensor/MatrixT.h"

id<MTLBuffer> Matrix::getGpuData() const {
    return tensor.getGpuData();
}

void Matrix::mmGpu(
    const Matrix &mat2,
    Tensor &prod,
    id<MTLCommandBuffer> cmdBuf
) const {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatMatPipe(),
        cmdBuf
    );
}

void Matrix::mmTGpu(
    const MatrixT &mat2,
    Tensor &prod,
    id<MTLCommandBuffer> cmdBuf
) const {
    Matrix::checkSizeMatch(getNumCols(), mat2.getNumRows());
    matMatEngine(
        getGpuData(),
        mat2.getGpuData(),
        prod.getGpuData(),
        getNumRows(),
        getNumCols(),
        mat2.getNumCols(),
        GpuEngine::getMatMatTPipe(),
        cmdBuf
    );
} 

void Matrix::matMatEngine(
    id<MTLBuffer> mat1Buf,
    id<MTLBuffer> mat2Buf,
    id<MTLBuffer> prodBuf,
    size_t mat1Rows,
    size_t mat1Cols,
    size_t mat2Cols,
    id<MTLComputePipelineState> pipeline,
    id<MTLCommandBuffer> cmdBuf
) {
    uint32_t mat1RowsU = (uint32_t) mat1Rows;
    uint32_t mat1ColsU = (uint32_t) mat1Cols;
    uint32_t mat2ColsU = (uint32_t) mat2Cols;
    uint32_t dims[3] = {mat1RowsU, mat1ColsU, mat2ColsU};

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:mat1Buf offset:0 atIndex:0];
    [encoder setBuffer:mat2Buf offset:0 atIndex:1];
    [encoder setBuffer:prodBuf offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];

    NSUInteger tgDim = 16;
    MTLSize threadGroupSize = MTLSizeMake(tgDim, tgDim, 1);

    NSUInteger numTgRows = (mat1Rows + tgDim - 1)/tgDim;
    NSUInteger numTgCols = (mat2Cols + tgDim - 1)/tgDim;
    MTLSize numThreadGroups = MTLSizeMake(numTgCols, numTgRows, 1);

    [encoder dispatchThreadgroups:numThreadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}

void Matrix::colSumsGpu(Tensor &vec, id<MTLCommandBuffer> cmdBuf) const {
    id<MTLBuffer> matBuf = getGpuData();
    id<MTLBuffer> vecBuf = vec.getGpuData();

    uint32_t numRows = (uint32_t) getNumRows();
    uint32_t numCols = (uint32_t) getNumCols();
    uint32_t dims[2] = {numRows, numCols};

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getColSumsPipe()];

    [encoder setBuffer:matBuf offset:0 atIndex:0];
    [encoder setBuffer:vecBuf offset:0 atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numCols, 1, 1);

    NSUInteger tgCols = MIN(numCols, 256);
    MTLSize threadSize = MTLSizeMake(tgCols, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Matrix::addToRowsGpu(const Tensor &vec, id<MTLCommandBuffer> cmdBuf) {
    id<MTLBuffer> matBuf = getGpuData();
    id<MTLBuffer> vecBuf = vec.getGpuData();

    uint32_t numRows = (uint32_t) getNumRows();
    uint32_t numCols = (uint32_t) getNumCols();
    uint32_t dims[2] = {numRows, numCols};

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getAddToRowsPipe()];

    [encoder setBuffer:matBuf offset:0 atIndex:0];
    [encoder setBuffer:vecBuf offset:0 atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];

    NSUInteger tgDim = 16;
    MTLSize threadGroupSize = MTLSizeMake(tgDim, tgDim, 1);
    MTLSize gridSize = MTLSizeMake(numCols, numRows, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}