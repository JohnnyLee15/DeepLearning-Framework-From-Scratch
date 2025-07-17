#include "core/activations/Softmax.h"
#include "core/losses/SoftmaxCrossEntropy.h"
#include "core/tensor/Tensor.h"
#include "core/tensor/Matrix.h"
#include "core/gpu/GpuEngine.h"

void Softmax::activateGpu(
    const Tensor &z,
    Tensor &a,
    GpuCommandBuffer cmdBufVoid
) const {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> aBuf = a.getGpuData();

    uint32_t numRows = (uint32_t) z.M().getNumRows();
    uint32_t numCols = (uint32_t) z.M().getNumCols();
    uint32_t dims[2] = {numRows, numCols};

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getActivateSoftmaxPipe()];

    [encoder setBuffer:zBuf offset:0 atIndex:0];
    [encoder setBuffer:aBuf offset:0 atIndex:1];
    [encoder setBytes:&dims length:(2 * sizeof(uint32_t)) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numRows, 1, 1);

    NSUInteger tgSize = MIN(numRows, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Softmax::calculateGradientGpu(
    const Tensor &z, 
    Tensor &dZ, 
    GpuCommandBuffer cmdBuf
) const {
    SoftmaxCrossEntropy::checkInvalidGradientCall();
}
