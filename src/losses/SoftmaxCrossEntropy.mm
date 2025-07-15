#include "core/Tensor.h"
#include "core/Matrix.h"
#include "losses/MSE.h"
#include "losses/SoftmaxCrossEntropy.h"

void SoftmaxCrossEntropy::calculateGradientGpu(  
    const Tensor &targets, 
    const Tensor &a,
    Tensor &dL,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> targetsBuf = targets.getGpuData();
    id<MTLBuffer> aBuf = a.getGpuData();
    id<MTLBuffer> dlBuf = dL.getGpuData();

    uint32_t numRows = (uint32_t) a.M().getNumRows();
    uint32_t numCols = (uint32_t) a.M().getNumCols();
    uint32_t size = numRows * numCols;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:GpuEngine::getCalculateSoftmaxCrossEntropyGradPipe()];
    [encoder setBuffer:targetsBuf offset:0 atIndex:0];
    [encoder setBuffer:aBuf offset:0 atIndex:1];
    [encoder setBuffer:dlBuf offset:0 atIndex:2];
    [encoder setBytes:&numRows length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&numCols length:sizeof(uint32_t) atIndex:4];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}
