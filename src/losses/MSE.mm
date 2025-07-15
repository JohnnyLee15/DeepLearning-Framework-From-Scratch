#include "losses/MSE.h"
#include "core/Tensor.h"

void MSE::calculateGradientGpu(  
    const Tensor &targets, 
    const Tensor &a,
    Tensor &dL,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> targetsBuf = targets.getGpuData();
    id<MTLBuffer> aBuf = a.getGpuData();
    id<MTLBuffer> dlBuf = dL.getGpuData();
    uint32_t size = (uint32_t) a.getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:GpuEngine::getCalculateMSEGradPipe()];
    [encoder setBuffer:targetsBuf offset:0 atIndex:0];
    [encoder setBuffer:aBuf offset:0 atIndex:1];
    [encoder setBuffer:dlBuf offset:0 atIndex:2];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:3];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

