#include "core/Tensor.h"
#include "activations/ReLU.h"

void ReLU::calculateGradientGpu(const Tensor &z, Tensor &dZ) const {
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> dzBuf = dZ.getGpuData();

    uint32_t size = dZ.getSize();

    id<MTLCommandBuffer> cmdBuf = [GpuEngine::getCmdQueue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:GpuEngine::getCalculateReluGradPipe()];
    [encoder setBuffer:zBuf offset:0 atIndex:0];
    [encoder setBuffer:dzBuf offset:0 atIndex:1];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];

    [encoder endEncoding];
    [cmdBuf commit];
}