#include "core/activations/ReLU.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void ReLU::activateGpu(
    const Tensor &z,
    Tensor &a,
    GpuCommandBuffer cmdBufVoid
) const {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> aBuf = a.getGpuData();

    uint32_t size = (uint32_t) z.getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getActivateReLUPipe()];

    [encoder setBuffer:zBuf offset:0 atIndex:0];
    [encoder setBuffer:aBuf offset:0 atIndex:1];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void ReLU::calculateGradientGpu(
    const Tensor &z, 
    Tensor &dZ,
    GpuCommandBuffer cmdBufVoid
) const {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> dzBuf = dZ.getGpuData();

    uint32_t size = dZ.getSize();

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
}