#include "core/activations/ReLU.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

#define COARSE_FACTOR 4
#define NUM_THREADS 256

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

    NSUInteger tgSize = MIN(size, NUM_THREADS);
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

    NSUInteger tgSize = MIN(size, NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void ReLU::backpropGpu(
    const Tensor &a,  
    Tensor &grad,        
    GpuCommandBuffer cmdBufVoid
) const {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    id<MTLBuffer> aBuf = a.getGpuData();
    id<MTLBuffer> gradBuf = grad.getGpuData();

    uint32_t size = grad.getSize();
    uint32_t gridWidth = (size + COARSE_FACTOR - 1) / COARSE_FACTOR;
    MTLSize gridSize = MTLSizeMake(gridWidth, 1, 1);
    MTLSize threadSize = MTLSizeMake(NUM_THREADS, 1, 1);

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getBackpropReLUPipe()];
    
    [encoder setBuffer:aBuf offset:0 atIndex:0];
    [encoder setBuffer:gradBuf offset:0 atIndex:1];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&gridWidth length:sizeof(uint32_t) atIndex:3];

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}