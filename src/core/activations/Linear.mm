#include "core/activations/Linear.h"
#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void Linear::activateGpu(
    const Tensor &z,
    Tensor &a,
    GpuCommandBuffer cmdBufVoid
) const {
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    z.copyGpu(a, cmdBuf);
}

void Linear::calculateGradientGpu(
    const Tensor &z, 
    Tensor &dZ,
    GpuCommandBuffer cmdBufVoid
) const {
    (void) z;
    id<MTLCommandBuffer> cmdBuf = (id<MTLCommandBuffer>) cmdBufVoid;
    id<MTLBuffer> dzBuf = dZ.getGpuData();
    uint32_t size = dZ.getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getCalculateLinearGradPipe()];

    [encoder setBuffer:dzBuf offset:0 atIndex:0];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:1];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}