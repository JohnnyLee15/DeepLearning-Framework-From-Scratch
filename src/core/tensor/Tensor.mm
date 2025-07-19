#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"

void Tensor::initGpuTensor() {
    size_t bytes = getSize() * sizeof(float);
    dataGpu = MetalBuffer(data.data(), bytes);
}

id<MTLBuffer> Tensor::getGpuData() {
    return dataGpu.getBuffer();
}

const id<MTLBuffer> Tensor::getGpuData() const {
    return dataGpu.getBuffer();
}

void Tensor::uploadToGpu() {
    size_t bytes = getSize() * sizeof(float);
    dataGpu.uploadFromHost(data.data(), bytes);
}

void Tensor::downloadFromGpu() {
    size_t bytes = getSize() * sizeof(float);
    dataGpu.downloadToHost(data.data(), bytes);
}

void Tensor::hadamardGpu(const Tensor &ten2, id<MTLCommandBuffer> cmdBuf) {
    id<MTLBuffer> ten1Buf = getGpuData();
    id<MTLBuffer> ten2Buf = ten2.getGpuData();
    uint32_t size = (uint32_t) getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getHadamardPipe()];

    [encoder setBuffer:ten1Buf offset:0 atIndex:0];
    [encoder setBuffer:ten2Buf offset:0 atIndex:1];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Tensor::applyGradGpu(
    const Tensor &grad, 
    float scaleFactor, 
    id<MTLCommandBuffer> cmdBuf
) {
    id<MTLBuffer> paramBuf = getGpuData();
    id<MTLBuffer> gradBuf = grad.getGpuData();
    uint32_t size = (uint32_t) getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getApplyGradPipe()];

    [encoder setBuffer:paramBuf offset:0 atIndex:0];
    [encoder setBuffer:gradBuf offset:0 atIndex:1];
    [encoder setBytes:&scaleFactor length:sizeof(float) atIndex:2];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:3];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, 256);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}