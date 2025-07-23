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

void Tensor::padWindowInputGpu(
    Tensor &toPad,
    const WindowDims &win,
    id<MTLCommandBuffer> cmdBuf
) const {
    // add fill gpu method
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> padBuf = toPad.getGpuData();

    uint32_t inDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]
    };

    uint32_t padDims[4] = {
        (uint32_t) toPad.shape[0], (uint32_t) toPad.shape[1], 
        (uint32_t) toPad.shape[2], (uint32_t) toPad.shape[3]
    };

    uint32_t padTop = (uint32_t) win.padTop;
    uint32_t padLeft = (uint32_t) win.padLeft;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getPadWindowInputPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:padBuf offset:0 atIndex:1];
    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:2];
    [encoder setBytes:&padDims length:sizeof(padDims)  atIndex:3];
    [encoder setBytes:&padTop length:sizeof(uint32_t)  atIndex:4];
    [encoder setBytes:&padLeft length:sizeof(uint32_t)  atIndex:5];   

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(inDims[2], inDims[1], inDims[0]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

const Tensor& Tensor::padIfNeededGpu(
    Tensor &toPad,
    const WindowDims &win,
    Tensor::Paddings padding,
    id<MTLCommandBuffer> cmdBuf,
    float padVal
) const {
    if (padding == Paddings::NONE) {
        return *this;
    }

    GpuEngine::fillFloat(toPad.getGpuData(), (uint32_t) toPad.getSize(), cmdBuf, padVal);
    padWindowInputGpu(toPad, win, cmdBuf);
    return toPad;
}

void Tensor::conv2dForwardGpu(
    const Tensor &kernals,
    size_t stride,
    Tensor &output,
    const Tensor &biases, 
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> kernBuf = kernals.getGpuData();
    id<MTLBuffer> biasBuf = biases.getGpuData();
    id<MTLBuffer> outBuf = output.getGpuData();

    uint32_t inDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]
    };

    uint32_t kDims[4] = {
        (uint32_t) kernals.shape[0], (uint32_t) kernals.shape[1], 
        (uint32_t) kernals.shape[2], (uint32_t) kernals.shape[3]
    };

    uint32_t outDims[4] = {
        (uint32_t) output.shape[0], (uint32_t) output.shape[1], 
        (uint32_t) output.shape[2], (uint32_t) output.shape[3]
    };

    uint32_t strideU = (uint32_t) stride;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getConv2dForwardPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:kernBuf offset:0 atIndex:1];
    [encoder setBuffer:biasBuf offset:0 atIndex:2];
    [encoder setBuffer:outBuf offset:0 atIndex:3];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims)   atIndex:5];
    [encoder setBytes:&outDims length:sizeof(outDims) atIndex:6];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:7];

    NSUInteger tgDim = 8;
    MTLSize tgSize = MTLSizeMake(tgDim, tgDim, 1);
    NSUInteger numCols = (outDims[2] + tgDim - 1)/tgDim;
    NSUInteger numRows = (outDims[1] + tgDim - 1)/tgDim;
    MTLSize numGroups = MTLSizeMake(numCols, numRows, inDims[0] * kDims[0]);

    [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

void Tensor::maxPool2dGpu(
    MetalBuffer &maxIndices,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor &pooledOutput, 
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> indBuf = maxIndices.getBuffer();
    id<MTLBuffer> outBuf = pooledOutput.getGpuData();

    uint32_t inDims[4] = {(uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]};
    uint32_t outDims[2] = {(uint32_t) pooledOutput.shape[1], (uint32_t) pooledOutput.shape[2]};
    uint32_t kDims[2] = {(uint32_t) kRows, (uint32_t) kCols};
    uint32_t strideU = (uint32_t) stride;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getMaxPool2dPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:indBuf offset:0 atIndex:1];
    [encoder setBuffer:outBuf offset:0 atIndex:2];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:3];
    [encoder setBytes:&outDims length:sizeof(outDims)   atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims) atIndex:5];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:6];

    MTLSize gridSize = MTLSizeMake(outDims[1], outDims[0], inDims[0] * inDims[3]);
    MTLSize tgSize = MTLSizeMake(16, 16, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}