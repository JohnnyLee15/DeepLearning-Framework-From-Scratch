#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"
#define SMALL_TILE 8
#define MEDIUM_TILE 16
#define CHANNEL_SLICE 4
#define MAX_KERNEL 7
#define MED_PATCH_DIM ((SMALL_TILE - 1) * 2 + MAX_KERNEL)
#define SMALL_PATCH_DIM ((SMALL_TILE - 1) + MAX_KERNEL)
#define NUM_THREADS 256

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

void Tensor::copyGpu(Tensor& out, id<MTLCommandBuffer> cmdBuf) const {
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> outBuf = out.getGpuData();
    uint32_t size = (uint32_t) getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getCopyTensorPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:outBuf offset:0 atIndex:1];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);

    NSUInteger tgSize = MIN(size, NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
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

    NSUInteger tgSize = MIN(size, NUM_THREADS);
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

    NSUInteger tgSize = MIN(size, NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Tensor::padWindowInputGpu(
    Tensor &toPad,
    const WindowDims &win,
    id<MTLCommandBuffer> cmdBuf
) const {
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

    MTLSize tgSize = MTLSizeMake(MEDIUM_TILE, MEDIUM_TILE, 1);
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

bool Tensor::setConv2dForwardPipe(
    id<MTLComputeCommandEncoder> encoder,
    uint32_t strideU,
    uint32_t kRows,
    uint32_t kCols,
    uint32_t baseDim
) const {
    if ((baseDim + kRows <= SMALL_PATCH_DIM) && (baseDim + kCols <= SMALL_PATCH_DIM)) {
        [encoder setComputePipelineState:GpuEngine::getConv2dForwardFastPipe()];
        return false;
    } else if ((baseDim + kRows <= MED_PATCH_DIM) && (baseDim + kCols <= MED_PATCH_DIM)) {
        [encoder setComputePipelineState:GpuEngine::getConv2dForwardMedPipe()];
        return false;
    } 

    [encoder setComputePipelineState:GpuEngine::getConv2dForwardNaivePipe()];
    return true;
}

void Tensor::setConv2dForwardThreads(
    id<MTLComputeCommandEncoder> encoder,
    bool naive,
    uint32_t outRows,
    uint32_t outCols,
    uint32_t zDim
) const {
    if (naive) {
        MTLSize tgNaiveSize = MTLSizeMake(MEDIUM_TILE, MEDIUM_TILE, 1);
        MTLSize gridSize = MTLSizeMake(outCols, outRows, zDim);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgNaiveSize];
    } else {
        MTLSize tgFastSize = MTLSizeMake(SMALL_TILE, SMALL_TILE, 1);

        NSUInteger numCols = (outCols + SMALL_TILE - 1)/SMALL_TILE;
        NSUInteger numRows = (outRows + SMALL_TILE - 1)/SMALL_TILE;

        MTLSize numGroups = MTLSizeMake(numCols, numRows, zDim);

        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:tgFastSize];
    }
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
    size_t baseDim = (SMALL_TILE - 1) * strideU;
    bool naive = setConv2dForwardPipe(encoder, strideU, kDims[1], kDims[2], baseDim);

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:kernBuf offset:0 atIndex:1];
    [encoder setBuffer:biasBuf offset:0 atIndex:2];
    [encoder setBuffer:outBuf offset:0 atIndex:3];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims)   atIndex:5];
    [encoder setBytes:&outDims length:sizeof(outDims) atIndex:6];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:7];

    setConv2dForwardThreads(encoder, naive, outDims[1], outDims[2], inDims[0] * kDims[0]);
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
    MTLSize tgSize = MTLSizeMake(MEDIUM_TILE, MEDIUM_TILE, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

bool Tensor::setConv2dWeightsPipe(
    id<MTLComputeCommandEncoder> encoder,
    uint32_t strideU,
    uint32_t kRows,
    uint32_t kCols,
    uint32_t baseDim
) const {
    // if ((baseDim + kRows <= SMALL_PATCH_DIM) && (baseDim + kCols <= SMALL_PATCH_DIM)) {
    //     [encoder setComputePipelineState:GpuEngine::getConv2dWeightsFastPipe()];
    //     return false;
    // } else if ((baseDim + kRows <= MED_PATCH_DIM) && (baseDim + kCols <= MED_PATCH_DIM)) {
    //     [encoder setComputePipelineState:GpuEngine::getConv2dWeightsMedPipe()];
    //     return false;
    // } 

    [encoder setComputePipelineState:GpuEngine::getConv2dWeightsNaivePipe()];
    return true;
}

void Tensor::setConv2dWeightsThreads(
    id<MTLComputeCommandEncoder> encoder,
    bool naive,
    uint32_t outRows,
    uint32_t outCols,
    uint32_t zDim
) const {
    // if (naive) {
        MTLSize tgNaiveSize = MTLSizeMake(MEDIUM_TILE, MEDIUM_TILE, 1);
        MTLSize gridSize = MTLSizeMake(outCols, outRows, zDim);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgNaiveSize];
    // } else {
    //     MTLSize tgFastSize = MTLSizeMake(SMALL_TILE, SMALL_TILE, 1);

    //     NSUInteger numCols = (outCols + SMALL_TILE - 1)/SMALL_TILE;
    //     NSUInteger numRows = (outRows + SMALL_TILE - 1)/SMALL_TILE;

    //     MTLSize numGroups = MTLSizeMake(numCols, numRows, zDim);

    //     [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:tgFastSize];
    // }
}

void Tensor::conv2dWeightsGpu(
    const Tensor &grad,
    size_t numKernals,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor &dW,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> gradBuf = grad.getGpuData();
    id<MTLBuffer> dwBuf = dW.getGpuData();

    uint32_t inDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]
    };
    uint32_t gradDims[2] = {(uint32_t) grad.shape[1], (uint32_t) grad.shape[2], };
    uint32_t kDims[3] = {(uint32_t) numKernals, (uint32_t) kRows, (uint32_t) kCols};

    uint32_t strideU = (uint32_t) stride;

    size_t baseDim = (SMALL_TILE - 1) * strideU;
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    bool naive = setConv2dWeightsPipe(encoder, strideU, kDims[1], kDims[2], baseDim);

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:gradBuf offset:0 atIndex:1];
    [encoder setBuffer:dwBuf offset:0 atIndex:2];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:3];
    [encoder setBytes:&gradDims length:sizeof(gradDims)   atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims) atIndex:5];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:6];

    setConv2dWeightsThreads(encoder, naive, kRows, kCols, inDims[3] * kDims[0]);
    [encoder endEncoding];
}

void Tensor::reduceSumBiasGpu(
    Tensor &dB,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> gradBuf = getGpuData();
    id<MTLBuffer> dbBuf = dB.getGpuData();
    uint32_t gradDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1],  
        (uint32_t) shape[2], (uint32_t) shape[3]
    };

    GpuEngine::fillFloat(dbBuf, (uint32_t) dB.getSize(), cmdBuf, 0.0f);

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getReduceBiasSumPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:dbBuf offset:0 atIndex:1];
    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:2];

    MTLSize gridSize = MTLSizeMake(gradDims[3], 1, 1);

    NSUInteger tgSize = MIN(gradDims[3], NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}