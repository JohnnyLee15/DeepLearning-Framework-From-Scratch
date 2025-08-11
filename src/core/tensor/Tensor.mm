#include "core/tensor/Tensor.h"
#include "core/gpu/GpuEngine.h"
#define TILE_SIZE 8
#define NUM_THREADS 256
#define COARSE_FACTOR 4

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

    MTLSize tgSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(
        inDims[3], 
        (inDims[2] + COARSE_FACTOR - 1) / COARSE_FACTOR, 
        inDims[1] * inDims[0]
    );

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
    [encoder setComputePipelineState:GpuEngine::getConv2dForwardNaivePipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:kernBuf offset:0 atIndex:1];
    [encoder setBuffer:biasBuf offset:0 atIndex:2];
    [encoder setBuffer:outBuf offset:0 atIndex:3];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims)   atIndex:5];
    [encoder setBytes:&outDims length:sizeof(outDims) atIndex:6];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:7];

    MTLSize tgNaiveSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(outDims[2], outDims[1], inDims[0] * kDims[0]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgNaiveSize];
    [encoder endEncoding];
}

void Tensor::maxPool2dGpu(
    MetalBuffer &maxIndices,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor &pooledOutput, 
    id<MTLCommandBuffer> cmdBuf,
    const WindowDims &winIn
) const {
    id<MTLBuffer> inBuf = getGpuData();
    id<MTLBuffer> indBuf = maxIndices.getBuffer();
    id<MTLBuffer> outBuf = pooledOutput.getGpuData();

    uint32_t inDims[4] = {(uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]};
    uint32_t outDims[2] = {(uint32_t) pooledOutput.shape[1], (uint32_t) pooledOutput.shape[2]};
    uint32_t kDims[2] = {(uint32_t) kRows, (uint32_t) kCols};
    uint32_t strideU = (uint32_t) stride;
    uint32_t padding[4] = {
        (uint32_t) winIn.padRows, (uint32_t) winIn.padCols, 
        (uint32_t) winIn.padTop, (uint32_t) winIn.padLeft
    };

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getMaxPool2dPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:indBuf offset:0 atIndex:1];
    [encoder setBuffer:outBuf offset:0 atIndex:2];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:3];
    [encoder setBytes:&outDims length:sizeof(outDims)   atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims) atIndex:5];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&padding length:sizeof(padding) atIndex:7];

    MTLSize gridSize = MTLSizeMake(outDims[1], outDims[0], inDims[0] * inDims[3]);
    MTLSize tgSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
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
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getConv2dWeightsNaivePipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:gradBuf offset:0 atIndex:1];
    [encoder setBuffer:dwBuf offset:0 atIndex:2];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:3];
    [encoder setBytes:&gradDims length:sizeof(gradDims)   atIndex:4];
    [encoder setBytes:&kDims length:sizeof(kDims) atIndex:5];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:6];

    MTLSize tgNaiveSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(kDims[2], kDims[1], inDims[3] * kDims[0]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgNaiveSize];
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
    [encoder setComputePipelineState:GpuEngine::getReduceSumBiasPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:dbBuf offset:0 atIndex:1];
    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:2];

    MTLSize gridSize = MTLSizeMake(gradDims[3], 1, 1);

    NSUInteger tgSize = MIN(gradDims[3], NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Tensor::applyBiasGradConv2D(
    Tensor &biases,
    float scaleFactor,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> gradBuf = getGpuData();
    id<MTLBuffer> biasBuf = biases.getGpuData();
    uint32_t gradDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1],  
        (uint32_t) shape[2], (uint32_t) shape[3]
    };

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getApplyBiasGradConv2DPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:biasBuf offset:0 atIndex:1];
    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:2];
    [encoder setBytes:&scaleFactor length:sizeof(float)  atIndex:3];

    MTLSize gridSize = MTLSizeMake(gradDims[3], 1, 1);
    MTLSize threadSize = MTLSizeMake(NUM_THREADS, 1, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}

void Tensor::padAndUpsampleGradGpu(
    Tensor &outGrad, 
    const WindowDims &winGrad, 
    size_t stride,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> gradBuf = getGpuData();
    id<MTLBuffer> outBuf = outGrad.getGpuData();

    uint32_t batchSize = shape[0];
    uint32_t gradRows = shape[1];
    uint32_t gradCols = shape[2];
    uint32_t numKernals = shape[3];

    uint32_t outRows = (stride > 1) ? stride * (gradRows - 1) + 1 : gradRows;
    uint32_t outCols = (stride > 1) ? stride * (gradCols - 1) + 1 : gradCols;
    outRows += winGrad.padRows;
    outCols += winGrad.padCols;

    outGrad.reShapeInPlace(
        {batchSize, outRows, outCols, numKernals}
    );

    uint32_t gradDims[4] = {batchSize, gradRows, gradCols, numKernals};
    uint32_t outDims[2] = {outRows, outCols};
    uint32_t padding[2] = {(uint32_t) winGrad.padTop, (uint32_t) winGrad.padLeft};
    uint32_t strideU = (uint32_t) stride;

    GpuEngine::fillInt(outBuf, (uint32_t) outGrad.getSize(), cmdBuf, 0.0f);

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getPadAndUpsampleGradPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:outBuf offset:0 atIndex:1];
    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:2];
    [encoder setBytes:&outDims length:sizeof(outDims)  atIndex:3];
    [encoder setBytes:&padding length:sizeof(padding)  atIndex:4];
    [encoder setBytes:&strideU length:sizeof(uint32_t)  atIndex:5];   

    MTLSize tgSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(gradCols, gradRows, batchSize);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

void Tensor::conv2dInputGpu(
    const Tensor &kernals,
    Tensor &dX,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> gradBuf = getGpuData();
    id<MTLBuffer> kBuf = kernals.getGpuData();
    id<MTLBuffer> dxBuf = dX.getGpuData();

    uint32_t gradDims[4] = {
        (uint32_t) shape[0], (uint32_t) shape[1], (uint32_t) shape[2], (uint32_t) shape[3]
    };
    uint32_t kDims[4] = {
        (uint32_t) kernals.shape[0], (uint32_t) kernals.shape[1], 
        (uint32_t) kernals.shape[2], (uint32_t) kernals.shape[3]
    };
    uint32_t dxDims[2] = {(uint32_t) dX.shape[1], (uint32_t) dX.shape[2], };


    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getConv2dInputNaivePipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:kBuf offset:0 atIndex:1];
    [encoder setBuffer:dxBuf offset:0 atIndex:2];

    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:3];
    [encoder setBytes:&kDims length:sizeof(kDims)   atIndex:4];
    [encoder setBytes:&dxDims length:sizeof(dxDims) atIndex:5];

    MTLSize tgNaiveSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(dxDims[1], dxDims[0], gradDims[0] * kDims[3]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgNaiveSize];
    [encoder endEncoding];
}

void Tensor::maxPool2dGradGpu(
    MetalBuffer &maxIndices,
    Tensor &dX,
    id<MTLCommandBuffer> cmdBuf
) const {
    id<MTLBuffer> dxBuf = dX.getGpuData();
    id<MTLBuffer> gradBuf = getGpuData();
    id<MTLBuffer> indBuf = maxIndices.getBuffer();
    uint32_t gradSize = (uint32_t) getSize();

     GpuEngine::fillFloat(dxBuf, (uint32_t) dX.getSize(), cmdBuf, 0.0f);

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getMaxPool2dGradPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:indBuf offset:0 atIndex:1];
    [encoder setBuffer:dxBuf offset:0 atIndex:2];
    [encoder setBytes:&gradSize length:sizeof(uint32_t)  atIndex:3];

    MTLSize gridSize = MTLSizeMake(gradSize, 1, 1);

    NSUInteger tgSize = MIN(gradSize, NUM_THREADS);
    MTLSize threadSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadSize];
    [encoder endEncoding];
}