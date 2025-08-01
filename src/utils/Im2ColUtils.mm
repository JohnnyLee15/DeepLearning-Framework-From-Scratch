#include "utils/Im2ColUtils.h"
#include <cstdint>
#include "core/gpu/GpuEngine.h"

#define TILE_SIZE 8
#define NUM_THREADS 256
#define COARSE_FACTOR 4

void Im2ColUtils::im2Col(
    const Tensor &input,
    Tensor &im2ColInBuf,
    size_t kRows,
    size_t kCols,
    size_t stride,
    const WindowDims &win,
    id<MTLCommandBuffer> cmdBuf
) {
    id<MTLBuffer> inBuf = input.getGpuData();
    id<MTLBuffer> outBuf = im2ColInBuf.getGpuData();

    uint32_t inDims[4] = {
        (uint32_t) input.getShape()[0], (uint32_t) input.getShape()[1], 
        (uint32_t) input.getShape()[2], (uint32_t) input.getShape()[3]
    };

    uint32_t kDims[2] = {(uint32_t) kRows, (uint32_t)  kCols};
    uint32_t outDims[2] = {(uint32_t) win.outRows, (uint32_t) win.outCols};
    uint32_t flatCols = (uint32_t) im2ColInBuf.getShape()[1];
    uint32_t strideU = (uint32_t) stride;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getIm2ColPipe()];

    [encoder setBuffer:inBuf offset:0 atIndex:0];
    [encoder setBuffer:outBuf offset:0 atIndex:1];

    [encoder setBytes:&inDims length:sizeof(inDims)  atIndex:2];
    [encoder setBytes:&kDims length:sizeof(kDims)  atIndex:3];
    [encoder setBytes:&outDims length:sizeof(outDims)   atIndex:4];
    [encoder setBytes:&flatCols length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&strideU length:sizeof(uint32_t) atIndex:6];

    MTLSize tgSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    uint paddedCols = ((win.outCols + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    uint paddedRows = ((win.outRows + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    MTLSize gridSize = MTLSizeMake(paddedCols, paddedRows, inDims[0]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

void Im2ColUtils::addBiasApplyReLUIm2Col(
    Tensor &z, 
    Tensor &a,
    const Tensor &biases, 
    id<MTLCommandBuffer> cmdBuf
) {
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> aBuf = a.getGpuData();
    id<MTLBuffer> bBuf = biases.getGpuData();

    uint32_t numKernals = (uint32_t) z.getShape()[3];
    uint32_t size = (uint32_t) z.getSize();

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getAddBiasApplyReLUIm2ColPipe()];

    [encoder setBuffer:zBuf offset:0 atIndex:0];
    [encoder setBuffer:aBuf offset:0 atIndex:1];
    [encoder setBuffer:bBuf offset:0 atIndex:2];
    [encoder setBytes:&numKernals length:sizeof(uint32_t)  atIndex:3];
    [encoder setBytes:&size length:sizeof(uint32_t)  atIndex:4];
    

    MTLSize grid = MTLSizeMake((size + COARSE_FACTOR - 1) / COARSE_FACTOR, 1, 1);
    MTLSize tg = MTLSizeMake(NUM_THREADS, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];
}

void Im2ColUtils::col2Im(
    const Tensor &grad,
    Tensor &dX,
    size_t gradRows,
    size_t gradCols,
    size_t kRows,
    size_t kCols,
    size_t stride,
    size_t padTop,
    size_t padLeft,
    id<MTLCommandBuffer> cmdBuf
) {
    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];
    [blitEncoder fillBuffer:dX.getGpuData() range:NSMakeRange(0, dX.getSize() * sizeof(float)) value:0.0f];
    [blitEncoder endEncoding];

    id<MTLBuffer> gradBuf = grad.getGpuData();
    id<MTLBuffer> dxBuf = dX.getGpuData();

    uint32_t gradDims[2] = {(uint32_t) gradRows, (uint32_t) gradCols};
    uint32_t kDims[2] = {(uint32_t) kRows, (uint32_t) kCols};
    uint32_t dxDims[4] = {
        (uint32_t) dX.getShape()[0], (uint32_t) dX.getShape()[1], 
        (uint32_t) dX.getShape()[2], (uint32_t) dX.getShape()[3]
    };
    uint32_t padding[2] = {(uint32_t) padTop, (uint32_t) padLeft};
    uint32_t strideU = (uint32_t) stride;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getCol2ImPipe()];

    [encoder setBuffer:gradBuf offset:0 atIndex:0];
    [encoder setBuffer:dxBuf offset:0 atIndex:1];
    [encoder setBytes:&gradDims length:sizeof(gradDims)  atIndex:2];
    [encoder setBytes:&kDims length:sizeof(kDims)  atIndex:3];
    [encoder setBytes:&dxDims length:sizeof(dxDims)  atIndex:4];
    [encoder setBytes:&strideU length:sizeof(uint32_t)  atIndex:5];
    [encoder setBytes:&padding length:sizeof(padding)  atIndex:6];

    MTLSize grid = MTLSizeMake(gradDims[0] * gradDims[1], dxDims[0], 1);
    MTLSize tg = MTLSizeMake(NUM_THREADS, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];
}