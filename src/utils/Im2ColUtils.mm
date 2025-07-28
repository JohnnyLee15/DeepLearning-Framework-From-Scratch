#include "utils/Im2ColUtils.h"
#include <cstdint>
#include "core/gpu/GpuEngine.h"

#define TILE_SIZE 8
#define NUM_THREADS 256

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
    MTLSize gridSize = MTLSizeMake(win.outCols, win.outRows, inDims[0]);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
}

void Im2ColUtils::addBiasIm2Col(Tensor &z, const Tensor &biases, id<MTLCommandBuffer> cmdBuf) {
    id<MTLBuffer> zBuf = z.getGpuData();
    id<MTLBuffer> bBuf = biases.getGpuData();

    uint32_t numKernals = (uint32_t) z.getShape()[3];
    uint32_t size = (uint32_t) z.getShape()[0] * z.getShape()[1] * z.getShape()[2] * z.getShape()[3];

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:GpuEngine::getAddBiasIm2ColPipe()];

    [encoder setBuffer:zBuf offset:0 atIndex:0];
    [encoder setBuffer:bBuf offset:0 atIndex:1];
    [encoder setBytes:&numKernals length:sizeof(uint32_t)  atIndex:2];

    MTLSize grid = MTLSizeMake(size, 1, 1);
    MTLSize tg = MTLSizeMake(MIN(size, NUM_THREADS), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];
}