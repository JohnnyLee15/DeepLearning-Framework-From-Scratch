#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8
#define CHANNEL_SLICE 4
#define MAX_KERNEL 7

kernel void hadamard(
    device float *ten1 [[ buffer(0) ]],
    device const float *ten2 [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) 
        return;

    ten1[gid] *= ten2[gid];
}

kernel void applyGrad(
    device float *param [[ buffer(0) ]],
    device const float *grad [[ buffer(1) ]],
    constant float &scaleFactor [[ buffer(2) ]],
    constant uint &size [[ buffer(3) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    param[gid] += (scaleFactor * grad[gid]);
}

kernel void padWindowInput(
    device const float *input [[ buffer(0) ]],
    device float *toPad [[ buffer(1) ]],
    constant uint4 &inDims [[ buffer(2)]],
    constant uint4 &padDims [[ buffer(3) ]],
    constant uint &padTop [[ buffer(4) ]],
    constant uint &padLeft [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint numSamples = inDims[0];
    uint inRows = inDims[1];
    uint inCols = inDims[2];
    uint depth = inDims[3];

    uint newRows = padDims[1];
    uint newCols = padDims[2];

    uint n = gid.z;
    uint r = gid.y;
    uint c = gid.x;
    
    if (n >= numSamples || r >= inRows || c >= inCols)
        return;

    for (uint d = 0; d < depth; d++) {
        uint inIdx = (((n * inRows + r) * inCols + c) * depth) + d;
        uint padIdx = (((n * newRows + (r + padTop)) * newCols + (c + padLeft)) * depth)+ d;
        toPad[padIdx] = input[inIdx];
    }
}

kernel void conv2dForward(
    device const float *input [[ buffer(0) ]],
    device const float *kernals [[ buffer(1) ]],
    device const float *biases [[ buffer(2) ]],
    device float *output [[ buffer(3) ]],
    constant uint4 &inputDims [[ buffer(4) ]],
    constant uint4 &kernalDims [[ buffer(5) ]],
    constant uint4 &outputDims [[ buffer(6) ]],
    constant uint &stride [[ buffer(7) ]],
    uint3 gid [[ thread_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]]
) {
    uint numKernals = kernalDims[0];
    uint kRows = kernalDims[1];
    uint kCols = kernalDims[2];

    if (kRows > MAX_KERNEL || kCols > MAX_KERNEL)
        return;

    uint numSamples = inputDims[0];
    uint inRows = inputDims[1];
    uint inCols = inputDims[2];
    uint inDepth = inputDims[3];

    uint outRows = outputDims[1];
    uint outCols = outputDims[2];

    uint n = gid.z / numKernals;
    uint r = gid.y;
    uint c = gid.x;
    uint o = gid.z % numKernals;

    uint tRow = tid.y;
    uint tCol = tid.x;
    
    uint baseRow = (r - tRow) * stride;
    uint baseCol = (c - tCol) * stride;      

    bool inBounds = (n < numSamples) && (r < outRows) && (c < outCols);

    uint numSlices = (inDepth + CHANNEL_SLICE - 1)/CHANNEL_SLICE;
    threadgroup float patch[TILE_SIZE + MAX_KERNEL - 1][TILE_SIZE + MAX_KERNEL - 1][CHANNEL_SLICE];

    float val = inBounds ? biases[o] : 0.0f;
    for (uint p = 0; p < numSlices; p++) {

        for (uint i = tRow; i < kRows + TILE_SIZE - 1; i+= TILE_SIZE) {
            uint inRow = baseRow+ i;
            for (uint j = tCol; j < kCols + TILE_SIZE - 1; j += TILE_SIZE) {
                uint inCol = baseCol + j;
                for (uint d = 0; d < CHANNEL_SLICE; d++) {
                    uint inChan = p*CHANNEL_SLICE + d;
                    float sharedVal = 0.0f;

                    if (inRow < inRows && inCol < inCols && inChan < inDepth) {
                        uint inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + inChan;
                        sharedVal = input[inIdx];
                    }

                    patch[i][j][d] = sharedVal;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint pRow = tRow * stride;
        uint pCol = tCol * stride;
        for (uint i = 0; i < kRows; i++) {
            for (uint j = 0; j < kCols; j++) {
                for (uint d = 0; d < CHANNEL_SLICE; d++) {
                    uint inChan = p*CHANNEL_SLICE + d;
                    if (inChan < inDepth) {
                        uint kIdx = (((o * kRows + i) * kCols + j) * inDepth) + inChan;
                        val += (patch[pRow + i][pCol + j][d] * kernals[kIdx]);
                    }
                }
            }
        }
        

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        uint outIdx = (((n * outRows + r) * outCols + c) * numKernals) + o;
        output[outIdx] = val;
    }
}

kernel void maxPool2d(
    device const float *input [[ buffer(0) ]],
    device uint *maxIndices [[ buffer(1) ]],
    device float *output [[ buffer(2) ]],
    constant uint4 &inDims [[ buffer(3) ]],
    constant uint2 &outDims [[ buffer(4) ]],
    constant uint2 &kDims [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint numSamples = inDims[0];
    uint inRows = inDims[1];
    uint inCols = inDims[2];
    uint depth = inDims[3];

    uint outRows = outDims[0];
    uint outCols = outDims[1];

    uint kRows = kDims[0];
    uint kCols = kDims[1];

    uint n = gid.z/depth;
    uint r = gid.y;
    uint c = gid.x;
    uint d = gid.z % depth;
    
    if (n >= numSamples || r >= outRows || c >= outCols)
        return;

    float maxVal = -FLT_MAX;
    uint maxIdx = UINT_MAX;

    for (uint i = 0; i < kRows; i++) {
        uint inRow = r * stride + i;
        for (uint j = 0; j < kCols; j++) {
            uint inCol = c * stride + j;
            uint idx = (((n * inRows + inRow) * inCols + inCol) * depth) + d;

            float val = input[idx];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = idx;
            }
        }
    }

    uint outIdx = (((n * outRows + r) * outCols + c) * depth) + d;
    output[outIdx] = maxVal;
    maxIndices[outIdx] = maxIdx;
}
