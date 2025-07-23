#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8
#define CHANNEL_SLICE 4
#define MAX_KERNEL 7
#define MED_PATCH_DIM ((TILE_SIZE - 1) * 2 + MAX_KERNEL)
#define SMALL_PATCH_DIM ((TILE_SIZE - 1) + MAX_KERNEL)

static inline packed_float4 loadConv2dFwdPatch(
    device const float *input,
    uint inRow,
    uint inRows,
    uint inCol,
    uint inCols,
    uint inDepth,
    uint baseChan,
    uint n
) {
    if (inRow < inRows && inCol < inCols && baseChan + 3 < inDepth) {
        uint inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + baseChan;
        return  *((device packed_float4*)&input[inIdx]);

    } 

    packed_float4 sharedVals = packed_float4(0.0f);
    for (uint d = 0; d < CHANNEL_SLICE; d++) {
        uint inChan = baseChan + d;

        if (inRow < inRows && inCol < inCols && inChan < inDepth) {
            uint inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + inChan;
            sharedVals[d] = input[inIdx];
            
        } else {
            sharedVals[d] = 0.0f;
        }
    }
    return sharedVals;
}

static inline float conv2dFwdDot(
    uint kIdx,
    packed_float4 patchVec,
    device const float *kernals,
    uint baseChan,
    uint inDepth
) {
    packed_float4 kvec = *((device packed_float4*)&kernals[kIdx]);
    packed_float4 mask = packed_float4(
        (baseChan + 0 < inDepth) ? 1.0f : 0.0f,
        (baseChan + 1 < inDepth) ? 1.0f : 0.0f,
        (baseChan + 2 < inDepth) ? 1.0f : 0.0f,
        (baseChan + 3 < inDepth) ? 1.0f : 0.0f
    );
    kvec *= mask;
    return dot(patchVec, kvec);
}

static inline void conv2dForward(
    device const float *input,
    device const float *kernals,
    device const float *biases,
    device float *output,
    constant uint4 &inputDims,
    constant uint4 &kernalDims,
    constant uint4 &outputDims,
    uint stride,
    uint3 gid,
    uint3 tid,
    threadgroup packed_float4 *patch,
    uint patchCols
) {
    uint numKernals = kernalDims[0];
    uint kRows = kernalDims[1];
    uint kCols = kernalDims[2];

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
    uint pRow = tRow * stride;
    uint pCol = tCol * stride;
    
    uint baseRow = (r - tRow) * stride;
    uint baseCol = (c - tCol) * stride;      

    bool inBounds = (n < numSamples) && (r < outRows) && (c < outCols);
    uint numSlices = (inDepth + CHANNEL_SLICE - 1)/CHANNEL_SLICE;

    float val = inBounds ? biases[o] : 0.0f;
    for (uint p = 0; p < numSlices; p++) {
        uint baseChan = p*CHANNEL_SLICE;
        for (uint i = tRow; i < kRows + TILE_SIZE - 1; i+= TILE_SIZE) {
            uint inRow = baseRow+ i;
            for (uint j = tCol; j < kCols + TILE_SIZE - 1; j += TILE_SIZE) {
                uint inCol = baseCol + j;
                patch[i * patchCols + j] = loadConv2dFwdPatch(
                    input, inRow, inRows, inCol, inCols, inDepth, baseChan, n
                    );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < kRows; i++) {
            for (uint j = 0; j < kCols; j++) {
                uint kIdx = ((o * kRows + i) * kCols + j) * inDepth + baseChan;
                packed_float4 patchVec = patch[(pRow + i) * patchCols + (pCol + j)];
                val += conv2dFwdDot(kIdx, patchVec, kernals, baseChan, inDepth);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inBounds) {
        uint outIdx = (((n * outRows + r) * outCols + c) * numKernals) + o;
        output[outIdx] = val;
    }
}

kernel void conv2dForwardMed(
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
    threadgroup packed_float4 patch[MED_PATCH_DIM][MED_PATCH_DIM];
    conv2dForward(
        input, kernals, biases, output, inputDims,
        kernalDims, outputDims, stride, gid, tid, 
        (threadgroup packed_float4*)patch, MED_PATCH_DIM
    );
}

kernel void conv2dForwardFast(
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
    threadgroup packed_float4 patch[SMALL_PATCH_DIM][SMALL_PATCH_DIM];
    conv2dForward(
        input, kernals, biases, output, inputDims,
        kernalDims, outputDims, stride, gid, tid, 
        (threadgroup packed_float4*)patch, SMALL_PATCH_DIM
    );
}