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
    if (inRow >= inRows || inCol >= inCols || baseChan >= inDepth) {
        return packed_float4(0.0f);
    }

    uint inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + baseChan;
    if (baseChan + 3 < inDepth) {
        return *((device packed_float4*)&input[inIdx]);
    } 

    float v0 = (baseChan < inDepth) ? input[inIdx] : 0.0f;
    float v1 = (baseChan + 1 < inDepth) ? input[inIdx + 1] : 0.0f;
    float v2 = (baseChan + 2 < inDepth) ? input[inIdx + 2] : 0.0f;
    float v3 = (baseChan + 3 < inDepth) ? input[inIdx + 3] : 0.0f;
    return packed_float4(v0, v1, v2, v3);
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
    uint patchDim,
    threadgroup packed_float4 *kPatch
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

        if (tRow < kRows && tCol < kCols){
            kPatch[tRow * kCols + tCol] = loadConv2dFwdPatch(
                kernals, tRow, kRows, tCol, kCols, inDepth, baseChan, o
            );
        }
        
        for (uint i = tRow; i < patchDim; i+= TILE_SIZE) {
            uint inRow = baseRow+ i;
            for (uint j = tCol; j < patchDim; j += TILE_SIZE) {
                uint inCol = baseCol + j;
                patch[i * patchDim + j] = loadConv2dFwdPatch(
                    input, inRow, inRows, inCol, inCols, inDepth, baseChan, n
                    );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < kRows; i++) {
            for (uint j = 0; j < kCols; j++) {
                packed_float4 kPatchVec = kPatch[i * kCols + j];
                packed_float4 patchVec = patch[(pRow + i) * patchDim + (pCol + j)];
                val += dot(kPatchVec, patchVec);
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
    threadgroup packed_float4 kPatch[MAX_KERNEL][MAX_KERNEL];
    conv2dForward(
        input, kernals, biases, output, inputDims,
        kernalDims, outputDims, stride, gid, tid, 
        (threadgroup packed_float4*)patch, MED_PATCH_DIM,
        (threadgroup packed_float4*)kPatch
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
    threadgroup packed_float4 kPatch[MAX_KERNEL][MAX_KERNEL];
    conv2dForward(
        input, kernals, biases, output, inputDims,
        kernalDims, outputDims, stride, gid, tid, 
        (threadgroup packed_float4*)patch, SMALL_PATCH_DIM,
        (threadgroup packed_float4*)kPatch
    );
}