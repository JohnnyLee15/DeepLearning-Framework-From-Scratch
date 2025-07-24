#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8
#define CHANNEL_SLICE 4
#define MAX_KERNEL 7
#define MED_PATCH_DIM ((TILE_SIZE - 1) * 2 + MAX_KERNEL)
#define SMALL_PATCH_DIM ((TILE_SIZE - 1) + MAX_KERNEL)

static inline packed_float4 loadConv2dWtPatch(
    device const float *input,
    uint inRow,
    uint inRows,
    uint inCol,
    uint inCols,
    uint inDepth,
    uint n,
    uint baseChan
) {
    if (inRow >= inRows || inCol >= inCols || baseChan >= inDepth) {
        return packed_float4(0.0f);
    } 
    
    uint inIdx = ((n * inRows + inRow) * inCols + inCol) * inDepth + baseChan;
    if (baseChan + 3 < inDepth) {
        return *((device packed_float4*)&input[inIdx]);
    } 
    
    float v0 = (baseChan < inDepth) ? input[inIdx] : 0.0f;
    float v1 = (baseChan + 1 < inDepth) ? input[inIdx + 1] : 0.0f;
    float v2 = (baseChan + 2 < inDepth ) ? input[inIdx + 2] : 0.0f;
    float v3 = (baseChan  + 3 < inDepth) ? input[inIdx + 3] : 0.0f;

    return packed_float4(v0, v1, v2, v3);
}

static inline void conv2dWeights(
    device const float *input [[ buffer(0) ]],
    device const float *grad [[ buffer(1) ]],
    device float *dW [[ buffer(2) ]],
    constant uint4 &inDims [[ buffer(3) ]],
    constant uint2 &gradDims [[ buffer(4) ]],
    constant uint3 &kDims [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    threadgroup packed_float4 *patch,
    uint patchCols,
    threadgroup float *gradPatch,
    uint3 gid,
    uint3 tid
) {
    uint numSamples = inDims[0];
    uint inRows = inDims[1];
    uint inCols = inDims[2];
    uint inDepth = inDims[3];
    
    uint gradRows = gradDims[0];
    uint gradCols = gradDims[1];

    uint numKernals = kDims[0];
    uint kRows = kDims[1];
    uint kCols = kDims[2];

    uint k = gid.z / inDepth;
    uint kRow = gid.y;
    uint kCol = gid.x;
    uint d = gid.z % inDepth;

    uint tRow = tid.y;
    uint tCol = tid.x;

    uint baseRow = kRow * stride;
    uint baseCol = kCol * stride;

    uint numSlices = (inDepth + CHANNEL_SLICE - 1)/CHANNEL_SLICE;

    float val = 0.0f;
    for (uint n = 0; n < numSamples; n++) {
        for (uint r = 0; r < gradRows; r+=TILE_SIZE) {
            uint gradRow = r + tRow;
            for (uint c = 0; c < gradCols; c+=TILE_SIZE) {
                uint gradCol = c + tCol;
                if (gradRow < gradRows && gradCol < gradCols) {
                    uint gradIdx = ((n * gradRows + gradRow) * gradCols + gradCol) * numKernals + k;
                    gradPatch[tRow * TILE_SIZE + tCol] = grad[gradIdx];
                } else {
                    gradPatch[tRow * TILE_SIZE + tCol] = 0.0f;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint p = 0; p < numSlices; p++) {
                    uint baseChan = p * CHANNEL_SLICE;

                    for (uint i = tRow; i < TILE_SIZE + kRows - 1; i += TILE_SIZE) {
                        uint inRow = baseRow + (r + i)*stride;
                        for (uint j = tCol; j < TILE_SIZE + kCols - 1; j += TILE_SIZE) {
                            uint inCol = baseCol + (c + j)*stride;
                            patch[i * patchCols + j] = loadConv2dWtPatch(
                                input, inRow, inRows, inCol, inCols, inDepth, n, baseChan
                                );
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    uint patchChan = d - baseChan;
                    if (patchChan < CHANNEL_SLICE) {
                        for (uint gr = 0; gr < TILE_SIZE; gr++) {
                            for (uint gc = 0; gc < TILE_SIZE; gc++) {
                                float gVal = gradPatch[gr * TILE_SIZE + gc];
                                float inVal = patch[(gr + kRow) * patchCols + (gc + kCol)][patchChan];
                                val += gVal * inVal;
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (k < numKernals && kRow < kRows && kCol < kCols) {
        uint dwIdx = ((k * kRows + kRow) * kCols + kCol) * inDepth + d;
        dW[dwIdx] = val;
    }
}

kernel void conv2dWeightsFast(
    device const float *input [[ buffer(0) ]],
    device const float *grad [[ buffer(1) ]],
    device float *dW [[ buffer(2) ]],
    constant uint4 &inDims [[ buffer(3) ]],
    constant uint2 &gradDims [[ buffer(4) ]],
    constant uint3 &kDims [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]]
) {
    threadgroup packed_float4 patch[SMALL_PATCH_DIM][SMALL_PATCH_DIM];
    threadgroup float gradPatch[TILE_SIZE][TILE_SIZE];
    conv2dWeights(
        input, grad, dW, inDims, gradDims, kDims, stride,
        (threadgroup packed_float4*)patch, SMALL_PATCH_DIM,
        (threadgroup float*)gradPatch, gid, tid
    );
}

kernel void conv2dWeightsMed(
    device const float *input [[ buffer(0) ]],
    device const float *grad [[ buffer(1) ]],
    device float *dW [[ buffer(2) ]],
    constant uint4 &inDims [[ buffer(3) ]],
    constant uint2 &gradDims [[ buffer(4) ]],
    constant uint3 &kDims [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]]
) {
    threadgroup packed_float4 patch[MED_PATCH_DIM][MED_PATCH_DIM];
    threadgroup float gradPatch[TILE_SIZE][TILE_SIZE];
    conv2dWeights(
        input, grad, dW, inDims, gradDims, kDims, stride,
        (threadgroup packed_float4*)patch, MED_PATCH_DIM,
        (threadgroup float*)gradPatch, gid, tid
    );
}