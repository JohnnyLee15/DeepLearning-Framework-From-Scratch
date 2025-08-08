#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8
#define CHANNEL_SLICE 4
#define MAX_KERNEL 7
#define MAX_STRIDE 2
#define PATCH_DIM ((TILE_SIZE - 1) * MAX_STRIDE + MAX_KERNEL)
#define COARSE_FACTOR 4

static inline float4 loadIm2ColPatch(
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
        return float4(0.0f);
    }

    uint inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + baseChan;
    if (baseChan + 3 < inDepth) {
        return *((device float4*)&input[inIdx]);
    } 

    float v0 = (baseChan < inDepth) ? input[inIdx] : 0.0f;
    float v1 = (baseChan + 1 < inDepth) ? input[inIdx + 1] : 0.0f;
    float v2 = (baseChan + 2 < inDepth) ? input[inIdx + 2] : 0.0f;
    float v3 = (baseChan + 3 < inDepth) ? input[inIdx + 3] : 0.0f;
    return float4(v0, v1, v2, v3);
}


kernel void im2Col(
    device const float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    constant uint4 &inDims [[ buffer(2) ]],
    constant uint2 &kDims [[ buffer(3) ]],
    constant uint2 &outDims [[ buffer(4) ]],
    constant uint &flatCols [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]]
) {
    uint n = gid.z;
    uint r = gid.y;
    uint c = gid.x;

    uint numSamples = inDims[0];
    uint inRows = inDims[1];
    uint inCols = inDims[2];
    uint inDepth = inDims[3];

    uint kRows = kDims[0];
    uint kCols = kDims[1];
    
    uint outRows = outDims[0];
    uint outCols = outDims[1];

    uint tRow = tid.y;
    uint tCol = tid.x;
    uint pRow = tRow * stride;
    uint pCol = tCol * stride;

    uint baseRow = (r - tRow) * stride;
    uint baseCol = (c - tCol) * stride;  

    uint flatRow = (n * outRows + r) * outCols + c;
    uint numSlices = (inDepth + CHANNEL_SLICE - 1)/CHANNEL_SLICE;

    bool inBounds = (n < numSamples) && (r < outRows) && (c < outCols);

    threadgroup float4 patch[PATCH_DIM][PATCH_DIM];

    for (uint p = 0; p < numSlices; p++) {
        uint baseChan = p*CHANNEL_SLICE;
        
        for (uint i = tRow; i < PATCH_DIM; i+= TILE_SIZE) {
            uint inRow = baseRow+ i;
            for (uint j = tCol; j < PATCH_DIM; j += TILE_SIZE) {
                uint inCol = baseCol + j;
                patch[i][j] = loadIm2ColPatch(
                    input, inRow, inRows, inCol, inCols, inDepth, baseChan, n
                    );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (inBounds) {
            for (uint i = 0; i < kRows; i++) {
                for (uint j = 0; j < kCols; j++) {
                    float4 vec = patch[pRow + i][pCol + j];
                    uint flatCol = (i * kCols + j) * inDepth + baseChan;
                    uint baseOutIdx = flatRow * flatCols + flatCol;

                    if (baseChan < inDepth) output[baseOutIdx] = vec[0];
                    if (baseChan + 1 < inDepth) output[baseOutIdx + 1] = vec[1];
                    if (baseChan + 2 < inDepth) output[baseOutIdx + 2] = vec[2];
                    if (baseChan + 3 < inDepth) output[baseOutIdx + 3] = vec[3];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void addBiasApplyReLUIm2Col(
    device float *z [[ buffer (0) ]],
    device float *a [[ buffer (1) ]],
    constant float *biases [[ buffer(2) ]],
    constant uint &numKernels [[ buffer(3) ]],
    constant uint &size [[ buffer(4) ]],
    constant uint &gridWidth [[ buffer(5 )]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint idx = gid;

    if (idx >= size) return;
    float v0 = z[idx] + biases[idx % numKernels];
    z[idx] = v0;
    a[idx] = max(v0, 0.0f);

    idx += gridWidth; 
    if (idx >= size) return;
    float v1 = z[idx] + biases[(idx) % numKernels];
    z[idx] = v1;
    a[idx] = max(v1, 0.0f);


    idx += gridWidth; 
    if (idx >= size) return;
    float v2 = z[idx] + biases[(idx) % numKernels];
    z[idx] = v2;
    a[idx] = max(v2, 0.0f);

 
    idx += gridWidth; 
    if (idx >= size) return;
    float v3 = z[idx] + biases[(idx) % numKernels];
    z[idx] = v3;
    a[idx] = max(v3, 0.0f);
}

kernel void col2ImSlow(
    device const float *grad [[ buffer(0) ]],
    device float *dX [[ buffer(1) ]],
    constant uint2 &gradDims [[ buffer(2) ]],
    constant uint2 &kDims [[ buffer(3) ]],
    constant uint4 &dxDims [[ buffer(4) ]],
    constant uint &stride [[ buffer(5) ]],
    constant uint2 &padding [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint gradRows = gradDims[0];
    uint gradCols = gradDims[1];

    uint numSamples = dxDims[0];
    uint dxRows = dxDims[1];
    uint dxCols = dxDims[2];
    uint inDepth = dxDims[3];

    uint kRows = kDims[0];
    uint kCols = kDims[1];

    uint padTop = padding[0];
    uint padLeft = padding[1];

    uint dxCol = gid.x;
    uint dxRow = gid.y;
    uint n = gid.z / inDepth;
    uint d = gid.z % inDepth;

    int rowBase = (int) dxRow + (int) padTop;
    int colBase = (int) dxCol + (int) padLeft;
    uint gradStride = kRows * kCols * inDepth;
    
    if (dxRow >= dxRows || dxCol >= dxCols || n >= numSamples)
        return;

    float val = 0.0f;

    for (uint i = 0; i < kRows; i++) {
        int tempRow = rowBase - (int) i;
        if (tempRow < 0 || (tempRow % (int) stride != 0)) continue;
        uint gradRow = tempRow / stride;
        if (gradRow >= gradRows) continue;

        for (uint j = 0; j < kCols; j++) {
            int tempCol = colBase - (int) j;
            if (tempCol < 0 || (tempCol % (int) stride != 0)) continue;
            uint gradCol = tempCol / stride;
            if (gradCol >= gradCols) continue;
            
            uint row = (n * gradRows + gradRow) * gradCols + gradCol;
            uint col = (i * kCols + j) * inDepth + d;
            val += grad[row * gradStride + col];
        }
    }

    uint idx = ((n * dxRows + dxRow) * dxCols + dxCol) * inDepth + d;
    dX[idx] = val;
}

kernel void col2ImFast(
    device const float *grad [[ buffer(0) ]],
    device float *dX [[ buffer(1) ]],
    constant uint2 &gradDims [[ buffer(2) ]],
    constant uint2 &kDims [[ buffer(3) ]],
    constant uint4 &dxDims [[ buffer(4) ]],
    constant uint &stride [[ buffer(5) ]],
    constant uint2 &padding [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint gradRows = gradDims[0];
    uint gradCols = gradDims[1];

    uint numSamples = dxDims[0];
    uint dxRows = dxDims[1];
    uint dxCols = dxDims[2];
    uint inDepth = dxDims[3];

    uint kRows = kDims[0];
    uint kCols = kDims[1];

    uint padTop = padding[0];
    uint padLeft = padding[1];

    uint dxCol = gid.x;
    uint dxRow = gid.y;
    uint n = gid.z / (inDepth / 4);
    uint dPack = gid.z % (inDepth / 4);
    uint d = dPack * 4;

    int rowBase = (int) dxRow + (int) padTop;
    int colBase = (int) dxCol + (int) padLeft;
    uint gradStride = kRows * kCols * inDepth;
    
    if (dxRow >= dxRows || dxCol >= dxCols || n >= numSamples)
        return;

    float4 val = float4(0.0f);

    for (uint i = 0; i < kRows; i++) {
        int tempRow = rowBase - (int) i;
        if (tempRow < 0 || (tempRow % (int) stride != 0)) continue;
        uint gradRow = tempRow / stride;
        if (gradRow >= gradRows) continue;

        for (uint j = 0; j < kCols; j++) {
            int tempCol = colBase - (int) j;
            if (tempCol < 0 || (tempCol % (int) stride != 0)) continue;
            uint gradCol = tempCol / stride;
            if (gradCol >= gradCols) continue;
            
            uint row = (n * gradRows + gradRow) * gradCols + gradCol;
            uint col = (i * kCols + j) * inDepth + d;
            val += *(device const float4*)&grad[row * gradStride + col];
        }
    }

    uint idx = ((n * dxRows + dxRow) * dxCols + dxCol) * inDepth + d;
    *(device float4*)&dX[idx] = val;
}