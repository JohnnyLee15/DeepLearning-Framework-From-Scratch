#include <metal_stdlib>
using namespace metal;

#define COARSE_FACTOR 4
#define THREADS_PER_GROUP 256

kernel void copy(
    device const float *in [[ buffer(0) ]],
    device float *out [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) 
        return;

    out[gid] = in[gid];
}

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

    uint d = gid.x;
    uint cStart = gid.y * COARSE_FACTOR;
    uint r = gid.z % inRows;
    uint n = gid.z / inRows;
    
    if (n >= numSamples || cStart >= inCols || d >= depth)
        return;

    for (uint c = 0; c < COARSE_FACTOR; c++) {
        uint col = cStart + c;
        if (col >= inCols) 
            return;

        uint inIdx = ((n * inRows + r) * inCols + col) * depth + d;
        uint padIdx = ((n * newRows + (r + padTop)) * newCols + (col + padLeft)) * depth + d;
        toPad[padIdx] = input[inIdx];
    }
}

kernel void conv2dForwardNaive(
    device const float *input [[ buffer(0) ]],
    device const float *kernals [[ buffer(1) ]],
    device const float *biases [[ buffer(2) ]],
    device float *output [[ buffer(3) ]],
    constant uint4 &inputDims [[ buffer(4) ]],
    constant uint4 &kernalDims [[ buffer(5) ]],
    constant uint4 &outputDims [[ buffer(6) ]],
    constant uint &stride [[ buffer(7) ]],
    uint3 gid [[ thread_position_in_grid ]]
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

    if (n >= numSamples || r >= outRows || c >= outCols)
        return;
    
    float val = biases[o];
    for (uint i = 0; i < kRows; i++) {
        uint inRow = r * stride + i;
        for (uint j = 0; j < kCols; j++) {
            uint inCol = c * stride + j;
            uint baseK = ((o * kRows + i) * kCols + j) * inDepth;
            uint baseI = ((n * inRows + inRow) * inCols + inCol) * inDepth;
            for (uint d = 0; d < inDepth; d++) {
                uint kIdx = baseK + d;
                uint inIdx = baseI + d;
                val += (input[inIdx] * kernals[kIdx]);
            }
        }
    }
    uint outIdx = ((n * outRows + r) * outCols + c) * numKernals + o;
    output[outIdx] = val;
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

kernel void conv2dWeightsNaive(
    device const float *input [[ buffer(0) ]],
    device const float *grad [[ buffer(1) ]],
    device float *dW [[ buffer(2) ]],
    constant uint4 &inDims [[ buffer(3) ]],
    constant uint2 &gradDims [[ buffer(4) ]],
    constant uint3 &kDims [[ buffer(5) ]],
    constant uint &stride [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
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
    uint i = gid.y;
    uint j = gid.x;
    uint d = gid.z % inDepth;

    if (k >= numKernals || i >= kRows || j >= kCols)
        return;

    float val = 0.0f;
    for (uint n = 0; n < numSamples; n++) {
        for (uint r = 0; r < gradRows; r++) {
            uint inRow = r*stride + i;
            for (uint c = 0; c < gradCols; c++) {
                uint inCol = c*stride + j;
                uint inIdx = ((n * inRows + inRow) * inCols + inCol) * inDepth + d;
                uint gradIdx = ((n * gradRows + r) * gradCols + c) * numKernals + k;
                val += (input[inIdx] * grad[gradIdx]);
            }
        }
    }

    uint dwIdx = ((k * kRows + i) * kCols + j) * inDepth + d;
    dW[dwIdx] = val;
}

kernel void reduceSumBias(
    device const float *grad [[ buffer(0) ]],
    device float *dB [[ buffer(1) ]],
    constant uint4 &gradDims [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
 ) {
    uint numSamples = gradDims[0];
    uint gradRows = gradDims[1];
    uint gradCols = gradDims[2];
    uint numKernals = gradDims[3];

    uint k = gid;

    if (k >= numKernals)
        return;

    float sum = 0.0f;
    for (uint n = 0; n < numSamples; n++) {
        for (uint r = 0; r < gradRows; r++) {
            for (uint c = 0; c < gradCols; c++) {
                uint gradIdx = ((n * gradRows + r) * gradCols + c) * numKernals + k;
                sum += grad[gradIdx];
            }
        }
    }

    dB[k] = sum;
}

kernel void applyBiasGrad(
    device const float *grad [[ buffer(0) ]],
    device float *biases [[ buffer(1) ]],
    constant uint4 &gradDims [[ buffer(2) ]],
    constant float &scaleFactor [[ buffer(3) ]],
    uint group_id [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]]
 ) {
    uint numSamples = gradDims[0];
    uint gradRows = gradDims[1];
    uint gradCols = gradDims[2];
    uint numKernals = gradDims[3];

    uint d = group_id;
    uint numElements = numSamples * gradRows * gradCols;
    threadgroup float tile[THREADS_PER_GROUP];
    float sum = 0.0f;

    for (uint i = tid; i < numElements; i += THREADS_PER_GROUP) {
        uint c = i % gradCols;
        uint r = (i / gradCols) % gradRows;
        uint n = i / (gradCols * gradRows);

        uint gradIdx = ((n * gradRows + r) * gradCols + c) * numKernals + d;
        sum += grad[gradIdx];
    }

    tile[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREADS_PER_GROUP/2; s > 0; s/=2) {
        if (tid < s) {
            tile[tid] += tile[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        biases[d] += scaleFactor * tile[tid];
    }
}

kernel void padAndUpsampleGrad(
    device const float *grad [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    constant uint4 &gradDims [[ buffer(2) ]],
    constant uint2 &outDims [[ buffer(3) ]],
    constant uint2 &padding [[ buffer(4) ]],
    constant uint &stride [[ buffer(5) ]], 
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint n = gid.z;
    uint r = gid.y;
    uint c = gid.x;

    uint numSamples = gradDims[0];
    uint gradRows = gradDims[1];
    uint gradCols = gradDims[2];
    uint numKernals = gradDims[3];

    if (n >= numSamples || r >= gradRows || c >= gradCols)
        return;

    uint outRows = outDims[0];
    uint outCols = outDims[1];

    uint padTop = padding[0];
    uint padLeft = padding[1];

    uint upRow = r * stride + padTop;
    uint upCol = c * stride + padLeft;

    uint baseIn = ((n * gradRows + r) * gradCols + c) * numKernals;
    uint baseUp = ((n * outRows + upRow) * outCols + upCol) * numKernals;

    for (uint d = 0; d < numKernals; d++) {
        uint inIdx = baseIn + d;
        uint upIdx = baseUp + d;
        output[upIdx] = grad[inIdx];
    }
}

kernel void conv2dInputNaive(
    device const float *grad [[ buffer(0) ]],
    device const float *kernals [[ buffer (1) ]],
    device float *dX [[ buffer(2) ]],
    constant uint4 &gradDims [[ buffer(3) ]],
    constant uint4 &kDims [[ buffer(4) ]],
    constant uint2 &dxDims [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint numSamples = gradDims[0];
    uint gradRows = gradDims[1];
    uint gradCols = gradDims[2];
    uint numKernals = gradDims[3];

    uint kRows = kDims[1];
    uint kCols = kDims[2];
    uint inDepth = kDims[3];

    uint dxRows = dxDims[0];
    uint dxCols = dxDims[1];

    uint n = gid.z / inDepth;
    uint r = gid.y;
    uint c = gid.x;
    uint d = gid.z % inDepth;

    if (n >= numSamples || r >= dxRows || c >= dxCols)
        return;

    float val = 0.0f;
    for (uint i = 0; i < kRows; i++) {
        uint gradRow = r + i;
        uint flipI = kRows - 1 - i;

        for (uint j = 0; j < kCols; j++) {
            uint gradCol = c + j;
            uint flipJ = kCols - 1 - j;

            for (uint k = 0; k < numKernals; k++) {
                uint kIdx = ((k * kRows + flipI) * kCols + flipJ) * inDepth + d;
                uint gradIdx = ((n * gradRows + gradRow) * gradCols + gradCol) * numKernals + k;
                val += kernals[kIdx] * grad[gradIdx];
            }
        }
    }

    uint dxIdx = ((n * dxRows + r) * dxCols + c) * inDepth + d;
    dX[dxIdx] = val;
}

kernel void maxPool2dGrad(
    device const float *grad [[ buffer(0) ]],
    device const uint *maxIndices [[ buffer(1) ]],
    device atomic_float *dX [[ buffer(2) ]],
    constant uint &gradSize [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= gradSize)
        return; 
        
    uint dxIdx = maxIndices[gid];
    atomic_fetch_add_explicit(&dX[dxIdx], grad[gid], memory_order_relaxed);
}