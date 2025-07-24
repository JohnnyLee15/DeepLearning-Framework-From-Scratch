#include <metal_stdlib>
using namespace metal;

kernel void activateReLU(
    device const float *z [[ buffer(0) ]],
    device float *a [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    a[gid] = max(0.0, z[gid]);
}

kernel void activateSoftmax(
    device const float *z [[ buffer(0) ]],
    device float *a [[ buffer(1) ]],
    constant uint2 &dims [[ buffer(2) ]],
    uint row [[thread_position_in_grid]]
) {
    uint numRows = dims.x;
    uint numCols = dims.y;

    if (row >= numRows)
        return;

    float maxVal = -INFINITY;
    for (uint j = 0; j < numCols; j++) {
        maxVal = max(z[row * numCols + j], maxVal);
    }

    float totalSum = 0.0;
    for (uint j = 0; j < numCols; j++) {
        float expVal = exp(z[row * numCols + j] - maxVal);
        a[row * numCols + j] = expVal;
        totalSum += expVal;
    }

    for (uint j = 0; j < numCols; j++) {
        a[row * numCols + j] /= totalSum;
    }
}

kernel void calculateLinearGrad(
    device float *dZ [[ buffer(0) ]],
    constant uint &size [[ buffer(1) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;
    
    dZ[gid] = 1.0;
}

kernel void calculateReluGrad(
    device const float *z [[ buffer(0) ]],
    device float *dZ [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    if (z[gid] > 0) {
        dZ[gid] = 1.0;
    } else {
        dZ[gid] = 0.0;
    }
}