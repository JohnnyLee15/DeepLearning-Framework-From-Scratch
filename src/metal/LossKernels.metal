#include <metal_stdlib>
using namespace metal;

kernel void calculateMSEGrad(
    device const float *targets [[ buffer(0) ]],
    device const float *a [[ buffer(1) ]],
    device float *dL [[ buffer(2) ]],
    constant uint &size [[ buffer(3) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    dL[gid] = 2.0 * (a[gid] - targets[gid]);
}

kernel void calculateSoftmaxCrossEntropyGrad(
    device const float *labels [[ buffer(0) ]],
    device const float *a [[ buffer(1) ]],
    device float *dL [[ buffer(2) ]],
    constant uint &numRows [[ buffer(3) ]],
    constant uint &numCols [[ buffer(4) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numRows * numCols)
        return;

    uint i = gid / numCols;
    uint j = gid % numCols;
    uint labelIdx = (uint) labels[i];

    dL[gid] = (j == labelIdx) ? a[gid] - 1 : a[gid];
}