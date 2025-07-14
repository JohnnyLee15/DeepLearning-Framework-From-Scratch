#include <metal_stdlib>
using namespace metal;

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