#include <metal_stdlib>
using namespace metal;

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

kernel void scale(
    device float *ten [[ buffer(0) ]],
    constant float &scaleFactor [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    ten[gid] *= scaleFactor;
}

kernel void add(
    device float *ten1 [[ buffer(0) ]],
    device const float *ten2 [[ buffer(1) ]],
    constant uint &size [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) 
        return;

    ten1[gid] += ten2[gid];
}
