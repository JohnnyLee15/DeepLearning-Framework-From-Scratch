#include <metal_stdlib>
using namespace metal;

kernel void fillFloat(
    device float *ten [[ buffer(0) ]],
    constant uint &size [[ buffer(1) ]],
    constant float &val [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    ten[gid] = val;
}

kernel void fillInt(
    device uint *ten [[ buffer(0) ]],
    constant uint &size [[ buffer(1) ]],
    constant uint &val [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    ten[gid] = val;
}