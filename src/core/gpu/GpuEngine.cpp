#include "core/gpu/GpuEngine.h"

bool GpuEngine::usingGpu = false;

bool GpuEngine::isUsingGpu() {
    return usingGpu;
}

void GpuEngine::disableGpu() {
    usingGpu = false;
}

void GpuEngine::enableGpu() {
    usingGpu = true;
}