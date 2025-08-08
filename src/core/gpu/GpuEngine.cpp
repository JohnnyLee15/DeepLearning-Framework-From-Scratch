#include "core/gpu/GpuEngine.h"

bool GpuEngine::usingGpu = false;

bool GpuEngine::isUsingGpu() {
    return usingGpu;
}

void GpuEngine::disableGpu() {
    #ifdef __APPLE__
        usingGpu = false;
    #endif
}

void GpuEngine::enableGpu() {
    #ifdef __APPLE__
        usingGpu = true;
    #endif
}

void GpuEngine::init() {
    #ifdef __APPLE__
        initInternal();
        enableGpu();
    #endif
}