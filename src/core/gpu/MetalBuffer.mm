#include "core/gpu/MetalBuffer.h"
#include "core/gpu/GpuEngine.h"

MetalBuffer::MetalBuffer() : buffer(nil) {}

MetalBuffer::MetalBuffer(const void *host, size_t numBytes) {
    buffer = [GpuEngine::getGpuDevice()
        newBufferWithBytes:host
        length:numBytes
        options:MTLResourceStorageModeShared];
}

MetalBuffer& MetalBuffer::operator=(const MetalBuffer &other) {
    if (this != &other) {
        [buffer release];
        buffer = [other.buffer retain];
    }
    return *this;
}

void MetalBuffer::downloadToHost(void *host, size_t numBytes) const {
    memcpy(host, [buffer contents], numBytes);
}

void MetalBuffer::uploadFromHost(const void *host, size_t numBytes) {
    memcpy([buffer contents], host, numBytes);
}

MetalBuffer::~MetalBuffer() {
    [buffer release];
}

id<MTLBuffer> MetalBuffer::getBuffer() {
    return buffer;
}

const id<MTLBuffer> MetalBuffer::getBuffer() const {
    return buffer;
}