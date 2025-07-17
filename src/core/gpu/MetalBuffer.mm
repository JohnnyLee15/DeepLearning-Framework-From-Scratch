#include "core/gpu/MetalBuffer.h"

MetalBuffer::MetalBuffer() : buf(nil) {}

MetalBuffer::MetalBuffer(const MetalBuffer &other) :
    buf([other.buf retain]) {}

MetalBuffer::MetalBuffer(id<MTLBuffer> metalBuf) : buf([metalBuf retain]) {}

MetalBuffer& MetalBuffer::operator=(const MetalBuffer &other) {
    if (this != &other) {
        [buf release];
        buf = [other.buf retain];
    }
    return *this;
}


MetalBuffer::~MetalBuffer() {
    [buf release];
}

id<MTLBuffer> MetalBuffer::getBuf() {
    return buf;
}

const id<MTLBuffer> MetalBuffer::getBuf() const {
    return buf;
}