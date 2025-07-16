#pragma once

#include "core/GpuEngine.h"

class MetalBuffer { 
    private:
        #ifdef __OBJC__
            id<MTLBuffer> buf;
        #endif

    public:
        MetalBuffer();
        MetalBuffer(const MetalBuffer&);
        MetalBuffer& operator=(const MetalBuffer&);
        ~MetalBuffer();

        #ifdef __OBJC__
            MetalBuffer(id<MTLBuffer>);
            id<MTLBuffer> getBuf();
            const id<MTLBuffer> getBuf() const;
        #endif
};