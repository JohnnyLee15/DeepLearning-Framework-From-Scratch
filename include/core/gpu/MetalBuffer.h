#pragma once

#ifdef __OBJC__
    #import <Metal/Metal.h> 
#endif

class MetalBuffer { 
    private:
        // Instance Variable
        #ifdef __OBJC__
            id<MTLBuffer> buffer;
        #endif

    public:
        // Constructors
        MetalBuffer();
        MetalBuffer(const void*, size_t);

        // Destructor
        ~MetalBuffer();
    
        // Methods
        MetalBuffer& operator=(const MetalBuffer&);

        void downloadToHost(void*, size_t) const;
        void uploadFromHost(const void*, size_t);

        #ifdef __OBJC__
            id<MTLBuffer> getBuffer();
            const id<MTLBuffer> getBuffer() const;
        #endif
};