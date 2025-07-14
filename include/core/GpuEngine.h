#pragma once

#ifdef __OBJC__
    #import <Metal/Metal.h> 
#endif

class GpuEngine {
    private:
        #ifdef __OBJC__
            // Static Variables:
            static id<MTLDevice> gpuDevice;
            static id<MTLCommandQueue> cmdQueue;
            static id<MTLLibrary> defaultLib;

            static id<MTLComputePipelineState> matMatPipeline;
            static id<MTLComputePipelineState> matMatTPipeline;
            static id<MTLComputePipelineState> matTMatPipeline;
            static id<MTLComputePipelineState> matTMatTPipeline;

            static id<MTLComputePipelineState> colSumsPipeline;
            static id<MTLComputePipelineState> addToRowsPipeline;

            static id<MTLComputePipelineState> hadamardPipeline;
            static id<MTLComputePipelineState> scalePipeline;
            static id<MTLComputePipelineState> addPipeline;

            // Static Methods
            static void initLib();
            static void initPipe(const char*, id<MTLComputePipelineState>&);
            static void initAllPipes();
        #endif

    public:
        // Static Variables
        static bool usingGpu;

        #ifdef __OBJC__
            // Static Methods
            static void init();
            static id<MTLDevice> getGpuDevice();
            static id<MTLCommandQueue> getCmdQueue();
            static id<MTLLibrary> getLib();

            static id<MTLComputePipelineState> getMatMatPipe();
            static id<MTLComputePipelineState> getMatMatTPipe();
            static id<MTLComputePipelineState> getMatTMatPipe();
            static id<MTLComputePipelineState> getMatTMatTPipe();

            static id<MTLComputePipelineState> getColSumsPipe();
            static id<MTLComputePipelineState> getAddToRowsPipe();

            static id<MTLComputePipelineState> getHadamardPipe();
            static id<MTLComputePipelineState> getScalePipe();
            static id<MTLComputePipelineState> getAddPipe();
        #endif
};

