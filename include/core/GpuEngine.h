#pragma once

#ifdef __OBJC__
    #import <Metal/Metal.h> 
#endif

class GpuEngine {
    private:
        // Static Variables
        static bool usingGpu;

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
            static id<MTLComputePipelineState> applyGradPipeline;

            static id<MTLComputePipelineState> calculateLinearGradPipeline;
            static id<MTLComputePipelineState> calculateReluGradPipeline;

            static id<MTLComputePipelineState> activateLinearPipeline;
            static id<MTLComputePipelineState> activateReLUPipeline;
            static id<MTLComputePipelineState> activateSoftmaxPipeline;

            static id<MTLComputePipelineState> calculateMSEGradPipeline;
            static id<MTLComputePipelineState> calculateSoftmaxCrossEntropyGradPipeline;

            // Static Methods
            static void initLib();
            static void initPipe(const char*, id<MTLComputePipelineState>&);
            static void initAllPipes();
        #endif

    public:

        static bool isUsingGpu();
        static void init();

        #ifdef __OBJC__
            // Static Methods
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
            static id<MTLComputePipelineState> getApplyGradPipe();

            static id<MTLComputePipelineState> getCalculateLinearGradPipe();
            static id<MTLComputePipelineState> getCalculateReluGradPipe();

            static id<MTLComputePipelineState> getActivateLinearPipe();
            static id<MTLComputePipelineState> getActivateReLUPipe();
            static id<MTLComputePipelineState> getActivateSoftmaxPipe();

            static id<MTLComputePipelineState> getCalculateMSEGradPipe();
            static id<MTLComputePipelineState> getCalculateSoftmaxCrossEntropyGradPipe();
        #endif
};

