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
            static id<MTLComputePipelineState> mmBiasReLUPipeline;
            static id<MTLComputePipelineState> applyKernelGradsPipeline;
            static id<MTLComputePipelineState> matMatTPipeline;
            static id<MTLComputePipelineState> matTMatPipeline;
            static id<MTLComputePipelineState> matTMatTPipeline;

            static id<MTLComputePipelineState> colSumsPipeline;
            static id<MTLComputePipelineState> addToRowsPipeline;

            static id<MTLComputePipelineState> hadamardPipeline;
            static id<MTLComputePipelineState> applyGradPipeline;

            static id<MTLComputePipelineState> calculateLinearGradPipeline;
            static id<MTLComputePipelineState> calculateReluGradPipeline;
            static id<MTLComputePipelineState> backpropReLUPipeline;

            static id<MTLComputePipelineState> activateReLUPipeline;
            static id<MTLComputePipelineState> activateSoftmaxPipeline;

            static id<MTLComputePipelineState> calculateMSEGradPipeline;
            static id<MTLComputePipelineState> calculateSoftmaxCrossEntropyGradPipeline;

            static id<MTLComputePipelineState> copyTensorPipeline;
            static id<MTLComputePipelineState> padWindowInputPipeline;
            static id<MTLComputePipelineState> fillFloatPipeline;
            static id<MTLComputePipelineState> fillIntPipeline;
            static id<MTLComputePipelineState> maxPool2dPipeline;
            static id<MTLComputePipelineState> reduceSumBiasPipeline;
            static id<MTLComputePipelineState> applyBiasGradPipeline;

            static id<MTLComputePipelineState> conv2dForwardNaivePipeline;
            static id<MTLComputePipelineState> conv2dWeightsNaivePipeline;
            static id<MTLComputePipelineState> conv2dInputNaivePipeline;

            static id<MTLComputePipelineState> im2ColPipeline;
            static id<MTLComputePipelineState> addBiasApplyReLUIm2ColPipeline;
            static id<MTLComputePipelineState> col2ImPipeline;

            static id<MTLComputePipelineState> padAndUpsampleGradPipeline;

            static id<MTLComputePipelineState> maxPool2dGradPipeline;

            // Static Methods
            static void initLib();
            static void initPipe(const char*, id<MTLComputePipelineState>&);
            static void initAllPipes();
        #endif

    public:
        // Static Methods
        static bool isUsingGpu();
        static void disableGpu();
        static void enableGpu();
        static void init();

        #ifdef __APPLE__
            static void initInternal();
        #endif


        #ifdef __OBJC__
            static id<MTLDevice> getGpuDevice();
            static id<MTLCommandQueue> getCmdQueue();
            static id<MTLLibrary> getLib();

            static void fillFloat(id<MTLBuffer>, uint32_t, id<MTLCommandBuffer>, float);
            static void fillInt(id<MTLBuffer>, uint32_t, id<MTLCommandBuffer>, uint32_t);

            static id<MTLComputePipelineState> getMatMatPipe();
            static id<MTLComputePipelineState> getMMBiasReLUPipe();
            static id<MTLComputePipelineState> getApplyKernelGradsPipe();
            static id<MTLComputePipelineState> getMatMatTPipe();
            static id<MTLComputePipelineState> getMatTMatPipe();
            static id<MTLComputePipelineState> getMatTMatTPipe();

            static id<MTLComputePipelineState> getColSumsPipe();
            static id<MTLComputePipelineState> getAddToRowsPipe();

            static id<MTLComputePipelineState> getHadamardPipe();
            static id<MTLComputePipelineState> getApplyGradPipe();

            static id<MTLComputePipelineState> getCalculateLinearGradPipe();
            static id<MTLComputePipelineState> getCalculateReluGradPipe();
            static id<MTLComputePipelineState> getBackpropReLUPipe();

            static id<MTLComputePipelineState> getActivateReLUPipe();
            static id<MTLComputePipelineState> getActivateSoftmaxPipe();

            static id<MTLComputePipelineState> getCalculateMSEGradPipe();
            static id<MTLComputePipelineState> getCalculateSoftmaxCrossEntropyGradPipe();

            static id<MTLComputePipelineState> getCopyTensorPipe();
            static id<MTLComputePipelineState> getPadWindowInputPipe();
            static id<MTLComputePipelineState> getMaxPool2dPipe();
            static id<MTLComputePipelineState> getReduceSumBiasPipe();
            static id<MTLComputePipelineState> getApplyBiasGradPipe();

            static id<MTLComputePipelineState> getConv2dForwardNaivePipe();
            static id<MTLComputePipelineState> getConv2dWeightsNaivePipe();
            static id<MTLComputePipelineState> getConv2dInputNaivePipe();

            static id<MTLComputePipelineState> getIm2ColPipe();
            static id<MTLComputePipelineState> getCol2ImPipe();
            static id<MTLComputePipelineState> getAddBiasApplyReLUIm2ColPipe();

            static id<MTLComputePipelineState> getMaxPool2dGradPipe();

            static id<MTLComputePipelineState> getPadAndUpsampleGradPipe();
        #endif
};

