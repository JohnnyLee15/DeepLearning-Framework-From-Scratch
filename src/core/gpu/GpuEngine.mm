#include "utils/ConsoleUtils.h"
#include "core/gpu/GpuEngine.h"
 #include <iostream>

id<MTLDevice> GpuEngine::gpuDevice = nil;
id<MTLCommandQueue> GpuEngine::cmdQueue = nil;
id<MTLLibrary> GpuEngine::defaultLib = nil;

id<MTLComputePipelineState> GpuEngine::matMatPipeline = nil;
id<MTLComputePipelineState> GpuEngine::matMatTPipeline = nil;
id<MTLComputePipelineState> GpuEngine::matTMatPipeline = nil;
id<MTLComputePipelineState> GpuEngine::matTMatTPipeline = nil;

id<MTLComputePipelineState> GpuEngine::colSumsPipeline = nil;
id<MTLComputePipelineState> GpuEngine::addToRowsPipeline = nil;

id<MTLComputePipelineState> GpuEngine::hadamardPipeline = nil;
id<MTLComputePipelineState> GpuEngine::applyGradPipeline = nil;

id<MTLComputePipelineState> GpuEngine::calculateLinearGradPipeline = nil;
id<MTLComputePipelineState> GpuEngine::calculateReluGradPipeline  = nil;

id<MTLComputePipelineState> GpuEngine::activateLinearPipeline = nil;
id<MTLComputePipelineState> GpuEngine::activateReLUPipeline = nil;
id<MTLComputePipelineState> GpuEngine::activateSoftmaxPipeline = nil;

id<MTLComputePipelineState> GpuEngine::calculateMSEGradPipeline = nil;
id<MTLComputePipelineState> GpuEngine::calculateSoftmaxCrossEntropyGradPipeline  = nil;

void GpuEngine::init() {
    gpuDevice = MTLCreateSystemDefaultDevice();
    cmdQueue = [gpuDevice newCommandQueue];
    initLib();
    initAllPipes();
    usingGpu = true;
}

void GpuEngine::initLib() {
    NSError *error = nil;
    NSURL *libURL = [NSURL fileURLWithPath:@"./CoreKernels.metallib"];
    defaultLib = [gpuDevice newLibraryWithURL:libURL error:&error];

    if (error) {
        ConsoleUtils::fatalError(
            "Metal error: " + std::string([[error localizedDescription] UTF8String])
        );
    }
}

void GpuEngine::initPipe(const char *funcName, id<MTLComputePipelineState> &pipeline) {
    id<MTLFunction> func = [defaultLib newFunctionWithName:[NSString stringWithUTF8String:funcName]];

    if (!func) {
        ConsoleUtils::fatalError(
            "Failed to find '" + std::string(funcName) + "' function in Metal library."
        );
    }

    NSError *error = nil;
    pipeline = [gpuDevice newComputePipelineStateWithFunction:func error:&error];
    if (error) {
        ConsoleUtils::fatalError(
            "Metal pipeline error: " + std::string([[error localizedDescription] UTF8String])
        );
    }
}

void GpuEngine::initAllPipes() {
    initPipe("mm", matMatPipeline);
    initPipe("mmT", matMatTPipeline);
    initPipe("mTm", matTMatPipeline);
    initPipe("mTmT", matTMatTPipeline);

    initPipe("colSums", colSumsPipeline);
    initPipe("addToRows", addToRowsPipeline);

    initPipe("hadamard", hadamardPipeline);
    initPipe("applyGrad", applyGradPipeline);

    initPipe("calculateLinearGrad", calculateLinearGradPipeline);
    initPipe("calculateReluGrad", calculateReluGradPipeline);

    initPipe("activateLinear", activateLinearPipeline);
    initPipe("activateReLU", activateReLUPipeline);
    initPipe("activateSoftmax", activateSoftmaxPipeline);

    initPipe("calculateMSEGrad", calculateMSEGradPipeline);
    initPipe("calculateSoftmaxCrossEntropyGrad", calculateSoftmaxCrossEntropyGradPipeline);
}

id<MTLDevice> GpuEngine::getGpuDevice() { return gpuDevice; }
id<MTLCommandQueue> GpuEngine::getCmdQueue() { return cmdQueue; }
id<MTLLibrary> GpuEngine::getLib() { return defaultLib; }


id<MTLComputePipelineState> GpuEngine::getMatMatPipe() { return matMatPipeline; }
id<MTLComputePipelineState> GpuEngine::getMatMatTPipe() { return matMatTPipeline; }
id<MTLComputePipelineState> GpuEngine::getMatTMatPipe() { return matTMatPipeline; }
id<MTLComputePipelineState> GpuEngine::getMatTMatTPipe() { return matTMatTPipeline; }

id<MTLComputePipelineState> GpuEngine::getColSumsPipe() { return colSumsPipeline; }
id<MTLComputePipelineState> GpuEngine::getAddToRowsPipe() { return addToRowsPipeline; }

id<MTLComputePipelineState> GpuEngine::getHadamardPipe() { return hadamardPipeline; }
id<MTLComputePipelineState> GpuEngine::getApplyGradPipe() { return applyGradPipeline; }

id<MTLComputePipelineState> GpuEngine::getCalculateLinearGradPipe() { return calculateLinearGradPipeline; }
id<MTLComputePipelineState> GpuEngine::getCalculateReluGradPipe() { return calculateReluGradPipeline; }

id<MTLComputePipelineState> GpuEngine::getCalculateMSEGradPipe() { return calculateMSEGradPipeline; }
id<MTLComputePipelineState> GpuEngine::getCalculateSoftmaxCrossEntropyGradPipe() { return calculateSoftmaxCrossEntropyGradPipeline; }

id<MTLComputePipelineState> GpuEngine::getActivateLinearPipe() { return activateLinearPipeline; }
id<MTLComputePipelineState> GpuEngine::getActivateReLUPipe() { return activateReLUPipeline; }
id<MTLComputePipelineState> GpuEngine::getActivateSoftmaxPipe() { return activateSoftmaxPipeline; }