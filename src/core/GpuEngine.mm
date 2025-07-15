#include "utils/ConsoleUtils.h"
#include "core/GpuEngine.h"
 #include <iostream>

// Static vars
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

id<MTLComputePipelineState> GpuEngine::calculateMSEGradPipeline = nil;
id<MTLComputePipelineState> GpuEngine::calculateSoftmaxCrossEntropyGradPipeline  = nil;
id<MTLCommandBuffer> GpuEngine::lastCmdBuf = nil;

bool GpuEngine::usingGpu = false;

bool GpuEngine::isUsingGpu() {
    return usingGpu;
}

void GpuEngine::init() {
    gpuDevice = MTLCreateSystemDefaultDevice();
    cmdQueue = [gpuDevice newCommandQueue];
    initLib();
    initAllPipes();
    usingGpu = true;
    cout << "IN gpu init" << endl;
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

    initPipe("calculateMSEGrad", calculateMSEGradPipeline);
    initPipe("calculateSoftmaxCrossEntropyGrad", calculateSoftmaxCrossEntropyGradPipeline);
}

id<MTLDevice> GpuEngine::getGpuDevice() { 
    return gpuDevice; 
}

id<MTLCommandQueue> GpuEngine::getCmdQueue() { 
    return cmdQueue; 
}

id<MTLLibrary> GpuEngine::getLib() { 
    return defaultLib; 
}


id<MTLComputePipelineState> GpuEngine::getMatMatPipe() { 
    return matMatPipeline; 
}

id<MTLComputePipelineState> GpuEngine::getMatMatTPipe() { 
    return matMatTPipeline; 
}

id<MTLComputePipelineState> GpuEngine::getMatTMatPipe() { 
    return matTMatPipeline; 
}

id<MTLComputePipelineState> GpuEngine::getMatTMatTPipe() { 
    return matTMatTPipeline; 
}


id<MTLComputePipelineState> GpuEngine::getColSumsPipe() { 
    return colSumsPipeline; 
}
id<MTLComputePipelineState> GpuEngine::getAddToRowsPipe() { 
    return addToRowsPipeline; 
}


id<MTLComputePipelineState> GpuEngine::getHadamardPipe() { 
    return hadamardPipeline; 
}

id<MTLComputePipelineState> GpuEngine::getApplyGradPipe() {
    return applyGradPipeline;
}


id<MTLComputePipelineState> GpuEngine::getCalculateLinearGradPipe() { 
    return calculateLinearGradPipeline;
}

id<MTLComputePipelineState> GpuEngine::getCalculateReluGradPipe() { 
    return calculateReluGradPipeline;
}


id<MTLComputePipelineState> GpuEngine::getCalculateMSEGradPipe() { 
    return calculateMSEGradPipeline;
}

id<MTLComputePipelineState> GpuEngine::getCalculateSoftmaxCrossEntropyGradPipe() { 
    return calculateSoftmaxCrossEntropyGradPipeline;
}


void GpuEngine::setLastCmdBuf(id<MTLCommandBuffer> lastBuf) {
    lastCmdBuf = lastBuf;
}

id<MTLCommandBuffer> GpuEngine::getLastCmdBuf() {
    return lastCmdBuf;
}
