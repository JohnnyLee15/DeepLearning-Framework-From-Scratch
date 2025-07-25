// // testMaxPooling2D_BackpropCpuVsGpu.mm
// // Compile:
// // clang++ -ObjC++ testMaxPooling2D_BackpropCpuVsGpu.mm -framework Metal -framework Foundation -o maxpool_backprop_test

// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #import <mach/mach_time.h>

// #include "core/gpu/GpuEngine.h"
// #include "core/layers/MaxPooling2D.h"
// #include "core/tensor/Tensor.h"

// #include <random>
// #include <cmath>
// #include <cstdio>

// static double ns2ms(uint64_t dt) {
//     static mach_timebase_info_data_t tb{0,0};
//     if (tb.denom == 0) mach_timebase_info(&tb);
//     return (double)dt * tb.numer / tb.denom / 1.0e6;
// }

// void runMaxPoolBackpropTest() {
//     printf("\n=== MaxPooling2D Backprop CPU vs GPU ===\n");

//     // Test Parameters
//     const size_t batch = 8;
//     const size_t height = 32;
//     const size_t width = 32;
//     const size_t depth = 8;
//     const size_t kRows = 2;
//     const size_t kCols = 2;
//     const size_t stride = 2;
//     const std::string padding = "same";

//     // Initialize random inputs
//     Tensor input({batch, height, width, depth});
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//     for (auto &v : input.getFlat()) v = dist(gen);
//     input.uploadToGpu();

//     // Initialize MaxPooling layers
//     MaxPooling2D poolCpu(kRows, kCols, stride, padding);
//     MaxPooling2D poolGpu(kRows, kCols, stride, padding);
//     poolCpu.build(input.getShape());
//     poolGpu.build(input.getShape());

//     // Forward pass (CPU)
//     poolCpu.forward(input);
//     Tensor gradUp(poolCpu.getOutput().getShape());
//     for (auto &v : gradUp.getFlat()) v = dist(gen);
//     gradUp.uploadToGpu();

//     // Backprop CPU
//     uint64_t cpuStart = mach_absolute_time();
//     poolCpu.backprop(input, 0.0f, gradUp, false);
//     uint64_t cpuEnd = mach_absolute_time();
//     double cpuTime = ns2ms(cpuEnd - cpuStart);
//     const auto &cpuDX = poolCpu.getOutputGradient().getFlat();

//     // GPU Warm-up
//     id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//     id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
//     poolGpu.forwardGpu(input, (__bridge void*)cmdBuf);
//     Tensor gradUpGpu = gradUp;
//     uint64_t gpuStart = mach_absolute_time();
//     poolGpu.backpropGpu(input, 0.0f, gradUpGpu, false, (__bridge void*)cmdBuf);
//     [cmdBuf commit]; [cmdBuf waitUntilCompleted];
//     uint64_t gpuEnd = mach_absolute_time();
//     double gpuTime = ns2ms(gpuEnd - gpuStart);
    
//     poolGpu.getOutputGradient().downloadFromGpu();
//     const auto &gpuDX = poolGpu.getOutputGradient().getFlat();

//     // Accuracy Check
//     double sumDiff = 0.0, maxDiff = 0.0;
//     for (size_t i = 0; i < cpuDX.size(); ++i) {
//         double diff = std::fabs(cpuDX[i] - gpuDX[i]);
//         sumDiff += diff;
//         maxDiff = std::max(maxDiff, diff);
//     }

//     printf("CPU time: %.3f ms\n", cpuTime);
//     printf("GPU time: %.3f ms\n", gpuTime);
//     printf("Output size: %zu\n", cpuDX.size());
//     printf("Avg abs diff: %.6f\n", sumDiff / cpuDX.size());
//     printf("Max abs diff: %.6f\n", maxDiff);
// }

// int main() {
//     @autoreleasepool {
//         GpuEngine::init();
//         runMaxPoolBackpropTest();
//         printf("\n=== Test Completed ===\n");
//     }
//     return 0;
// }