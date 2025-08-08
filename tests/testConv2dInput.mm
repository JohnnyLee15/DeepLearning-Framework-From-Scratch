// // testConv2dInput_compare_fixed.mm
// // -----------------------------------------------
// // Compile:
// //   clang++ -ObjC++ testConv2dInput_compare_fixed.mm -framework Metal -framework Foundation -o conv2d_input_test
// // -----------------------------------------------

// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #import <mach/mach_time.h>

// #include "core/gpu/GpuEngine.h"
// #include "core/tensor/Tensor.h"

// #include <random>
// #include <cmath>
// #include <cstdio>

// static double ns2ms(uint64_t dt) {
//     static mach_timebase_info_data_t tb{0,0};
//     if (tb.denom == 0) mach_timebase_info(&tb);
//     return (double)dt * tb.numer / tb.denom / 1.0e6;
// }

// void runCompareInput(const char* name,
//                      size_t batchSize,
//                      size_t height,
//                      size_t width,
//                      size_t inChannels,
//                      size_t outChannels,
//                      size_t kernelSize,
//                      size_t stride)
// {
//     // compute output dims for printing only
//     size_t outH = (height - kernelSize) / stride + 1;
//     size_t outW = (width  - kernelSize) / stride + 1;

//     printf("\n=== %s ===\n", name);
//     printf("dY: [%zu, %zu, %zu, %zu], W: [%zu×%zu×%zu×%zu], stride=%zu\n",
//            batchSize, outH, outW, outChannels,
//            outChannels, kernelSize, kernelSize, inChannels,
//            stride);

//     @autoreleasepool {
//         // --- Create base tensors ---
//         Tensor dY({batchSize, outH, outW, outChannels});
//         Tensor W ({outChannels, kernelSize, kernelSize, inChannels});
//         std::mt19937 gen(444);
//         std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//         for (auto &v : dY.getFlat()) v = dist(gen);
//         for (auto &v : W .getFlat()) v = dist(gen);
//         dY.uploadToGpu();
//         W .uploadToGpu();

//         // Prepare dX on CPU and GPU
//         Tensor dXcpu({batchSize, height, width, inChannels});
//         Tensor dXgpu({batchSize, height, width, inChannels});

//         // Compute padding/upsample window dims using computeGradWindow logic
//         auto padParams = Tensor::decodePadding("none");
//         WindowDims winIn = dY.computeInputWindow(kernelSize, kernelSize, padParams, stride);
//         size_t paddedRows = outH + winIn.padRows;
//         size_t paddedCols = outW + winIn.padCols;
//         WindowDims winGrad = dY.computeGradWindow(
//             kernelSize, kernelSize,
//             paddedRows, paddedCols,
//             stride,
//             winIn
//         );

//         // --- CPU pad-and-upsample + backward ---
//         uint32_t outRows = (stride > 1) ? stride * (height - 1) + 1 : height;
//         uint32_t outCols = (stride > 1) ? stride * (width - 1) + 1 : width;
//         outRows += winGrad.padRows;
//         outCols += winGrad.padCols;
//         Tensor dY_up_cpu({batchSize, outRows, outCols, outChannels});
//         Tensor dY_up_gpu({batchSize, outRows, outCols, outChannels});
        
//         uint64_t c0 = mach_absolute_time();
//         dY.padAndUpsampleGrad(dY_up_cpu, winGrad, stride);
//         dY_up_cpu.conv2dInput(W, dXcpu);
//         uint64_t c1 = mach_absolute_time();
//         double cpuMs = ns2ms(c1 - c0);
//         const auto &cx = dXcpu.getFlat();

//         // --- GPU pad-and-upsample + backward ---
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> cmd = [queue commandBuffer];
//         dY.padAndUpsampleGradGpu(dY_up_gpu, winGrad, stride, cmd);
//         uint64_t t0 = mach_absolute_time();
//         dY_up_gpu.conv2dInputGpu(W, dXgpu, cmd);
//         [cmd commit]; 
//         [cmd waitUntilCompleted];
//         uint64_t t1 = mach_absolute_time();
//         double gpuMs = ns2ms(t1 - t0);
//         dXgpu.downloadFromGpu();
//         const auto &gx = dXgpu.getFlat();

//         // --- Compare accuracy ---
//         size_t N = gx.size();
//         double sumDiff = 0.0;
//         for (size_t i = 0; i < N; ++i) sumDiff += std::fabs(gx[i] - cx[i]);
//         double avgDiff = sumDiff / double(N);

//         // --- Print results ---
//         printf(" CPU time:    %.3f ms\n", cpuMs);
//         printf(" GPU time:    %.3f ms\n", gpuMs);
//         printf(" Sum abs Δ:   %.6f\n", sumDiff);
//         printf(" Avg abs Δ:   %.6f\n", avgDiff);
//     }
// }

// int main() {
//     @autoreleasepool {
//         GpuEngine::init();
//         runCompareInput("Conv2D Input Grad 3×3",     4, 32, 32, 16, 32, 3, 1);
//         runCompareInput("Conv2D Input Grad 5×5",     2, 64, 64,  8, 16, 5, 1);
//         runCompareInput("Conv2D Input Grad Stride2", 2, 64, 64, 16, 24, 3, 2);
//         printf("\n=== Conv2D Input Grad Compare Complete ===\n");
//     }
//     return 0;
// }
