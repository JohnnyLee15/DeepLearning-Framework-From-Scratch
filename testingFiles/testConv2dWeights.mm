// // testConv2dWeights_compare.mm
// // -----------------------------------------------
// // Compile:
// //   clang++ -ObjC++ testConv2dWeights_compare.mm -framework Metal -framework Foundation -o conv2d_weights_test
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

// void runCompareWeights(const char* name,
//                        size_t batchSize,
//                        size_t height,
//                        size_t width,
//                        size_t inChannels,
//                        size_t outChannels,
//                        size_t kernelSize,
//                        size_t stride)
// {
//     printf("\n=== %s ===\n", name);
//     printf("Input: [%zu, %zu, %zu, %zu], Grad: [%zu×%zu stride %zu → outC=%zu], WeightsΔ: [%zu×%zu×%zu×%zu]\n",
//            batchSize, height, width, inChannels,
//            kernelSize, kernelSize, stride, outChannels,
//            outChannels, kernelSize, kernelSize, inChannels);

//     @autoreleasepool {
//         // Compute output spatial dims
//         size_t outH = (height - kernelSize) / stride + 1;
//         size_t outW = (width  - kernelSize) / stride + 1;

//         // --- Create tensors ---
//         Tensor X   ({batchSize, height, width, inChannels});
//         Tensor dY  ({batchSize, outH,        outW,      outChannels});
//         Tensor dW_cpu({outChannels, kernelSize, kernelSize, inChannels});
//         Tensor dW_gpu({outChannels, kernelSize, kernelSize, inChannels});

//         // Fill X and dY with random data
//         std::mt19937 gen(123);
//         std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//         for (auto &v : X.getFlat())  v = dist(gen);
//         for (auto &v : dY.getFlat()) v = dist(gen);

//         // Zero‐initialize both weight‐grad tensors
//         dW_cpu.zero();
//         dW_gpu.zero();

//         // --- GPU weight‐gradient + timing ---
//         X.uploadToGpu();
//         dY.uploadToGpu();
//         dW_gpu.uploadToGpu();

//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();

//         // warmup
//         for (int i = 0; i < 3; ++i) {
//             id<MTLCommandBuffer> cmd = [queue commandBuffer];
//             X.conv2dWeightsGpu(dY, outChannels,
//                                kernelSize, kernelSize,
//                                stride, dW_gpu, cmd);
//             [cmd commit];
//             [cmd waitUntilCompleted];
//         }

//         // timed run
//         id<MTLCommandBuffer> cmd = [queue commandBuffer];
//         uint64_t t0 = mach_absolute_time();
//         X.conv2dWeightsGpu(dY, outChannels,
//                            kernelSize, kernelSize,
//                            stride, dW_gpu, cmd);
//         [cmd commit];
//         [cmd waitUntilCompleted];
//         uint64_t t1 = mach_absolute_time();
//         double gpuMs = ns2ms(t1 - t0);

//         // retrieve GPU result
//         dW_gpu.downloadFromGpu();
//         const auto &gw = dW_gpu.getFlat();

//         // --- CPU weight‐gradient ---
//         uint64_t c0 = mach_absolute_time();
//         X.conv2dWeights(dY, outChannels,
//                         kernelSize, kernelSize,
//                         stride, dW_cpu);
//         uint64_t c1 = mach_absolute_time();
//         double cpuMs = ns2ms(c1 - c0);
//         const auto &cw = dW_cpu.getFlat();

//         // --- Compare accuracy ---
//         size_t N = gw.size();
//         double sumDiff = 0.0;
//         for (size_t i = 0; i < N; ++i) {
//             sumDiff += std::fabs(gw[i] - cw[i]);
//         }
//         double avgDiff = sumDiff / (double)N;

//         // --- Compute GFLOPS ---
//         // For weights gradient: each output grad multiplies input patch → inC*kernelSize*kernelSize MACs,
//         // and summed over batch and spatial dims. Count 2 ops per MAC.
//         long long ops = (long long)batchSize * outH * outW * outChannels
//                        * inChannels * kernelSize * kernelSize * 2;
//         double gflops = ops / (gpuMs * 1e6);

//         // --- Print results ---
//         printf(" GPU time:    %.3f ms\n", gpuMs);
//         printf(" CPU time:    %.3f ms\n", cpuMs);
//         printf(" GFLOPS:      %.2f\n", gflops);
//         printf(" Sum abs Δ:   %.6f\n", sumDiff);
//         printf(" Avg abs Δ:   %.6f\n", avgDiff);
//     }
// }

// int main() {
//     @autoreleasepool {
//         GpuEngine::init();
//         runCompareWeights("Conv2D Weights Grad 3×3", 4, 32, 32, 16, 32, 3, 1);
//         runCompareWeights("Conv2D Weights Grad 5×5", 2, 64, 64,  8, 16, 5, 1);
//         runCompareWeights("Conv2D Weights Grad Stride2", 2, 64, 64, 16, 24, 3, 2);

//         printf("\n=== Conv2D Weights Grad Compare Complete ===\n");
//     }
//     return 0;
// }
