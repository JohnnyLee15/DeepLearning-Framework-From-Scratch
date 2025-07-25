// // testConv2dForward_compare.mm
// // -----------------------------------------------
// // Compile:
// //   clang++ -ObjC++ testConv2dForward_compare.mm -framework Metal -framework Foundation -o conv2d_forward_test
// // -----------------------------------------------

// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #import <mach/mach_time.h>
// #import "core/gpu/GpuEngine.h"
// #import "core/tensor/Tensor.h"

// #include <random>
// #include <cmath>
// #include <cstdio>

// static double ns2ms(uint64_t dt) {
//     static mach_timebase_info_data_t tb{0,0};
//     if (tb.denom == 0) mach_timebase_info(&tb);
//     return (double)dt * tb.numer / tb.denom / 1.0e6;
// }

// void runCompareForward(const char* name,
//                        size_t batchSize,
//                        size_t height,
//                        size_t width,
//                        size_t inChannels,
//                        size_t outChannels,
//                        size_t kernelSize,
//                        size_t stride)
// {
//     printf("\n=== %s ===\n", name);
//     printf("Input: [%zu, %zu, %zu, %zu], Kernel: [%zux%zu, inC=%zu], Stride: %zu\n",
//            batchSize, height, width, inChannels,
//            kernelSize, kernelSize, inChannels, stride);

//     @autoreleasepool {
//         // Compute output spatial dims (valid convolution)
//         size_t outH = (height - kernelSize) / stride + 1;
//         size_t outW = (width  - kernelSize) / stride + 1;

//         // --- Create tensors ---
//         Tensor X ({batchSize, height, width, inChannels});
//         Tensor W ({outChannels, kernelSize, kernelSize, inChannels});
//         Tensor B ({outChannels});
//         Tensor Y_gpu({batchSize, outH, outW, outChannels});
//         Tensor Y_cpu({batchSize, outH, outW, outChannels});

//         // Fill X, W, B with random data
//         std::mt19937 gen(2025);
//         std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

//         for (auto &v : X .getFlat()) v = dist(gen);
//         for (auto &v : W .getFlat()) v = dist(gen);
//         for (auto &v : B .getFlat()) v = dist(gen);

//         // Zero outputs
//         Y_gpu.zero();
//         Y_cpu.zero();

//         // --- GPU forward + timing --- 
//         X .uploadToGpu();
//         W .uploadToGpu();
//         B .uploadToGpu();
//         Y_gpu.uploadToGpu();

//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();

//         // Warm-up runs
//         for (int i = 0; i < 3; ++i) {
//             id<MTLCommandBuffer> cmd = [queue commandBuffer];
//             X.conv2dForwardGpu(W, stride, Y_gpu, B, cmd);
//             [cmd commit];
//             [cmd waitUntilCompleted];
//         }

//         // Timed GPU run
//         id<MTLCommandBuffer> cmd = [queue commandBuffer];
//         uint64_t t0 = mach_absolute_time();
//         X.conv2dForwardGpu(W, stride, Y_gpu, B, cmd);
//         [cmd commit];
//         [cmd waitUntilCompleted];
//         uint64_t t1 = mach_absolute_time();
//         double gpuMs = ns2ms(t1 - t0);

//         // Download GPU result
//         Y_gpu.downloadFromGpu();
//         const auto &gout = Y_gpu.getFlat();

//         // --- CPU forward + timing ---
//         uint64_t c0 = mach_absolute_time();
//         X.conv2dForward(W, stride, Y_cpu, B);
//         uint64_t c1 = mach_absolute_time();
//         double cpuMs = ns2ms(c1 - c0);
//         const auto &cout = Y_cpu.getFlat();

//         // --- Compare accuracy ---
//         size_t N = gout.size();
//         double sumDiff = 0.0;
//         for (size_t i = 0; i < N; ++i) {
//             sumDiff += std::fabs(gout[i] - cout[i]);
//         }
//         double avgDiff = sumDiff / (double)N;

//         // --- Compute GFLOPS ---
//         long long ops = (long long)batchSize * outH * outW * outChannels
//                        * kernelSize * kernelSize * inChannels * 2;
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

//         runCompareForward("Small Conv 3×3",     5,  32,  32, 16, 32, 3, 1);
//         runCompareForward("Medium Conv 5×5",    6,  64,  64, 32, 64, 5, 1);
//         runCompareForward("Strided Conv 3×3",   32, 128, 128, 64,128, 3, 2);

//         printf("\n=== Conv2D Forward Compare Complete ===\n");
//     }
//     return 0;
// }
