// // test_reduce_sum_bias.mm – Custom Metal ReduceSumBias vs. OpenMP CPU
// // ---------------------------------------------------------------------------
// // Build: clang++ -std=c++17 -ObjC++ test_reduce_sum_bias.mm             \
// //               -framework Metal -framework Foundation                  \
// //               -Xpreprocessor -fopenmp -lomp                           \
// //               -I./
// //
// // Note: This test assumes you have:
// //   1. A `Tensor` class with `reduceSumBias` (CPU) and `reduceSumBiasGpu` (GPU) methods.
// //   2. A `GpuEngine` class to manage the Metal device, command queue, and pipeline states.
// //   3. The necessary class files (Tensor.h, GpuEngine.h, etc.) are in the include path.
// // ---------------------------------------------------------------------------
// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #include <mach/mach_time.h>
// #include <random>
// #include <vector>

// // --- Assumed project headers ---
// #include "core/gpu/GpuEngine.h"
// #include "core/tensor/Tensor.h"

// // Utility to convert mach time units to milliseconds
// static double ns2ms(uint64_t t)
// {
//     static mach_timebase_info_data_t tb{0,0};
//     if(tb.denom==0) mach_timebase_info(&tb);
//     return (double)t*tb.numer/tb.denom/1e6;
// }

// /* ------------------------------------------------------------------ */
// // Test runner for Bias Gradient Reduction (reduce_sum over N,H,W)
// static void runReduceSumBiasTest(const char* tag,
//                                  size_t batchSize, size_t gradRows, size_t gradCols, size_t numKernels,
//                                  size_t runs = 50)
// {
//     printf("\n=== %s (Input: %zux%zux%zux%zu -> Output: %zu) ===\n",
//            tag, batchSize, gradRows, gradCols, numKernels, numKernels);

//     /* ---------------- Tensors ---------------- */
//     // Input gradients have shape (N, H, W, C) where C is numKernels
//     Tensor grad({batchSize, gradRows, gradCols, numKernels});
//     // Output bias gradients have shape (C,)
//     Tensor dB_cpu({numKernels});
//     Tensor dB_gpu({numKernels});

//     // --- Initialize with random data ---
//     std::mt19937 gen(1337);
//     std::uniform_real_distribution<float> dis(-1.f, 1.f);
//     for(float &v: grad.getFlat()) v = dis(gen);

//     /* =========== CPU OpenMP Reduction =========== */
//     // --- warm-up CPU ---
//     for(int i = 0; i < 3; i++) {
//         grad.reduceSumBias(dB_cpu);
//     }

//     // --- timed runs: CPU ---
//     double cpuTotalTime = 0;
//     for(size_t r = 0; r < runs; r++) {
//         uint64_t t0 = mach_absolute_time();
//         grad.reduceSumBias(dB_cpu);
//         uint64_t t1 = mach_absolute_time();
//         cpuTotalTime += ns2ms(t1 - t0);
//     }
//     double cpuAvg = cpuTotalTime / runs;
//     printf("-- CPU (OpenMP) -- avg %.3f ms\n", cpuAvg);


//     /* =========== GPU Metal Reduction =========== */
//     id<MTLCommandQueue> q = GpuEngine::getCmdQueue();
//     grad.uploadToGpu();
//     dB_gpu.uploadToGpu(); // Contents don't matter, will be zeroed by kernel dispatch

//     // --- warm-up GPU kernel ---
//     for(int i = 0; i < 3; i++) {
//         auto cmd = [q commandBuffer];
//         grad.reduceSumBiasGpu(dB_gpu, cmd);
//         [cmd commit];
//         [cmd waitUntilCompleted];
//     }

//     // --- timed runs: GPU ---
//     double gpuTotalTime = 0;
//     for(size_t r = 0; r < runs; r++) {
//         auto cmd = [q commandBuffer];
//         uint64_t t0 = mach_absolute_time();
//         grad.reduceSumBiasGpu(dB_gpu, cmd); // This call should handle the whole operation
//         [cmd commit];
//         [cmd waitUntilCompleted]; // Sync to get accurate wall-clock time
//         uint64_t t1 = mach_absolute_time();
//         gpuTotalTime += ns2ms(t1 - t0);
//     }
//     double gpuAvg = gpuTotalTime / runs;
//     printf("-- GPU (Metal)  -- avg %.3f ms  (%.2fx faster)\n",
//            gpuAvg, cpuAvg / gpuAvg);

//     // --- Verification ---
//     dB_gpu.downloadFromGpu();
//     auto& outCPU = dB_cpu.getFlat();
//     auto& outGPU = dB_gpu.getFlat();

//     // --- compute sum of absolute differences ---
//     double sumAbsDiff = 0.0;
//     for (size_t i = 0; i < numKernels; ++i) {
//         sumAbsDiff += fabs(outCPU[i] - outGPU[i]);
//     }
//     printf("-- Sum abs difference CPU vs GPU: %.6f\n", sumAbsDiff);

//     // --- simple checksum printout ---
//     printf("First 4 CPU : %.4f  %.4f  %.4f  %.4f\n",
//            outCPU[0], outCPU[1], outCPU[2], outCPU[3]);
//     printf("First 4 GPU : %.4f  %.4f  %.4f  %.4f\n",
//            outGPU[0], outGPU[1], outGPU[2], outGPU[3]);
// }
// /* ------------------------------------------------------------------ */
// int main()
// {
//     @autoreleasepool {
//         GpuEngine::init();

//         printf("\n=== Typical CNN Layer Shapes ===\n");
//         runReduceSumBiasTest("Small CNN",    16, 28, 28, 32, 50);
//         runReduceSumBiasTest("Medium CNN",   32, 14, 14, 64, 20);
//         runReduceSumBiasTest("Large CNN",    64,  7,  7, 128, 10);
//         runReduceSumBiasTest("Deep Kernel",  8,   4,  4, 1024, 10);

//         printf("\n=== Boundary Condition Shapes ===\n");
//         // Test dimensions that are not multiples of typical threadgroup sizes
//         runReduceSumBiasTest("Prime Dims",    7, 13, 13, 31, 50);
//         // Test case where reduction dimension (N*H*W) is very large
//         runReduceSumBiasTest("Large Reduce", 128, 32, 32, 16, 5);
//          // Test case where numKernels is small
//         runReduceSumBiasTest("Small Kernels", 64, 16, 16, 4, 50);

//         printf("\n=== Bias gradient reduction benchmark complete ===\n");
//     }
//     return 0;
// }