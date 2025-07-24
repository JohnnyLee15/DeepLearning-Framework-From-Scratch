// // testConv2dWeights_benchmark.mm
// // ----------------------------------------------------
// // Compile: clang++ -ObjC++ testConv2dWeights_benchmark.mm \
// //          -framework Metal -framework Foundation
// // ----------------------------------------------------
// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #include <mach/mach_time.h>
// #include "core/gpu/GpuEngine.h"
// #include "core/tensor/Tensor.h"
// #include <random>

// static double ns2ms(uint64_t dt)
// {
//     static mach_timebase_info_data_t tb{0,0};
//     if (tb.denom == 0) mach_timebase_info(&tb);
//     return (double)dt * tb.numer / tb.denom / 1e6;   // → ms
// }

// void runWeightsBenchmark(const char* name,
//                          size_t batchSize, size_t height, size_t width,
//                          size_t inChannels, size_t outChannels,
//                          size_t kSize, size_t stride,
//                          size_t runs = 100)
// {
//     printf("\n=== %s ===\n", name);
//     printf("Input   : [%zu, %zu, %zu, %zu]\n", batchSize, height, width, inChannels);
//     printf("Kernel  : [%zu, %zu, %zu, %zu]\n", outChannels, kSize, kSize, inChannels);
//     printf("Stride  : %zu\n", stride);

//     @autoreleasepool
//     {
//         /* -------------  Tensor shapes ------------------------------ */
//         Tensor X ({batchSize, height, width, inChannels});                  // activations
//         Tensor dY({batchSize,
//                    (height - kSize)/stride + 1,
//                    (width  - kSize)/stride + 1,
//                    outChannels});                                           // output-grad
//         Tensor dW({outChannels, kSize, kSize, inChannels});                 // weight-grad

//         printf("dY shape : [%zu, %zu, %zu, %zu]\n",
//                dY.getShape()[0], dY.getShape()[1], dY.getShape()[2], dY.getShape()[3]);
//         printf("dW shape : [%zu, %zu, %zu, %zu]\n",
//                dW.getShape()[0], dW.getShape()[1], dW.getShape()[2], dW.getShape()[3]);

//         /* -------------  Random initialisation ---------------------- */
//         std::mt19937 rng(1234);
//         std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

//         auto fill = [&](Tensor& T)
//         {
//             auto& vec = T.getFlat();
//             vec.resize(T.getSize());
//             for (auto& v : vec) v = uni(rng);
//         };

//         fill(X);
//         fill(dY);
//         std::fill(dW.getFlat().begin(), dW.getFlat().end(), 0.f);   // clear output

//         /* -------------  Upload to GPU ------------------------------ */
//         X.uploadToGpu();
//         dY.uploadToGpu();
//         dW.uploadToGpu();

//         /* -------------  Warm-up ------------------------------------ */
//         id<MTLCommandQueue> Q = GpuEngine::getCmdQueue();
//         for (int i = 0; i < 5; ++i) {
//             id<MTLCommandBuffer> cmd = [Q commandBuffer];
//             X.conv2dWeightsGpu(dY, outChannels, kSize, kSize, stride, dW, cmd);
//             [cmd commit];
//             [cmd waitUntilCompleted];
//         }

//         /* -------------  Timed runs --------------------------------- */
//         double totMs = 0, minMs = 1e9, maxMs = 0;

//         for (size_t r = 0; r < runs; ++r) {
//             uint64_t t0 = mach_absolute_time();

//             id<MTLCommandBuffer> cmd = [Q commandBuffer];
//             X.conv2dWeightsGpu(dY, outChannels, kSize, kSize, stride, dW, cmd);
//             [cmd commit];
//             [cmd waitUntilCompleted];

//             uint64_t t1 = mach_absolute_time();
//             double ms = ns2ms(t1 - t0);

//             totMs += ms;
//             minMs  = std::min(minMs, ms);
//             maxMs  = std::max(maxMs, ms);

//             if (r < 3 || r % 20 == 0)
//                 printf("Run %3zu: %.3f ms\n", r, ms);
//         }

//         double avgMs = totMs / runs;

//         /* -------------  Stats -------------------------------------- */
//         size_t outH = dY.getShape()[1], outW = dY.getShape()[2];
//         long long ops = (long long)batchSize * outH * outW * outChannels *
//                         kSize * kSize * inChannels * 2;           // MAC
//         double gflops = ops / (avgMs * 1e6);

//         long long bytes =
//               (long long)X.getSize()
//             + (long long)dY.getSize()
//             + (long long)dW.getSize();
//         double gbps = bytes * sizeof(float) / (avgMs * 1e6);

//         printf("\nResults:\n");
//         printf("  Avg   : %.3f ms\n", avgMs);
//         printf("  Min   : %.3f ms\n", minMs);
//         printf("  Max   : %.3f ms\n", maxMs);
//         printf("  GFLOPS: %.2f\n", gflops);
//         printf("  GB/s  : %.2f\n", gbps);

//         /* -------------  Spot-check output -------------------------- */


//         /* ----------  CPU reference & verification -------------------- */
//         Tensor dW_cpu(dW.getShape());   // allocate same shape as dW
//         dW_cpu.zero();                  // fills with 0 and allocates GPU buffer (if needed)

//         X.conv2dWeights(dY,             // forward-grad tensor
//                         outChannels,    // numKernals
//                         kSize, kSize,   // kRows, kCols
//                         stride,
//                         dW_cpu);        // output on CPU

//         dW.downloadFromGpu();


//         const auto& gpu = dW.getFlat();       // vectors are now on host
//         const auto& cpu = dW_cpu.getFlat();

//         printf("First 4 dW values: [%.4f, %.4f, %.4f, %.4f]\n",
//                gpu[0], gpu[1], gpu[2], gpu[3]);
//         printf("First 4 dW values: [%.4f, %.4f, %.4f, %.4f]\n",
//             cpu[0], cpu[1], cpu[2], cpu[3]);

//         double maxErr = 0.0, meanErr = 0.0;
//         size_t maxIdx = -1;
//         for (std::size_t i = 0; i < gpu.size(); ++i) {
//             double e = std::fabs(gpu[i] - cpu[i]);
//             maxErr   = std::max(maxErr, e);
//             if (maxErr == e) {
//                 maxIdx = i;
//             }
//             meanErr += e;
//         }
//         meanErr /= gpu.size();

//         printf("  Max |Δ| : %.6e   Mean |Δ| : %.6e\n", maxErr, meanErr);
//         printf("Difference: %.6f, %.6f\n", cpu[maxIdx], gpu[maxIdx]);
//     }
// }

// int main()
// {
//     @autoreleasepool
//     {
//         GpuEngine::init();

//         /* Same test matrix style as forward benchmark */
//         runWeightsBenchmark("Small k=3",   1, 32,  32,  16, 32, 3, 1);
//         runWeightsBenchmark("Medium k=3",  1, 64,  64,  32, 64, 3, 1);
//         runWeightsBenchmark("Large k=3",   1, 128, 128, 64, 128, 3, 1);
//         runWeightsBenchmark("Batch k=3",   4, 64,  64,  32, 64, 3, 1);
//         runWeightsBenchmark("Kernel 5×5",  1, 64,  64,  32, 64, 5, 1);
//         runWeightsBenchmark("Kernel 7×7",  1, 64,  64,  32, 64, 7, 1);
//         runWeightsBenchmark("Stride 2",    1, 128, 128, 64, 128, 3, 2);
//         runWeightsBenchmark("Deep block",  1, 56,  56, 128, 256, 3, 1);
//         printf("\n=== Weight-gradient Benchmark Complete ===\n");
//     }
//     return 0;
// }
