// // tests/TestGlobalAveragePooling2D.mm
// // Compare CPU vs GPU for GlobalAveragePooling2D (forward + backprop)

// #include "core/layers/GlobalAveragePooling2D.h"
// #include "core/tensor/Tensor.h"
// #include "core/gpu/GpuEngine.h"

// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>

// #include <cassert>
// #include <cmath>
// #include <random>
// #include <chrono>
// #include <iostream>
// using std::vector;

// static inline bool closeEnough(float a, float b) {
//     const float atol = 1e-7f;   // safe for fp32 GAP
//     const float rtol = 1e-7f;
//     return fabsf(a - b) <= (atol + rtol * fmaxf(fabsf(a), fabsf(b)));
// }

// static void compareTensors(const Tensor& t1, const Tensor& t2, const std::string& name) {
//     const auto &v1 = t1.getFlat(), &v2 = t2.getFlat();
//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Size mismatch in %s: %zu vs %zu\n", name.c_str(), v1.size(), v2.size());
//         assert(false);
//     }
//     for (size_t i = 0; i < v1.size(); ++i) {
//         if (!closeEnough(v1[i], v2[i])) {
//             printf("Mismatch in %s at %zu: CPU=%.7f GPU=%.7f\n", name.c_str(), i, v1[i], v2[i]);
//             assert(false);
//         }
//     }
// }

// // ---------- GAP forward+backprop comparison ----------
// static void testGapCompare(size_t N, size_t H, size_t W, size_t C) {
//     double gpuMs = 0.0;
//     double cpuMs = 0.0;

//     for (size_t t = 0; t < 10; ++t) {
//         vector<size_t> inShape = {N, H, W, C};

//         // Build CPU/GPU layers with identical config
//         GpuEngine::disableGpu();
//         GlobalAveragePooling2D cpuGap;
//         cpuGap.build(inShape, /*isInference=*/false);

//         GpuEngine::enableGpu();
//         GlobalAveragePooling2D gpuGap;
//         gpuGap.build(inShape, /*isInference=*/false);

//         // Random input and upstream gradient
//         Tensor x(inShape);
//         Tensor grad({N, C});
//         std::mt19937 rng(1337u + (unsigned)t);
//         std::uniform_real_distribution<float> dist(-1.f, 1.f);
//         for (auto &v : x.getFlat())   v = dist(rng);
//         for (auto &v : grad.getFlat()) v = dist(rng);
//         x.uploadToGpu();
//         grad.uploadToGpu();

//         // Forward (CPU)
//         GpuEngine::disableGpu();
//         cpuGap.forward(x);

//         // Forward (GPU)
//         GpuEngine::enableGpu();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//         gpuGap.forwardGpu(x, (GpuCommandBuffer)fwdCmd);
//         [fwdCmd commit];
//         [fwdCmd waitUntilCompleted];

//         // Backprop (CPU) — time it
//         GpuEngine::disableGpu();
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuGap.backprop(x, 0.0f, grad, /*isFirstLayer=*/false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         // Backprop (GPU) — time it (includes dispatch + wait)
//         GpuEngine::enableGpu();
//         id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//         auto t2 = std::chrono::high_resolution_clock::now();
//         gpuGap.backpropGpu(x, 0.0f, grad, /*isFirstLayer=*/false, (GpuCommandBuffer)bwdCmd);
//         [bwdCmd commit];
//         [bwdCmd waitUntilCompleted];
//         auto t3 = std::chrono::high_resolution_clock::now();
//         gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

//         // Pull GPU results to host for comparison
//         const_cast<Tensor&>(gpuGap.getOutput()).downloadFromGpu();
//         const_cast<Tensor&>(gpuGap.getOutputGradient()).downloadFromGpu();

//         // Compare forward outputs and dX
//         compareTensors(cpuGap.getOutput(),         gpuGap.getOutput(),         "GAP forward output");
//         compareTensors(cpuGap.getOutputGradient(), gpuGap.getOutputGradient(), "GAP dX");
//     }

//     // Report
//     printf("✅ GAP CPU vs GPU match!\n");
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// // ---------- Driver ----------
// static void testGAP() {
//     // Minimal tensor
//     testGapCompare(1, 1, 1, 1);

//     // Single-channel cases
//     testGapCompare(2, 3, 3, 1);
//     testGapCompare(2, 64, 64, 1);

//     // Odd primes to catch stride math bugs
//     testGapCompare(3, 5, 11, 13);
//     testGapCompare(5, 7, 7, 17);

//     // Non-multiples of 4 or 8 for C (vec tails in future)
//     testGapCompare(3, 3, 3, 65);
//     testGapCompare(6, 14, 14, 63);
//     testGapCompare(3, 7, 7, 257);

//     // Long skinny maps and degenerate spatial dims
//     testGapCompare(1, 128, 1, 32);
//     testGapCompare(1, 1, 128, 32);

//     // Common RGB-ish small C with medium spatial
//     testGapCompare(2, 32, 32, 3);

//     // Batch-heavy small maps
//     testGapCompare(9, 2, 2, 33);

//     // Larger spatial stress with modest C
//     testGapCompare(1, 112, 112, 16);
// }

// int main() {
//     GpuEngine::init();
//     testGAP();
//     return 0;
// }
