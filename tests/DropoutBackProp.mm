// // tests/test_dropout_gpu.mm
// #include "core/layers/Dropout.h"
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

// // ---------- Helpers ----------
// static void compareTensors(const Tensor& t1, const Tensor& t2, const char* name, float tol=1e-6f) {
//     const auto &v1 = t1.getFlat(), &v2 = t2.getFlat();
//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Assertion failed: Size mismatch in '%s'. CPU size: %zu, GPU size: %zu\n",
//                 name, v1.size(), v2.size());
//         assert(false);
//     }
//     for (size_t i = 0; i < v1.size(); ++i) {
//         float d = std::fabs(v1[i] - v2[i]);
//         if (d > tol) {
//             printf("Mismatch in %s at %zu: CPU=%.6f GPU=%.6f (|Δ|=%.3g)\n", name, i, v1[i], v2[i], d);
//             assert(false);
//         }
//     }
// }

// static void fillUniform(Tensor& t, uint32_t seed, float lo=-1.f, float hi=1.f) {
//     std::mt19937 rng(seed);
//     std::uniform_real_distribution<float> dist(lo, hi);
//     for (auto &x : t.getFlat()) x = dist(rng);
// }

// // ---------- Dropout forward/backward comparison ----------
// static void testDropoutCompare(const vector<size_t>& inShape, float p) {
//     double cpuMs = 0.0;
//     double gpuMs = 0.0;

//     for (int iter = 0; iter < 10; ++iter) {
//         // 1) Build CPU/GPU layers with identical config (training mode)
//         GpuEngine::disableGpu();
//         Dropout cpuLayer(p);
//         cpuLayer.build(inShape, /*isInference=*/false);

//         GpuEngine::enableGpu();
//         Dropout gpuLayer(p);
//         gpuLayer.build(inShape, /*isInference=*/false);

//         // 2) Random input and upstream grad (same shape as input for Dropout)
//         Tensor x(inShape);
//         Tensor grad(inShape);
//         fillUniform(x,   42u + 31u*iter);
//         fillUniform(grad,99u + 17u*iter);

//         x.uploadToGpu();
//         grad.uploadToGpu();

//         // 3) Forward to set internal outputs
//         GpuEngine::disableGpu();
//         cpuLayer.forward(x);

//         GpuEngine::enableGpu();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//         gpuLayer.forwardGpu(x, (GpuCommandBuffer)fwdCmd); // if your API takes id<MTLCommandBuffer>, pass fwdCmd directly
//         [fwdCmd commit];
//         [fwdCmd waitUntilCompleted];

//         // 4) CPU backprop timing
//         GpuEngine::disableGpu();
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuLayer.backprop(x, /*lr*/0.0f, grad, /*isFirstLayer=*/false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         // 5) GPU backprop timing
//         GpuEngine::enableGpu();
//         id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//         auto t2 = std::chrono::high_resolution_clock::now();
//         gpuLayer.backpropGpu(x, /*lr*/0.0f, grad, /*isInference=*/false, (GpuCommandBuffer)bwdCmd);
//         [bwdCmd commit];
//         [bwdCmd waitUntilCompleted];
//         auto t3 = std::chrono::high_resolution_clock::now();
//         gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

//         // 6) Pull GPU results to host
//         const_cast<Tensor&>(gpuLayer.getOutput()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getOutputGradient()).downloadFromGpu();

//         // 7) Compare: output (y) and dX
//         compareTensors(cpuLayer.getOutput(),         gpuLayer.getOutput(),         "dropout forward output");
//         compareTensors(cpuLayer.getOutputGradient(), gpuLayer.getOutputGradient(), "dropout dX");
//     }

//     // 8) Report
//     printf("✅ Dropout CPU vs GPU match | shape=[");
//     for (size_t i=0;i<inShape.size();++i) printf("%zu%s", inShape[i], (i+1<inShape.size()? ",":""));
//     printf("], p=%.2f\n", p);
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// // ---------- Driver ----------
// int main() {
//     GpuEngine::init();

//     // 2D (N,F)
//     testDropoutCompare({1,   1024}, 0.0f);
//     testDropoutCompare({8,   4096}, 0.3f);
//     testDropoutCompare({32,  8192}, 0.5f);
//     testDropoutCompare({64, 16384}, 0.7f);

//     // 4D (N,H,W,C)
//     testDropoutCompare({8,  32, 32,  64}, 0.3f);
//     testDropoutCompare({16, 14, 14, 128}, 0.5f);
//     testDropoutCompare({8,   7,  7, 512}, 0.5f);

//     return 0;
// }
