// #include "core/layers/Dense.h"
// #include "core/tensor/Tensor.h"
// #include "core/tensor/Matrix.h"
// #include "core/activations/ReLU.h"
// #include "core/gpu/GpuEngine.h"

// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>

// #include <cassert>
// #include <cmath>
// #include <random>
// #include <chrono>
// #include <iostream>
// using std::vector;

// // ---------- Helpers (same style as your Conv2D suite) ----------

// // Compare two tensors elementwise with a relaxed tolerance for backprop noise
// void compareTensors(const Tensor& t1, const Tensor& t2, const std::string& name) {
//     const auto &v1 = t1.getFlat();
//     const auto &v2 = t2.getFlat();

//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Assertion failed: Size mismatch in tensor '%s'. CPU size: %zu, GPU size: %zu\n",
//                 name.c_str(), v1.size(), v2.size());
//         assert(false);
//     }
//     for (size_t i = 0; i < v1.size(); ++i) {
//         if (fabsf(v1[i] - v2[i]) > 1.0f) {
//             printf("Mismatch in %s at idx %zu: CPU=%.6f GPU=%.6f\n", name.c_str(), i, v1[i], v2[i]);
//             assert(false);
//         }
//     }
// }
// // ---------- Dense backprop comparison ----------

// void testDenseBackpropCompare(size_t N, size_t inFeatures, size_t outFeatures) {
//     double gpuMs = 0.0;
//     double cpuMs = 0.0;

//     for (size_t t = 0; t < 10; ++t) {
//         // 1) Build CPU/GPU layers with identical config
//         vector<size_t> inShape = {N, inFeatures};

//         GpuEngine::disableGpu();
//         Dense cpuLayer(outFeatures, new ReLU());
//         cpuLayer.build(inShape);

//         GpuEngine::enableGpu();
//         Dense gpuLayer(outFeatures, new ReLU());
//         gpuLayer.build(inShape);

//         // 2) Random input and upstream gradient (match output shape)
//         Tensor x(inShape);
//         Tensor grad({N, outFeatures});

//         std::mt19937 rng(42 + (unsigned)t);
//         std::uniform_real_distribution<float> dist(-1.f, 1.f);
//         for (auto &v : x.getFlat())   v = dist(rng);
//         for (auto &v : grad.getFlat()) v = dist(rng);
//         x.uploadToGpu();
//         grad.uploadToGpu();

//         // 3) Forward (to set internal caches)
//         GpuEngine::disableGpu();
//         cpuLayer.forward(x);

//         GpuEngine::enableGpu();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//         gpuLayer.forwardGpu(x, (GpuCommandBuffer)fwdCmd);
//         [fwdCmd commit];
//         [fwdCmd waitUntilCompleted];

//         // 4) CPU backprop timing
//         GpuEngine::disableGpu();
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuLayer.backprop(x, 0.01f, grad, /*isInference=*/false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         // 5) GPU backprop timing
//         GpuEngine::enableGpu();
//         id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//         auto t2 = std::chrono::high_resolution_clock::now();
//         gpuLayer.backpropGpu(x, 0.01f, grad, /*isInference=*/false, (GpuCommandBuffer)bwdCmd);
//         [bwdCmd commit];
//         [bwdCmd waitUntilCompleted];
//         auto t3 = std::chrono::high_resolution_clock::now();
//         gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

//         // 6) Pull GPU results to host
//         const_cast<Tensor&>(gpuLayer.getWeights()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getBiases()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getDeltaInputs()).downloadFromGpu();

//         // // 7) Compare: dW, dB, dX
//         // compareTensors(cpuLayer.getWeights(),       gpuLayer.getWeights(),         "dW");
//         // compareTensors(cpuLayer.getBiases(),        gpuLayer.getBiases(),   "biases");
//         // compareTensors(cpuLayer.getDeltaInputs(),   gpuLayer.getDeltaInputs(), "dX");
//     }

//     // 8) Report
//     printf("✅ Dense backprop CPU vs GPU match!\n");
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// // ---------- Driver ----------

// void testDense() {
//     // Small
//     testDenseBackpropCompare(1,   4,   4);
//     testDenseBackpropCompare(2,   8,   8);

//     // Medium
//     testDenseBackpropCompare(4,  64,  64);
//     testDenseBackpropCompare(8, 128, 128);

//     // Rectangular
//     testDenseBackpropCompare(8,  64, 256);
//     testDenseBackpropCompare(8, 256,  64);

//     // Larger batches
//     testDenseBackpropCompare(16, 256, 256);
//     testDenseBackpropCompare(32, 512, 512);

//     // Very Larger batches
//     testDenseBackpropCompare(64, 4096, 256);
//     testDenseBackpropCompare(128, 8192, 512);
// }

// int main() {
//     GpuEngine::init();
//     testDense();
//     return 0;
// }
